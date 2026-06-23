import contextlib
import datetime
import glob
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb

import modules
from sft_chat_templetes import (
    HiRALinear,
    encode_chat_example,
    format_chat_prompt,
    get_hira_ab_norm as hira_ab_norm,
    get_hira_update_ratio as hira_update_ratio,
    load_tokenizer,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# Release SFT Configuration
# =========================

RUN_STAGE = "assistant_sft_stage3f_v4_first_token_audit"
SEED = 1337

DATA_PATTERN = "./data/re_sft_assistant_stage3_mode_boundary.jsonl"
INIT_CKPT_PATH = "./data/shengoovlei_assistant_sft_stage3e_v4_mode_boundary_final.pt"
RESUME_OPTIMIZER = False

CHECKPOINT_DIR = "./checkpoints"
FINAL_CKPT_PATH = "./data/shengoovlei_assistant_sft_stage3f_v4_first_token_audit_final.pt"
CHECKPOINT_FORMAT = "shengoovlei_assistant_sft_v1"

WANDB_PROJECT = "shengoovlei_sft_release"
WANDB_ENTITY = None
WANDB_RUN_NAME = f"{RUN_STAGE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
WANDB_API_KEY_FILE = "./pswd.json"

MODEL_CONTEXT_LENGTH = 1024
BATCH_SIZE = 64
MAX_ITERS = 0
MAX_LR = 3e-6
MIN_LR = 1e-6
WARMUP_ITERS = 50
MAX_GRAD_NORM = 1.0

HIRA_R = 32
HIRA_ALPHA = 32
HIRA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")
TRAIN_FULL_LM_HEAD = True
TRAIN_ONLY_SPECIAL_EMBEDDINGS = True

LOG_INTERVAL = 50
EVAL_INTERVAL = 100
EVAL_ITERS = 3
EVAL_EXAMPLES_PER_TASK = 128
GREEDY_EVAL_INTERVAL = 500
GREEDY_EXAMPLES_PER_TASK = 32
SAMPLE_INTERVAL = 500
CHECKPOINT_INTERVAL = 500
MAX_GENERATE_TOKENS = 128

VAL_FRAC = 0.10
TASK_WEIGHTS = {
    "repeat": 0.25,
    "yesno": 0.10,
    "short_qa": 0.00,
    "assistant_qa": 0.60,
    "identity": 0.05,
    "general_short": 0.00,
    "count": 0.00,
}

IGNORE_INDEX = -666
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]
EXACT_MATCH_TASKS = {"repeat", "identity", "yesno", "short_qa", "general_short", "count"}
OPEN_GENERATION_TASKS = {"assistant_qa"}


def release_config():
    return {
        "run_stage": RUN_STAGE,
        "seed": SEED,
        "data_pattern": DATA_PATTERN,
        "init_ckpt_path": INIT_CKPT_PATH,
        "resume_optimizer": RESUME_OPTIMIZER,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_context_length": MODEL_CONTEXT_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_iters": MAX_ITERS,
        "max_lr": MAX_LR,
        "min_lr": MIN_LR,
        "warmup_iters": WARMUP_ITERS,
        "max_grad_norm": MAX_GRAD_NORM,
        "hira_r": HIRA_R,
        "hira_alpha": HIRA_ALPHA,
        "hira_targets": HIRA_TARGETS,
        "train_full_lm_head": TRAIN_FULL_LM_HEAD,
        "train_only_special_embeddings": TRAIN_ONLY_SPECIAL_EMBEDDINGS,
        "task_weights": TASK_WEIGHTS,
        "val_frac": VAL_FRAC,
    }


class TaskSampler:
    def __init__(self, examples):
        self.pools = defaultdict(list)
        for example in examples:
            self.pools[example["task"]].append(example)

        active = {task: weight for task, weight in TASK_WEIGHTS.items() if weight > 0 and self.pools[task]}
        if not active:
            active = {task: 1.0 for task, pool in self.pools.items() if pool}

        self.tasks = list(active)
        self.weights = [active[task] for task in self.tasks]

    def sample(self):
        task = random.choices(self.tasks, weights=self.weights, k=1)[0]
        return random.choice(self.pools[task])


def make_batch(source, device, tokenizer, batch_size=BATCH_SIZE):
    items = []
    for row in range(batch_size):
        example = source.sample() if isinstance(source, TaskSampler) else source[row]
        prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)
        if len(full_tokens) > MODEL_CONTEXT_LENGTH:
            raise RuntimeError(f"example too long after filtering: {len(full_tokens)} tokens")
        items.append((example["task"], prompt_tokens, full_tokens))

    seq_len = max(len(full_tokens) for _, _, full_tokens in items)
    pad_id = tokenizer.vocab_inv["<|pad|>".encode("utf-8")]

    x = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    tasks = []
    supervised_tokens = 0

    for row, (task, prompt_tokens, full_tokens) in enumerate(items):
        labels = [IGNORE_INDEX] * len(full_tokens)
        labels[len(prompt_tokens):] = full_tokens[len(prompt_tokens):]
        supervised_tokens += len(full_tokens) - len(prompt_tokens)
        tasks.append(task)

        x[row, :len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long, device=device)
        y[row, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)

    return x, y, supervised_tokens, tasks


def sft_loss(logits, labels, tasks):
    B, S, V = logits.shape
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    mask = labels != IGNORE_INDEX

    losses = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(B, S - 1)

    counts = mask.sum(dim=1).clamp_min(1)
    losses = (losses * mask).sum(dim=1) / counts
    task_losses = []
    task_weights = []

    for task in sorted(set(tasks)):
        indices = [i for i, name in enumerate(tasks) if name == task]
        task_losses.append(losses[torch.tensor(indices, dtype=torch.long, device=losses.device)].mean())
        task_weights.append(float(TASK_WEIGHTS.get(task, 1.0)))

    weight_tensor = torch.tensor(task_weights, dtype=losses.dtype, device=losses.device)
    if weight_tensor.sum() <= 0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    return torch.stack(task_losses).mul(weight_tensor).sum()


@torch.no_grad()
def estimate_loss(model, sampler, device, tokenizer, autocast):
    model.eval()
    losses = torch.zeros(EVAL_ITERS, device=device)
    for i in range(EVAL_ITERS):
        x, y, _, tasks = make_batch(sampler, device, tokenizer)
        with autocast:
            losses[i] = sft_loss(model(x), y, tasks)
    model.train()
    return losses.mean().item()


@torch.no_grad()
def teacher_forced_scores(model, examples, device, tokenizer, autocast):
    model.eval()
    by_task = defaultdict(list)
    for example in examples:
        if len(by_task[example["task"]]) < EVAL_EXAMPLES_PER_TASK:
            by_task[example["task"]].append(example)

    exact_scores = {}
    token_scores = {}
    for task, task_examples in sorted(by_task.items()):
        exact_correct = 0
        token_correct = 0
        token_total = 0
        for start in range(0, len(task_examples), BATCH_SIZE):
            batch_examples = task_examples[start:start + BATCH_SIZE]
            x, y, _, _ = make_batch(batch_examples, device, tokenizer, batch_size=len(batch_examples))
            with autocast:
                logits = model(x)

            pred = logits[:, :-1, :].argmax(dim=-1)
            target = y[:, 1:]
            mask = target != IGNORE_INDEX
            for row in range(len(batch_examples)):
                positions = torch.nonzero(mask[row], as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                row_correct = pred[row, positions] == target[row, positions]
                token_correct += int(row_correct.sum().item())
                token_total += int(positions.numel())
                exact_correct += int(torch.equal(pred[row, positions], target[row, positions]))

        if task in EXACT_MATCH_TASKS:
            exact_scores[task] = (exact_correct / len(task_examples), len(task_examples))
        if task in OPEN_GENERATION_TASKS or task in EXACT_MATCH_TASKS:
            token_scores[task] = (token_correct / token_total if token_total else 0.0, token_total)

    model.train()
    return exact_scores, token_scores


@torch.no_grad()
def greedy_answer(model, example, tokenizer):
    end_id = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]
    prompt_tokens = tokenizer.encode(format_chat_prompt(example["instruction"]))
    output_ids = modules.generating(
        model=model,
        enc_user_prompt=prompt_tokens,
        end_token=end_id,
        context_len=MODEL_CONTEXT_LENGTH,
        max_token=MAX_GENERATE_TOKENS,
        do_sample=False,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    )
    text = tokenizer.decode(output_ids.tolist() if hasattr(output_ids, "tolist") else output_ids)
    return text.split("<|endoftext|>", 1)[0].strip()


@torch.no_grad()
def greedy_scores(model, examples, tokenizer):
    model.eval()
    by_task = defaultdict(list)
    for example in examples:
        if len(by_task[example["task"]]) < GREEDY_EXAMPLES_PER_TASK:
            by_task[example["task"]].append(example)

    scores = {}
    for task, task_examples in sorted(by_task.items()):
        if task in OPEN_GENERATION_TASKS:
            correct = 0
            for example in task_examples:
                text = greedy_answer(model, example, tokenizer)
                words = text.strip().split()
                lowered = text.lower()
                grams = [" ".join(words[i:i + 3]).lower() for i in range(max(0, len(words) - 2))]
                correct += int(
                    4 <= len(words) <= 90
                    and "shengoovlei" not in lowered
                    and "i don't know" not in lowered
                    and "as an ai" not in lowered
                    and "sorry" not in lowered
                    and "not sure" not in lowered
                    and len(grams) == len(set(grams))
                )
        else:
            correct = sum(greedy_answer(model, example, tokenizer) == example["output"] for example in task_examples)
        scores[task] = (correct / len(task_examples), len(task_examples))

    model.train()
    return scores


def score_line(scores):
    return " ".join(f"{task}:{acc:.3f}/{n}" for task, (acc, n) in sorted(scores.items()))


def save_release_checkpoint(path, model, optimizer, it, train_counts, val_counts, metrics, wandb_run_id):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "it": it,
            "config": release_config(),
            "train_task_counts": dict(train_counts),
            "val_task_counts": dict(val_counts),
            "metrics": metrics,
            "wandb_project": WANDB_PROJECT,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    print(f"saved checkpoint to {path}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer, vocab_size = load_tokenizer()

    all_examples = []
    for filepath in sorted(glob.glob(DATA_PATTERN)):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                instruction = str(obj.get("instruction", "")).strip()
                output = str(obj.get("output", "")).strip()
                task = str(obj.get("task", "unknown")).strip() or "unknown"
                if not instruction or not output:
                    continue
                if task == "short_qa":
                    if re.search(
                        r"(?i)\b(?:maximum|minimum|largest|smallest|highest|lowest|fastest|oldest|"
                        r"first|last|year|date|temperature|weight|height|speed|score|record|"
                        r"standard|regulation|formula|atomic|molar|wavelength|century|invented|"
                        r"assassinated|war|revolution|president|dynasty)\b",
                        instruction,
                    ):
                        continue
                    if any(char.isdigit() for char in output) or len(output.split()) > 3 or len(output) > 40:
                        continue

                _, full_tokens = encode_chat_example(
                    {
                        "instruction": instruction,
                        "output": output,
                    },
                    tokenizer,
                )
                if len(full_tokens) > MODEL_CONTEXT_LENGTH:
                    continue

                all_examples.append(
                    {
                        "instruction": instruction,
                        "output": output,
                        "task": task,
                        "split_key": str(obj.get("split_key", f"{instruction}\t{output}")),
                        "source": str(obj.get("source", "unknown")),
                    }
                )

    if not all_examples:
        raise RuntimeError(f"No examples found in {DATA_PATTERN}. Run the dataset build script first.")

    groups_by_task = defaultdict(lambda: defaultdict(list))
    for example in all_examples:
        groups_by_task[example["task"]][example["split_key"]].append(example)

    train_examples, val_examples = [], []
    for task_idx, task in enumerate(sorted(groups_by_task)):
        groups = list(groups_by_task[task].values())
        random.Random(SEED + task_idx).shuffle(groups)
        val_group_count = max(1, int(len(groups) * VAL_FRAC)) if len(groups) > 1 else 0
        for i, group in enumerate(groups):
            (val_examples if i < val_group_count else train_examples).extend(group)

    random.Random(SEED + 100).shuffle(train_examples)
    random.Random(SEED + 101).shuffle(val_examples)

    train_counts = Counter(example["task"] for example in train_examples)
    val_counts = Counter(example["task"] for example in val_examples)
    train_sampler = TaskSampler(train_examples)
    val_sampler = TaskSampler(val_examples)

    print(f"device: {device}")
    print(f"stage: {RUN_STAGE}")
    print(f"data: {DATA_PATTERN}")
    print(f"init_ckpt: {INIT_CKPT_PATH or '<none>'}")
    print(f"model_context_length: {MODEL_CONTEXT_LENGTH}")
    print(f"hira_targets: {HIRA_TARGETS}")
    print(f"train_examples: {len(train_examples)} val_examples: {len(val_examples)}")
    print(f"train_task_counts: {dict(sorted(train_counts.items()))}")
    print(f"val_task_counts: {dict(sorted(val_counts.items()))}")
    print(f"task_weights: {TASK_WEIGHTS}")
    print("arithmetic_policy: excluded from this assistant SFT release path")

    if os.path.exists(WANDB_API_KEY_FILE):
        with open(WANDB_API_KEY_FILE, encoding="utf-8") as f:
            os.environ["WANDB_API_KEY"] = json.load(f)["wandb-api-key"]
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            **release_config(),
            "train_task_counts": dict(train_counts),
            "val_task_counts": dict(val_counts),
        },
    )

    model = modules.TransformerLM(
        vocab_size=vocab_size,
        context_length=MODEL_CONTEXT_LENGTH,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=2752,
        rope_theta=10000,
    )
    for layer in model.layers:
        layer.attn.q_proj = HiRALinear(layer.attn.q_proj, r=HIRA_R, alpha=HIRA_ALPHA)
        layer.attn.k_proj = HiRALinear(layer.attn.k_proj, r=HIRA_R, alpha=HIRA_ALPHA)
        layer.attn.v_proj = HiRALinear(layer.attn.v_proj, r=HIRA_R, alpha=HIRA_ALPHA)
        layer.attn.o_proj = HiRALinear(layer.attn.o_proj, r=HIRA_R, alpha=HIRA_ALPHA)
        layer.ffn.w1 = HiRALinear(layer.ffn.w1, r=HIRA_R, alpha=HIRA_ALPHA)
        layer.ffn.w2 = HiRALinear(layer.ffn.w2, r=HIRA_R, alpha=HIRA_ALPHA)
        layer.ffn.w3 = HiRALinear(layer.ffn.w3, r=HIRA_R, alpha=HIRA_ALPHA)

    init_obj = torch.load(INIT_CKPT_PATH, map_location="cpu")
    model.load_state_dict(init_obj["model"])
    print(f"continued_from: {INIT_CKPT_PATH}")

    for param in model.parameters():
        param.requires_grad = False

    if TRAIN_FULL_LM_HEAD:
        model.lm_head.W.requires_grad = True

    if TRAIN_ONLY_SPECIAL_EMBEDDINGS:
        model.token_embeddings.embedding_weights.requires_grad = True
        special_ids = [tokenizer.vocab_inv[token.encode("utf-8")] for token in SPECIAL_TOKENS]

        def special_embedding_hook(grad):
            grad = grad.clone()
            keep = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
            keep[special_ids] = True
            grad[~keep] = 0
            return grad

        model.token_embeddings.embedding_weights.register_hook(special_embedding_hook)

    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True

    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = modules.AdamW(trainable_params, lr=MAX_LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    start_iter = 1
    if RESUME_OPTIMIZER and init_obj is not None and "optim" in init_obj:
        optimizer.load_state_dict(init_obj["optim"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_iter = int(init_obj.get("it", 0)) + 1
        print(f"resumed_optimizer_from_iter: {start_iter - 1}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"params: total={total_params:,} trainable={trainable_count:,} ({100 * trainable_count / total_params:.2f}%)")

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    last_metrics = {}

    if MAX_ITERS == 0:
        train_loss = estimate_loss(model, train_sampler, device, tokenizer, autocast)
        val_loss = estimate_loss(model, val_sampler, device, tokenizer, autocast)
        train_tf_exact, train_tf_token = teacher_forced_scores(model, train_examples, device, tokenizer, autocast)
        val_tf_exact, val_tf_token = teacher_forced_scores(model, val_examples, device, tokenizer, autocast)
        train_greedy = greedy_scores(model, train_examples, tokenizer)
        val_greedy = greedy_scores(model, val_examples, tokenizer)
        last_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "tf_exact_train": {task: acc for task, (acc, _) in train_tf_exact.items()},
            "tf_exact_val": {task: acc for task, (acc, _) in val_tf_exact.items()},
            "tf_token_train": {task: acc for task, (acc, _) in train_tf_token.items()},
            "tf_token_val": {task: acc for task, (acc, _) in val_tf_token.items()},
            "greedy_train": {task: acc for task, (acc, _) in train_greedy.items()},
            "greedy_val": {task: acc for task, (acc, _) in val_greedy.items()},
        }
        print(f"[eval] iter {0:6d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        print(f"[eval/train exact] {score_line(train_tf_exact)}")
        print(f"[eval/val exact]   {score_line(val_tf_exact)}")
        print(f"[eval/train token] {score_line(train_tf_token)}")
        print(f"[eval/val token]   {score_line(val_tf_token)}")
        print(f"[greedy/train] {score_line(train_greedy)}")
        print(f"[greedy/val]   {score_line(val_greedy)}")
        print("[mode/val]")
        by_task = defaultdict(list)
        for example in val_examples:
            if len(by_task[example["task"]]) < GREEDY_EXAMPLES_PER_TASK:
                by_task[example["task"]].append(example)
        for task in sorted(by_task):
            ok = 0
            counts = Counter()
            for example in by_task[task]:
                pred = greedy_answer(model, example, tokenizer)
                lowered = pred.strip().lower()
                words = pred.strip().split()
                if task == "assistant_qa":
                    good = (
                        3 <= len(words) <= 60
                        and lowered not in {"yes", "no"}
                        and "shengoovlei" not in lowered
                        and "as an ai" not in lowered
                        and "sorry" not in lowered
                    )
                    counts["sentence" if good else lowered[:30]] += 1
                elif task == "yesno":
                    good = lowered in {"yes", "no"}
                    counts[lowered[:30]] += 1
                else:
                    good = pred == example["output"]
                    counts["exact" if good else pred[:30]] += 1
                ok += int(good)
            print(f"[mode/val] task={task} mode_ok={ok / max(1, len(by_task[task])):.3f}/{len(by_task[task])} common={counts.most_common(8)}")
        model.eval()
        print("[first_token/val]")
        for task in sorted(set(example["task"] for example in val_examples)):
            task_examples = [example for example in val_examples if example["task"] == task][:EVAL_EXAMPLES_PER_TASK]
            ranks = []
            top1_ok = 0
            yesno_top1 = 0
            top1_counts = Counter()
            examples = []
            for example in task_examples:
                prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)
                target = full_tokens[len(prompt_tokens)]
                x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
                with torch.no_grad():
                    with autocast:
                        logits = model(x)[0, -1].float()
                top = torch.topk(logits, k=5)
                top_ids = top.indices.tolist()
                top_texts = [tokenizer.decode([token_id]).replace("\n", "\\n") for token_id in top_ids]
                target_text = tokenizer.decode([target]).replace("\n", "\\n")
                rank = int((logits > logits[target]).sum().item()) + 1
                ranks.append(rank)
                top1_ok += int(top_ids[0] == target)
                yesno_top1 += int(top_texts[0].strip().lower() in {"yes", "no"})
                top1_counts[top_texts[0]] += 1
                if len(examples) < 3:
                    examples.append((example["instruction"], target_text, rank, top_texts))
            print(
                f"[first_token/val] task={task} n={len(task_examples)} "
                f"target_top1={top1_ok / max(1, len(task_examples)):.3f} "
                f"avg_rank={sum(ranks) / max(1, len(ranks)):.1f} "
                f"yesno_top1={yesno_top1 / max(1, len(task_examples)):.3f} "
                f"top1_common={top1_counts.most_common(8)}"
            )
            for instruction, target_text, rank, top_texts in examples:
                print(
                    f"[first_token/sample] task={task} prompt={instruction!r} "
                    f"target={target_text!r} rank={rank} top5={top_texts!r}"
                )
        print("[sample/val]")
        by_task = defaultdict(list)
        for example in val_examples:
            if len(by_task[example["task"]]) < 3:
                by_task[example["task"]].append(example)
        for task in sorted(by_task):
            for example in by_task[task]:
                pred = greedy_answer(model, example, tokenizer)
                print(
                    f"[sample] task={example['task']} prompt={example['instruction']!r} "
                    f"gold={example['output']!r} pred={pred!r}"
                )
        wandb.log(
            {
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
                **{f"tf_exact_train/{task}": acc for task, (acc, _) in train_tf_exact.items()},
                **{f"tf_exact_val/{task}": acc for task, (acc, _) in val_tf_exact.items()},
                **{f"tf_token_train/{task}": acc for task, (acc, _) in train_tf_token.items()},
                **{f"tf_token_val/{task}": acc for task, (acc, _) in val_tf_token.items()},
                **{f"greedy_train/{task}": acc for task, (acc, _) in train_greedy.items()},
                **{f"greedy_val/{task}": acc for task, (acc, _) in val_greedy.items()},
            },
            step=0,
        )
        save_release_checkpoint(
            FINAL_CKPT_PATH,
            model,
            optimizer,
            0,
            train_counts,
            val_counts,
            last_metrics,
            run.id,
        )
        wandb.finish()
        return

    tokens_since_log = 0
    supervised_since_log = 0
    t0 = time.time()

    for it in range(start_iter, MAX_ITERS + 1):
        lr = modules.run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=MAX_LR,
            min_learning_rate=MIN_LR,
            warmup_iters=WARMUP_ITERS,
            cosine_cycle_iters=MAX_ITERS,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y, supervised_tokens, tasks = make_batch(train_sampler, device, tokenizer)
        tokens_since_log += x.numel()
        supervised_since_log += supervised_tokens

        with autocast:
            loss = sft_loss(model(x), y, tasks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        modules.run_gradient_clipping(model.parameters(), max_l2_norm=MAX_GRAD_NORM)
        optimizer.step()

        if it % LOG_INTERVAL == 0:
            dt = time.time() - t0
            a_norm, b_norm = hira_ab_norm(model)
            update_ratio = hira_update_ratio(model)
            tok_s = tokens_since_log / dt
            sup_tok_s = supervised_since_log / dt
            print(
                f"iter {it:6d} loss {loss.item():.4f} lr {lr:.2e} | "
                f"tok/s {tok_s:.0f} supervised_tok/s {sup_tok_s:.0f} | "
                f"seq_len {x.shape[1]} update_ratio {update_ratio:.4f}"
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tok_s": tok_s,
                    "train/supervised_tok_s": sup_tok_s,
                    "train/batch_seq_len": x.shape[1],
                    "adapter/A_norm": a_norm,
                    "adapter/B_norm": b_norm,
                    "adapter/update_ratio": update_ratio,
                },
                step=it,
            )
            tokens_since_log = 0
            supervised_since_log = 0
            t0 = time.time()

        if it % EVAL_INTERVAL == 0:
            train_loss = estimate_loss(model, train_sampler, device, tokenizer, autocast)
            val_loss = estimate_loss(model, val_sampler, device, tokenizer, autocast)
            train_tf_exact, train_tf_token = teacher_forced_scores(model, train_examples, device, tokenizer, autocast)
            val_tf_exact, val_tf_token = teacher_forced_scores(model, val_examples, device, tokenizer, autocast)
            last_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tf_exact_train": {task: acc for task, (acc, _) in train_tf_exact.items()},
                "tf_exact_val": {task: acc for task, (acc, _) in val_tf_exact.items()},
                "tf_token_train": {task: acc for task, (acc, _) in train_tf_token.items()},
                "tf_token_val": {task: acc for task, (acc, _) in val_tf_token.items()},
            }

            print(f"[eval] iter {it:6d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
            print(f"[eval/train exact] {score_line(train_tf_exact)}")
            print(f"[eval/val exact]   {score_line(val_tf_exact)}")
            print(f"[eval/train token] {score_line(train_tf_token)}")
            print(f"[eval/val token]   {score_line(val_tf_token)}")

            log_payload = {
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
            }
            log_payload.update({f"tf_exact_train/{task}": acc for task, (acc, _) in train_tf_exact.items()})
            log_payload.update({f"tf_exact_val/{task}": acc for task, (acc, _) in val_tf_exact.items()})
            log_payload.update({f"tf_token_train/{task}": acc for task, (acc, _) in train_tf_token.items()})
            log_payload.update({f"tf_token_val/{task}": acc for task, (acc, _) in val_tf_token.items()})
            wandb.log(log_payload, step=it)

            if it % GREEDY_EVAL_INTERVAL == 0:
                train_greedy = greedy_scores(model, train_examples, tokenizer)
                val_greedy = greedy_scores(model, val_examples, tokenizer)
                last_metrics["greedy_train"] = {task: acc for task, (acc, _) in train_greedy.items()}
                last_metrics["greedy_val"] = {task: acc for task, (acc, _) in val_greedy.items()}

                print(f"[greedy/train] {score_line(train_greedy)}")
                print(f"[greedy/val]   {score_line(val_greedy)}")

                greedy_payload = {}
                greedy_payload.update({f"greedy_train/{task}": acc for task, (acc, _) in train_greedy.items()})
                greedy_payload.update({f"greedy_val/{task}": acc for task, (acc, _) in val_greedy.items()})
                wandb.log(greedy_payload, step=it)

            if it % SAMPLE_INTERVAL == 0:
                print("[sample/val]")
                by_task = defaultdict(list)
                for example in val_examples:
                    if len(by_task[example["task"]]) < 2:
                        by_task[example["task"]].append(example)
                for task in sorted(by_task):
                    for example in by_task[task]:
                        pred = greedy_answer(model, example, tokenizer)
                        print(
                            f"[sample] task={example['task']} prompt={example['instruction']!r} "
                            f"gold={example['output']!r} pred={pred!r}"
                        )

        if it % CHECKPOINT_INTERVAL == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/re_sft_{RUN_STAGE}_iter_{it}.pt"
            save_release_checkpoint(
                ckpt_path,
                model,
                optimizer,
                it,
                train_counts,
                val_counts,
                last_metrics,
                run.id,
            )

    final_val_greedy = greedy_scores(model, val_examples, tokenizer)
    last_metrics["greedy_final_val"] = {task: acc for task, (acc, _) in final_val_greedy.items()}
    print(f"[greedy/final val] {score_line(final_val_greedy)}")
    wandb.log({f"greedy_final_val/{task}": acc for task, (acc, _) in final_val_greedy.items()}, step=MAX_ITERS)

    save_release_checkpoint(
        FINAL_CKPT_PATH,
        model,
        optimizer,
        MAX_ITERS,
        train_counts,
        val_counts,
        last_metrics,
        run.id,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
