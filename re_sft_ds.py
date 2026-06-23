import os
import glob
import json
import re
import time
import random
import datetime
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
    get_hira_ab_norm,
    get_hira_update_ratio,
    load_tokenizer,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ignore_index = -666


def make_batch(source, task_weights, device, tokenizer, batch_size):
    # source 既可以是 (pools, tasks, weights) 采样器，也可以是一段固定的 example 列表
    items = []
    for b in range(batch_size):
        if isinstance(source, tuple):
            pools, tasks, weights = source
            task = random.choices(tasks, weights=weights, k=1)[0]
            example = random.choice(pools[task])
        else:
            example = source[b]
        prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)
        items.append((example["task"], prompt_tokens, full_tokens))

    seq_len = max(len(full_tokens) for _, _, full_tokens in items)
    pad_id = tokenizer.vocab_inv["<|pad|>".encode("utf-8")]

    x = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long, device=device)
    tasks_out = []
    supervised_tokens = 0

    for b, (task, prompt_tokens, full_tokens) in enumerate(items):
        labels = [ignore_index] * len(full_tokens)
        labels[len(prompt_tokens):] = full_tokens[len(prompt_tokens):]
        supervised_tokens += len(full_tokens) - len(prompt_tokens)
        tasks_out.append(task)
        x[b, :len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long, device=device)
        y[b, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)

    return x, y, supervised_tokens, tasks_out


def sft_loss(logits, labels, tasks, task_weights):
    # 每条样本先在自己的监督 token 上平均，再按 task 平均，最后按权重加权
    # 这样长回答(assistant_qa)不会因为 token 多而天然抢梯度
    B, S, V = logits.shape
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    mask = labels != ignore_index

    losses = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, S - 1)

    counts = mask.sum(dim=1).clamp_min(1)
    losses = (losses * mask).sum(dim=1) / counts

    task_losses = []
    weights = []
    for task in sorted(set(tasks)):
        idx = [i for i, name in enumerate(tasks) if name == task]
        task_losses.append(losses[torch.tensor(idx, dtype=torch.long, device=losses.device)].mean())
        weights.append(float(task_weights.get(task, 1.0)))

    weight_tensor = torch.tensor(weights, dtype=losses.dtype, device=losses.device)
    if weight_tensor.sum() <= 0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    return torch.stack(task_losses).mul(weight_tensor).sum()


@torch.no_grad()
def greedy_answer(model, example, tokenizer, context_length, max_generate_tokens):
    end_id = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]
    prompt_tokens = tokenizer.encode(format_chat_prompt(example["instruction"]))
    output_ids = modules.generating(
        model=model,
        enc_user_prompt=prompt_tokens,
        end_token=end_id,
        context_len=context_length,
        max_token=max_generate_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    )
    text = tokenizer.decode(output_ids.tolist() if hasattr(output_ids, "tolist") else output_ids)
    return text.split("<|endoftext|>", 1)[0].strip()


def score_line(scores):
    return " ".join(f"{task}:{acc:.3f}/{n}" for task, (acc, n) in sorted(scores.items()))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # =========================
    # stage3g: assistant_qa-only 救援
    # 目的: 把 assistant_qa 的首 token 从 yes/no 里拉回来。
    # 依据 stage3f first_token 审计: assistant_qa prompt 下 yesno_top1=0.805,
    # identity/repeat/yesno 三个模式路由已达标。报告处方是“不能再混训 yesno”。
    # 所以这一阶段 yesno 权重严格为 0, 只用 repeat/identity 当锚点防遗忘。
    # =========================
    run_stage = "assistant_sft_stage3g_v4_assistant_only_recover"
    data_pattern = "./data/re_sft_assistant_stage3_mode_boundary.jsonl"
    init_ckpt_path = "./data/shengoovlei_assistant_sft_stage3e_v4_mode_boundary_final.pt"
    final_ckpt_path = "./data/shengoovlei_assistant_sft_stage3g_v4_assistant_only_recover_final.pt"
    checkpoint_dir = "./checkpoints"
    checkpoint_format = "shengoovlei_assistant_sft_v1"

    context_length = 1024
    batch_size = 64
    max_iters = 600
    max_learning_rate = 5e-6
    min_learning_rate = 2e-6
    warmup_iters = 50
    max_grad_norm = 1.0

    hira_r = 32
    hira_alpha = 32
    train_full_lm_head = True

    log_interval = 50
    eval_interval = 100
    eval_iters = 3
    eval_examples_per_task = 128
    greedy_eval_interval = 300
    greedy_examples_per_task = 32
    sample_interval = 300
    checkpoint_interval = 300
    max_generate_tokens = 128

    val_frac = 0.10
    task_weights = {
        "assistant_qa": 0.80,
        "repeat": 0.15,
        "identity": 0.05,
        "yesno": 0.00,
        "short_qa": 0.00,
    }

    special_tokens = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]
    exact_match_tasks = {"repeat", "identity", "yesno", "short_qa", "general_short", "count"}
    open_generation_tasks = {"assistant_qa"}

    tokenizer, vocab_size = load_tokenizer()

    # 读数据 + 过滤 short_qa(本阶段 short_qa 权重为 0, 过滤只为保持口径一致)
    all_examples = []
    for filepath in sorted(glob.glob(data_pattern)):
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
                    if any(ch.isdigit() for ch in output) or len(output.split()) > 3 or len(output) > 40:
                        continue
                _, full_tokens = encode_chat_example({"instruction": instruction, "output": output}, tokenizer)
                if len(full_tokens) > context_length:
                    continue
                all_examples.append({
                    "instruction": instruction,
                    "output": output,
                    "task": task,
                    "split_key": str(obj.get("split_key", f"{instruction}\t{output}")),
                    "source": str(obj.get("source", "unknown")),
                })

    if len(all_examples) == 0:
        raise RuntimeError(f"No examples found in {data_pattern}. Run the dataset build script first.")

    # 按 task + split_key 分组划分 train/val, 保证同一 payload 不跨 split
    groups_by_task = defaultdict(lambda: defaultdict(list))
    for example in all_examples:
        groups_by_task[example["task"]][example["split_key"]].append(example)

    train_examples, val_examples = [], []
    for task_idx, task in enumerate(sorted(groups_by_task)):
        groups = list(groups_by_task[task].values())
        random.Random(seed + task_idx).shuffle(groups)
        val_group_count = max(1, int(len(groups) * val_frac)) if len(groups) > 1 else 0
        for i, group in enumerate(groups):
            (val_examples if i < val_group_count else train_examples).extend(group)

    random.Random(seed + 100).shuffle(train_examples)
    random.Random(seed + 101).shuffle(val_examples)

    train_counts = Counter(example["task"] for example in train_examples)
    val_counts = Counter(example["task"] for example in val_examples)

    # 训练/验证采样器: (pools, tasks, weights), 只采权重>0 的 task
    train_pools = defaultdict(list)
    for example in train_examples:
        train_pools[example["task"]].append(example)
    val_pools = defaultdict(list)
    for example in val_examples:
        val_pools[example["task"]].append(example)

    active_train = [t for t in task_weights if task_weights[t] > 0 and train_pools[t]]
    train_sampler = (train_pools, active_train, [task_weights[t] for t in active_train])
    active_val = [t for t in task_weights if task_weights[t] > 0 and val_pools[t]]
    val_sampler = (val_pools, active_val, [task_weights[t] for t in active_val])

    print(f"device: {device}")
    print(f"stage: {run_stage}")
    print(f"data: {data_pattern}")
    print(f"init_ckpt: {init_ckpt_path}")
    print(f"train_examples: {len(train_examples)} val_examples: {len(val_examples)}")
    print(f"train_task_counts: {dict(sorted(train_counts.items()))}")
    print(f"val_task_counts: {dict(sorted(val_counts.items()))}")
    print(f"task_weights: {task_weights}")

    with open("./pswd.json", encoding="utf-8") as f:
        os.environ["WANDB_API_KEY"] = json.load(f)["wandb-api-key"]
    nowtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = wandb.init(
        project="shengoovlei_sft_release",
        name=f"{run_stage}_{nowtime}",
        config={
            "run_stage": run_stage,
            "init_ckpt_path": init_ckpt_path,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "task_weights": task_weights,
            "train_task_counts": dict(train_counts),
            "val_task_counts": dict(val_counts),
        },
    )

    # 建模型 + 给注意力和 FFN 套 HiRA adapter
    model = modules.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=2752,
        rope_theta=10000,
    )
    for layer in model.layers:
        layer.attn.q_proj = HiRALinear(layer.attn.q_proj, r=hira_r, alpha=hira_alpha)
        layer.attn.k_proj = HiRALinear(layer.attn.k_proj, r=hira_r, alpha=hira_alpha)
        layer.attn.v_proj = HiRALinear(layer.attn.v_proj, r=hira_r, alpha=hira_alpha)
        layer.attn.o_proj = HiRALinear(layer.attn.o_proj, r=hira_r, alpha=hira_alpha)
        layer.ffn.w1 = HiRALinear(layer.ffn.w1, r=hira_r, alpha=hira_alpha)
        layer.ffn.w2 = HiRALinear(layer.ffn.w2, r=hira_r, alpha=hira_alpha)
        layer.ffn.w3 = HiRALinear(layer.ffn.w3, r=hira_r, alpha=hira_alpha)

    init_obj = torch.load(init_ckpt_path, map_location="cpu")
    model.load_state_dict(init_obj["model"])
    print(f"continued_from: {init_ckpt_path}")

    # 冻结 base, 只放开 lm_head / 特殊 token embedding / HiRA A,B
    for param in model.parameters():
        param.requires_grad = False
    if train_full_lm_head:
        model.lm_head.W.requires_grad = True

    model.token_embeddings.embedding_weights.requires_grad = True
    special_ids = [tokenizer.vocab_inv[token.encode("utf-8")] for token in special_tokens]

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
    optimizer = modules.AdamW(trainable_params, lr=max_learning_rate, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"params: total={total_params:,} trainable={trainable_count:,} ({100 * trainable_count / total_params:.2f}%)")

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def estimate_loss(sampler):
        model.eval()
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y, _, tasks = make_batch(sampler, task_weights, device, tokenizer, batch_size)
            with autocast:
                losses[k] = sft_loss(model(x), y, tasks, task_weights)
        model.train()
        return losses.mean().item()

    # teacher-forced 的 exact / token accuracy
    @torch.no_grad()
    def teacher_forced_scores(examples):
        model.eval()
        by_task = defaultdict(list)
        for example in examples:
            if len(by_task[example["task"]]) < eval_examples_per_task:
                by_task[example["task"]].append(example)

        exact_scores, token_scores = {}, {}
        for task, task_examples in sorted(by_task.items()):
            exact_correct = token_correct = token_total = 0
            for start in range(0, len(task_examples), batch_size):
                batch_examples = task_examples[start:start + batch_size]
                x, y, _, _ = make_batch(batch_examples, task_weights, device, tokenizer, len(batch_examples))
                with autocast:
                    logits = model(x)
                pred = logits[:, :-1, :].argmax(dim=-1)
                target = y[:, 1:]
                mask = target != ignore_index
                for b in range(len(batch_examples)):
                    positions = torch.nonzero(mask[b], as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    row_correct = pred[b, positions] == target[b, positions]
                    token_correct += int(row_correct.sum().item())
                    token_total += int(positions.numel())
                    exact_correct += int(torch.equal(pred[b, positions], target[b, positions]))
            if task in exact_match_tasks:
                exact_scores[task] = (exact_correct / len(task_examples), len(task_examples))
            if task in open_generation_tasks or task in exact_match_tasks:
                token_scores[task] = (token_correct / token_total if token_total else 0.0, token_total)
        model.train()
        return exact_scores, token_scores

    # greedy 解码后的指标: 开放问答看是否一句相关短句, 其它看 exact
    @torch.no_grad()
    def greedy_scores(examples):
        model.eval()
        by_task = defaultdict(list)
        for example in examples:
            if len(by_task[example["task"]]) < greedy_examples_per_task:
                by_task[example["task"]].append(example)

        scores = {}
        for task, task_examples in sorted(by_task.items()):
            if task in open_generation_tasks:
                correct = 0
                for example in task_examples:
                    text = greedy_answer(model, example, tokenizer, context_length, max_generate_tokens)
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
                correct = sum(
                    greedy_answer(model, example, tokenizer, context_length, max_generate_tokens) == example["output"]
                    for example in task_examples
                )
            scores[task] = (correct / len(task_examples), len(task_examples))
        model.train()
        return scores

    def save_checkpoint(path, it, metrics):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save({
            "checkpoint_format": checkpoint_format,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "it": it,
            "train_task_counts": dict(train_counts),
            "val_task_counts": dict(val_counts),
            "metrics": metrics,
            "wandb_project": "shengoovlei_sft_release",
            "wandb_run_id": run.id,
        }, path)
        print(f"saved checkpoint to {path}")

    last_metrics = {}
    tokens_since_log = 0
    supervised_since_log = 0
    t0 = time.time()

    for it in range(1, max_iters + 1):
        lr = modules.run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y, supervised_tokens, tasks = make_batch(train_sampler, task_weights, device, tokenizer, batch_size)
        tokens_since_log += x.numel()
        supervised_since_log += supervised_tokens

        with autocast:
            loss = sft_loss(model(x), y, tasks, task_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        modules.run_gradient_clipping(model.parameters(), max_l2_norm=max_grad_norm)
        optimizer.step()

        if it % log_interval == 0:
            dt = time.time() - t0
            a_norm, b_norm = get_hira_ab_norm(model)
            update_ratio = get_hira_update_ratio(model)
            tok_s = tokens_since_log / dt
            sup_tok_s = supervised_since_log / dt
            print(
                f"iter {it:6d} loss {loss.item():.4f} lr {lr:.2e} | "
                f"tok/s {tok_s:.0f} supervised_tok/s {sup_tok_s:.0f} | "
                f"seq_len {x.shape[1]} update_ratio {update_ratio:.4f}"
            )
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/tok_s": tok_s,
                "train/supervised_tok_s": sup_tok_s,
                "adapter/A_norm": a_norm,
                "adapter/B_norm": b_norm,
                "adapter/update_ratio": update_ratio,
            }, step=it)
            tokens_since_log = 0
            supervised_since_log = 0
            t0 = time.time()

        if it % eval_interval == 0:
            train_loss = estimate_loss(train_sampler)
            val_loss = estimate_loss(val_sampler)
            train_tf_exact, train_tf_token = teacher_forced_scores(train_examples)
            val_tf_exact, val_tf_token = teacher_forced_scores(val_examples)
            last_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tf_exact_val": {task: acc for task, (acc, _) in val_tf_exact.items()},
                "tf_token_val": {task: acc for task, (acc, _) in val_tf_token.items()},
            }
            print(f"[eval] iter {it:6d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
            print(f"[eval/train exact] {score_line(train_tf_exact)}")
            print(f"[eval/val exact]   {score_line(val_tf_exact)}")
            print(f"[eval/train token] {score_line(train_tf_token)}")
            print(f"[eval/val token]   {score_line(val_tf_token)}")
            wandb.log({
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
                **{f"tf_exact_val/{task}": acc for task, (acc, _) in val_tf_exact.items()},
                **{f"tf_token_val/{task}": acc for task, (acc, _) in val_tf_token.items()},
            }, step=it)

            if it % greedy_eval_interval == 0:
                train_greedy = greedy_scores(train_examples)
                val_greedy = greedy_scores(val_examples)
                last_metrics["greedy_val"] = {task: acc for task, (acc, _) in val_greedy.items()}
                print(f"[greedy/train] {score_line(train_greedy)}")
                print(f"[greedy/val]   {score_line(val_greedy)}")
                wandb.log({
                    **{f"greedy_train/{task}": acc for task, (acc, _) in train_greedy.items()},
                    **{f"greedy_val/{task}": acc for task, (acc, _) in val_greedy.items()},
                }, step=it)

                # mode/val: 只看回答模式对不对, 不看知识正确性
                print("[mode/val]")
                mode_by_task = defaultdict(list)
                for example in val_examples:
                    if len(mode_by_task[example["task"]]) < greedy_examples_per_task:
                        mode_by_task[example["task"]].append(example)
                for task in sorted(mode_by_task):
                    ok = 0
                    counts = Counter()
                    for example in mode_by_task[task]:
                        pred = greedy_answer(model, example, tokenizer, context_length, max_generate_tokens)
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
                    print(f"[mode/val] task={task} mode_ok={ok / max(1, len(mode_by_task[task])):.3f}/{len(mode_by_task[task])} common={counts.most_common(8)}")

            if it % sample_interval == 0:
                print("[sample/val]")
                sample_by_task = defaultdict(list)
                for example in val_examples:
                    if len(sample_by_task[example["task"]]) < 2:
                        sample_by_task[example["task"]].append(example)
                for task in sorted(sample_by_task):
                    for example in sample_by_task[task]:
                        pred = greedy_answer(model, example, tokenizer, context_length, max_generate_tokens)
                        print(f"[sample] task={example['task']} prompt={example['instruction']!r} gold={example['output']!r} pred={pred!r}")

        if it % checkpoint_interval == 0:
            save_checkpoint(f"{checkpoint_dir}/re_sft_{run_stage}_iter_{it}.pt", it, last_metrics)

    final_val_greedy = greedy_scores(val_examples)
    last_metrics["greedy_final_val"] = {task: acc for task, (acc, _) in final_val_greedy.items()}
    print(f"[greedy/final val] {score_line(final_val_greedy)}")
    wandb.log({f"greedy_final_val/{task}": acc for task, (acc, _) in final_val_greedy.items()}, step=max_iters)

    save_checkpoint(final_ckpt_path, max_iters, last_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
