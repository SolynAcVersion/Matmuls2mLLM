import glob
import json
import os
import time
import random
import datetime
from collections import Counter

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



def make_batch(examples, indices, device, tokenizer, batch_size, ignore_index=-666):
    batch = [examples[indices[i % len(indices)]] for i in range(batch_size)]
    seq_len = 0
    items = []
    pad_id = tokenizer.vocab_inv["<|pad|>".encode("utf-8")]
    for ex in batch:
        prompt_tokens, full_tokens = encode_chat_example(ex, tokenizer)
        items.append((prompt_tokens, full_tokens))
        seq_len = max(seq_len, len(full_tokens))

    x = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long, device=device)
    supervised_tokens = 0
    for b, (prompt_tokens, full_tokens) in enumerate(items):
        labels = [ignore_index] * len(full_tokens)
        labels[len(prompt_tokens):] = full_tokens[len(prompt_tokens):]
        supervised_tokens += len(full_tokens) - len(prompt_tokens)
        x[b, :len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long, device=device)
        y[b, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y, supervised_tokens


def ce_loss(logits, labels, ignore_index=-666):
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().reshape(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().reshape(-1),
        ignore_index=ignore_index,
    )


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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_stage = "assistant_sft_stage4_task_aligned_v2"
    data_pattern = "./data/re_sft_stage4_task_aligned_deepseek_all_v2.jsonl"
    anchor_pattern = "./data/re_sft_stage3j_mix.jsonl"
    init_ckpt_path = "./data/shengoovlei_assistant_sft_stage3j_filtered_mix_final.pt"
    final_ckpt_path = "./data/shengoovlei_assistant_sft_stage4_task_aligned_v2_final.pt"
    checkpoint_dir = "./checkpoints"
    checkpoint_format = "shengoovlei_assistant_sft_stage4_task_aligned_v2"

    context_length = 1024
    batch_size = 32
    max_iters = 2500
    max_learning_rate = 8e-7
    min_learning_rate = 4e-7
    warmup_iters = 100
    max_grad_norm = 1.0

    hira_r = 32
    hira_alpha = 32
    train_full_lm_head = True

    log_interval = 100
    eval_interval = 500
    eval_iters = 5
    greedy_eval_interval = 1000
    greedy_examples_per_task = 32
    sample_interval = 1000
    checkpoint_interval = 1000
    max_generate_tokens = 128

    val_frac = 0.05
    special_tokens = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]

    eval_data_pattern = "./data/re_sft_assistant_stage3_mode_boundary.jsonl"

    tokenizer, vocab_size = load_tokenizer()

    # 主训练数据：task-aligned deepseek assistant 数据
    all_examples = []
    for filepath in sorted(glob.glob(data_pattern)):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = str(obj.get("instruction", "")).strip()
                output = str(obj.get("output", "")).strip()
                if not instruction or not output:
                    continue
                _, full_tokens = encode_chat_example(
                    {"instruction": instruction, "output": output}, tokenizer
                )
                if len(full_tokens) > 256:
                    continue
                all_examples.append({"instruction": instruction, "output": output, "task": "assistant_qa"})

    # 稳定锚点：只保 repeat / yesno / identity，防止遗忘
    anchor_keep = {"repeat": 1200, "yesno": 900, "identity": 200}
    anchor_count = Counter()
    for filepath in sorted(glob.glob(anchor_pattern)):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task = str(obj.get("task", "")).strip()
                if task not in anchor_keep:
                    continue
                if anchor_count[task] >= anchor_keep[task]:
                    continue
                instruction = str(obj.get("instruction", "")).strip()
                output = str(obj.get("output", "")).strip()
                if not instruction or not output:
                    continue
                _, full_tokens = encode_chat_example(
                    {"instruction": instruction, "output": output}, tokenizer
                )
                if len(full_tokens) > 256:
                    continue
                all_examples.append({"instruction": instruction, "output": output, "task": task})
                anchor_count[task] += 1

    if not all_examples:
        raise RuntimeError(f"No examples found in {data_pattern}")

    random.Random(seed).shuffle(all_examples)
    val_n = max(1, int(len(all_examples) * val_frac))
    val_examples_raw = all_examples[:val_n]
    train_examples = all_examples[val_n:]
    train_indices = list(range(len(train_examples)))
    random.Random(seed + 1).shuffle(train_indices)
    val_indices = list(range(len(val_examples_raw)))

    print(f"device: {device}")
    print(f"stage: {run_stage}")
    task_counts = Counter(ex["task"] for ex in all_examples)
    print(f"data: {data_pattern}  train={len(train_examples)} val={len(val_examples_raw)}")
    print(f"train_task_mix: {dict(sorted(task_counts.items()))}")
    print(f"init_ckpt: {init_ckpt_path}")

    # 读固定 eval 集（有 task 字段，用于 greedy/mode 评估）
    eval_examples = []
    for filepath in sorted(glob.glob(eval_data_pattern)):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = str(obj.get("instruction", "")).strip()
                output = str(obj.get("output", "")).strip()
                task = str(obj.get("task", "unknown")).strip() or "unknown"
                if not instruction or not output:
                    continue
                _, full_tokens = encode_chat_example(
                    {"instruction": instruction, "output": output}, tokenizer
                )
                if len(full_tokens) > context_length:
                    continue
                eval_examples.append({"instruction": instruction, "output": output, "task": task})

    # 每个 task 最多取 greedy_examples_per_task 条做 greedy/mode eval
    from collections import defaultdict
    eval_by_task = defaultdict(list)
    for ex in eval_examples:
        if len(eval_by_task[ex["task"]]) < greedy_examples_per_task:
            eval_by_task[ex["task"]].append(ex)
    print(f"eval tasks: { {t: len(v) for t, v in sorted(eval_by_task.items())} }")

    nowtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = wandb.init(
        project="shengoovlei_sft_release",
        name=f"{run_stage}_{nowtime}",
        mode="disabled",
        config={
            "run_stage": run_stage,
            "init_ckpt_path": init_ckpt_path,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "train_size": len(train_examples),
            "val_size": len(val_examples_raw),
        },
    )

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
    print(f"params: total={total_params:,} trainable={trainable_count:,} ({100*trainable_count/total_params:.2f}%)")

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y, _ = make_batch(val_examples_raw, val_indices, device, tokenizer, batch_size)
            with autocast:
                losses[k] = ce_loss(model(x), y)
        model.train()
        return losses.mean().item()

    @torch.no_grad()
    def run_greedy_and_mode():
        model.eval()
        open_tasks = {"assistant_qa"}

        def assistant_ok(text):
            lo = text.lower().strip()
            words = text.strip().split()
            if not words or len(words) > 80:
                return False
            first = words[0].lower().strip(".,!?;:")
            if first in {"yes", "no", "shengoovlei"}:
                return False
            bad = ["shengoovlei", "i don't know", "i do not know", "i cannot", "i can't", "not sure", "i'm not sure", "sorry", "as an ai", "no idea"]
            return not any(x in lo for x in bad)

        scores = {}
        for task, task_examples in sorted(eval_by_task.items()):
            if task in open_tasks:
                correct = 0
                for ex in task_examples:
                    text = greedy_answer(model, ex, tokenizer, context_length, max_generate_tokens)
                    correct += int(assistant_ok(text))
            else:
                correct = sum(
                    greedy_answer(model, ex, tokenizer, context_length, max_generate_tokens) == ex["output"]
                    for ex in task_examples
                )
            scores[task] = (correct / len(task_examples), len(task_examples))
        print(f"[greedy/eval] " + " ".join(f"{t}:{a:.3f}/{n}" for t, (a, n) in sorted(scores.items())))
        wandb.log({f"greedy_eval/{t}": a for t, (a, n) in scores.items()}, step=it)

        print("[mode/eval]")
        for task, task_examples in sorted(eval_by_task.items()):
            ok = 0
            counts = Counter()
            for ex in task_examples:
                pred = greedy_answer(model, ex, tokenizer, context_length, max_generate_tokens)
                lo = pred.strip().lower()
                if task == "assistant_qa":
                    good = assistant_ok(pred)
                    counts["answer" if good else lo[:30]] += 1
                elif task == "yesno":
                    good = lo in {"yes", "no"}
                    counts[lo[:30]] += 1
                else:
                    good = pred == ex["output"]
                    counts["exact" if good else pred[:30]] += 1
                ok += int(good)
            print(f"[mode/eval] task={task} mode_ok={ok/max(1,len(task_examples)):.3f}/{len(task_examples)} common={counts.most_common(5)}")

        print("[sample/eval]")
        for task, task_examples in sorted(eval_by_task.items()):
            for ex in task_examples[:2]:
                pred = greedy_answer(model, ex, tokenizer, context_length, max_generate_tokens)
                print(f"[sample] task={task} prompt={ex['instruction']!r} gold={ex['output']!r} pred={pred!r}")
        model.train()

    def save_checkpoint(path, it):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "checkpoint_format": checkpoint_format,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "it": it,
            "wandb_run_id": run.id,
        }, path)
        print(f"saved checkpoint to {path}")

    tokens_since_log = 0
    supervised_since_log = 0
    cursor = 0
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

        # 顺序采样，epoch 结束后 reshuffle
        batch_idx = train_indices[cursor:cursor + batch_size]
        cursor += batch_size
        if cursor >= len(train_indices):
            random.shuffle(train_indices)
            cursor = 0

        x, y, supervised_tokens = make_batch(train_examples, batch_idx, device, tokenizer, batch_size)
        tokens_since_log += x.numel()
        supervised_since_log += supervised_tokens

        with autocast:
            loss = ce_loss(model(x), y)

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
            val_loss = estimate_loss()
            print(f"[eval] iter {it:6d} | val loss {val_loss:.4f}")
            wandb.log({"eval/val_loss": val_loss}, step=it)

        if it % greedy_eval_interval == 0:
            run_greedy_and_mode()

        if it % checkpoint_interval == 0:
            save_checkpoint(f"{checkpoint_dir}/re_sft_{run_stage}_iter_{it}.pt", it)

    run_greedy_and_mode()
    save_checkpoint(final_ckpt_path, max_iters)
    wandb.finish()


if __name__ == "__main__":
    main()
