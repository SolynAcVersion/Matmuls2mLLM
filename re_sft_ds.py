import contextlib
import datetime
import glob
import json
import os
import random
import time

import numpy as np
import torch
import wandb

import modules
from sft_chat_templetes import (
    HiRALinear,
    encode_chat_example,
    get_hira_ab_norm,
    get_hira_update_ratio,
    load_tokenizer,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_simple_jsonl(pattern: str):

    reject_phrases_lower = [
        "not sure",
        "don't have access",
        "can't access",
        "i cannot",
        "i don't know",
        "i don't have",
        "i am not able",
        "i'm not able",
    ]
    
    examples = []
    for filepath in glob.glob(pattern):
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
                if output.strip().lower() in ["<nooutput>", "<no output>"]:
                    continue
                output_lower = output.lower()
                if any(phrase in output_lower for phrase in reject_phrases_lower):
                    continue
                examples.append({"instruction": instruction, "output": output})
    return examples


def split_examples(examples, seed):
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    valid_end = max(1, len(examples) // 10)
    return examples[valid_end:], examples[:valid_end]


def make_batch(examples, batch_size, context_length, device, tokenizer, ignore_index=-666):
    if not examples:
        raise ValueError("empty dataset")

    pad_token_id = tokenizer.vocab_inv["<|pad|>".encode("utf-8")]
    x = torch.full((batch_size, context_length), pad_token_id, dtype=torch.long, device=device)
    y = torch.full((batch_size, context_length), ignore_index, dtype=torch.long, device=device)

    for b in range(batch_size):
        for _ in range(10000):
            prompt_tokens, full_tokens = encode_chat_example(random.choice(examples), tokenizer)
            if len(full_tokens) <= context_length:
                break
        else:
            raise RuntimeError(f"failed to sample within context_length={context_length}")

        labels = [ignore_index] * len(full_tokens)
        labels[len(prompt_tokens):] = full_tokens[len(prompt_tokens):]

        x[b, :len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long, device=device)
        y[b, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)

    return x, y


def main():
    # ---------- 配置 ----------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context_length = 1024
    batch_size = 8
    eval_interval = 250
    eval_iters = 30
    log_interval = 50
    checkpoint_interval = 500
    max_learning_rate = 1e-4
    min_learning_rate = 1e-5
    warmup_iters = 200
    max_iters = 25000
    start_iter = 1
    cosine_cycle_iters = max_iters - start_iter + 1
    max_grad_norm = 1.0
    data_pattern = "./data/short_instruct_deepseek.jsonl"
    final_ckpt = "./data/simple_hira_final.pt"

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    tokenizer, vocab_size = load_tokenizer()

    # 加载数据
    all_examples = load_simple_jsonl(data_pattern)
    if not all_examples:
        raise RuntimeError(f"No examples found in {data_pattern}")
    train_examples, val_examples = split_examples(all_examples, 1337)

    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")

    # ---------- 模型初始化 ----------
    model = modules.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=2752,
        rope_theta=10000,
    )
    # 加载预训练权重（包含已扩展的特殊 token）
    pretrain_obj = torch.load(
        "./data/pretrain_gpt2med_iter_390000_chatvocab32003.pt",
        map_location="cpu",
    )
    model.load_state_dict(pretrain_obj["model"])

    # 替换为 HiRA
    for layer in model.layers:
        layer.attn.q_proj = HiRALinear(layer.attn.q_proj, r=32, alpha=32)
        layer.attn.k_proj = HiRALinear(layer.attn.k_proj, r=32, alpha=32)
        layer.attn.v_proj = HiRALinear(layer.attn.v_proj, r=32, alpha=32)
        layer.ffn.w2 = HiRALinear(layer.ffn.w2, r=32, alpha=32)
        layer.ffn.w3 = HiRALinear(layer.ffn.w3, r=32, alpha=32)

    # 冻结所有参数，然后选择性解冻
    for param in model.parameters():
        param.requires_grad = False

    trainable_token_ids = [31999, 32000, 32001, 32002]  

    # 2. 设置 Embedding 和 LM Head 为可训练（矩阵整体可训练，但梯度会被钩子过滤）
    model.token_embeddings.embedding_weights.requires_grad = True
    model.lm_head.W.requires_grad = True

    # 3. 注册钩子
    def embedding_grad_hook(grad):
        grad = grad.clone()
        keep = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
        keep[trainable_token_ids] = True
        grad[~keep] = 0
        return grad

    def lm_head_grad_hook(grad):
        grad = grad.clone()
        keep = torch.zeros(grad.shape[1], dtype=torch.bool, device=grad.device)
        keep[trainable_token_ids] = True
        grad[:, ~keep] = 0
        return grad

    model.token_embeddings.embedding_weights.register_hook(embedding_grad_hook)
    model.lm_head.W.register_hook(lm_head_grad_hook)

    # 解冻 HiRA 的 A/B 矩阵
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True

    model.to(device)
    model.train()

    # 优化器
    optimizer = modules.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # WandB 初始化
    with open("./pswd.json", encoding="utf-8") as f:
        os.environ["WANDB_API_KEY"] = json.load(f)["wandb-api-key"]
    wandb.init(
        project="simple_hira_sft",
        config={
            "data_pattern": data_pattern,
            "max_iters": max_iters,
            "batch_size": batch_size,
            "lr": max_learning_rate,
        },
        name=f"HiRA-r32_simple_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    print("device:", device)
    print(f"train_examples: {len(train_examples)}, val_examples: {len(val_examples)}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 混合精度
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    # 评估函数
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, split_examples in [("train", train_examples), ("val", val_examples)]:
            losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                x, y = make_batch(split_examples, batch_size, context_length, device, tokenizer)
                with autocast:
                    losses[k] = modules.run_cross_entropy_for_gem(model(x), y, ignore_index=-666)
            out[split] = losses.mean().item()
        model.train()
        return out

    # 训练循环
    previous_ckpt_path = ""
    tokens_per_iter = batch_size * context_length
    t0 = time.time()

    for it in range(start_iter, max_iters + 1):
        # 学习率调度
        lr = modules.run_get_lr_cosine_schedule(
            it=it - start_iter + 1,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = make_batch(train_examples, batch_size, context_length, device, tokenizer)
        with autocast:
            loss = modules.run_cross_entropy_for_gem(model(x), y, ignore_index=-666)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        modules.run_gradient_clipping(model.parameters(), max_l2_norm=max_grad_norm)
        optimizer.step()

        # 日志
        if it % log_interval == 0:
            dt = time.time() - t0
            a_norm, b_norm = get_hira_ab_norm(model)
            update_ratio = get_hira_update_ratio(model)
            print(
                f"iter {it:6d} loss {loss.item():.4f} lr {lr:.2e} | "
                f"tok/s {tokens_per_iter * log_interval / dt:.0f} | "
                f"update_ratio {update_ratio:.4f}"
            )
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "tok/s": tokens_per_iter * log_interval / dt,
                "adapter/A_norm": a_norm,
                "adapter/B_norm": b_norm,
                "adapter/update_ratio": update_ratio,
            }, step=it)
            t0 = time.time()

        # 评估
        if it % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"[eval] iter {it:8d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )
            wandb.log({"val_loss": losses["val"]}, step=it)
            inp = "<|user|>\nWhat is the capital of China?\n<|assistant|>\n"
            inp = tokenizer.encode(inp)
            out = modules.generating(
                model=model,
                enc_user_prompt=inp,
                end_token=31999,
                context_len=1024,
                max_token=32,
                temperature=0.2,
            )
            print(tokenizer.decode(out))

        # 保存 checkpoint
        if it % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/simple_hira_iter_{it}.pt"
            if previous_ckpt_path:
                os.remove(previous_ckpt_path)
            modules.run_save_checkpoint(model, optimizer, it, ckpt_path)
            previous_ckpt_path = ckpt_path
            print(f"saved checkpoint to {ckpt_path}")

    # 最终保存
    torch.save({"model": model.state_dict(), "it": max_iters}, final_ckpt)
    print(f"Saved final model to {final_ckpt}")
    wandb.finish()


if __name__ == "__main__":
    main()