import contextlib
import datetime
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


def load_sharegpt_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            conversation = json.loads(line).get("conversations", [])
            for i in range(0, len(conversation) - 1, 2):
                user = conversation[i]
                assistant = conversation[i + 1]
                if user.get("user") != "human" or assistant.get("user") != "gpt":
                    continue
                instruction = str(user.get("text", "")).strip()
                output = str(assistant.get("text", "")).strip()
                if instruction and output:
                    pairs.append({"instruction": instruction, "output": output})
    return pairs


def split_examples(examples, seed):
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    valid_end = max(1, len(examples) // 10)
    return examples[valid_end:], examples[:valid_end]


def load_instruction_jsonl(path):
    examples = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction = str(obj.get("instruction", "")).strip()
            output = str(obj.get("output", "")).strip()
            if instruction and output:
                examples.append({"instruction": instruction, "output": output})
    return examples


def make_batch(
    examples,
    batch_size,
    context_length,
    device,
    tokenizer,
    ignore_index=-666,
    short_examples=None,
    short_ratio=0.0,
):
    if not examples:
        raise ValueError("cannot sample from an empty dataset")

    x = torch.full(
        (batch_size, context_length),
        tokenizer.vocab_inv["<|pad|>".encode("utf-8")],
        dtype=torch.long,
        device=device,
    )
    y = torch.full(
        (batch_size, context_length),
        ignore_index,
        dtype=torch.long,
        device=device,
    )

    for b in range(batch_size):
        pool = short_examples if short_examples and random.random() < short_ratio else examples
        for _ in range(10000):
            prompt_tokens, full_tokens = encode_chat_example(
                random.choice(pool),
                tokenizer,
            )
            if len(full_tokens) <= context_length:
                break
        else:
            raise RuntimeError(
                f"failed to sample a ShareGPT example within context_length={context_length}"
            )

        labels = [ignore_index] * len(full_tokens)
        labels[len(prompt_tokens):] = full_tokens[len(prompt_tokens):]

        x[b, : len(full_tokens)] = torch.tensor(
            full_tokens,
            dtype=torch.long,
            device=device,
        )
        y[b, : len(labels)] = torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )

    return x, y


def sft_loss(logits, labels, ignore_index=-666, first_token_weight=1.0):
    if first_token_weight <= 1.0:
        return modules.run_cross_entropy_for_gem(
            logits,
            labels,
            ignore_index=ignore_index,
        )

    batch_size, _, vocab_size = logits.shape
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(batch_size, -1)
    mask = labels != ignore_index
    weights = mask.float()
    for b in range(batch_size):
        positions = torch.nonzero(mask[b], as_tuple=False)
        if positions.numel() > 0:
            weights[b, positions[0].item()] = first_token_weight
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context_length = 1024
    batch_size = 8
    eval_interval = 250
    eval_iters = 30
    log_interval = 50
    checkpoint_interval = 500
    max_learning_rate = 3e-5
    min_learning_rate = 5e-6
    warmup_iters = 200
    extra_iters = 10000
    first_token_weight = 5.0
    max_grad_norm = 0.5
    ckpt_path = "./data/sharegpt_only_hira_continue_final.pt"
    fallback_resume_iter = 30000
    short_instruct_path = "./data/short_instruct_deepseek.jsonl"
    short_ratio = 0.70

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    tokenizer, vocab_size = load_tokenizer()
    sharegpt_train, sharegpt_val = split_examples(
        load_sharegpt_pairs("./data/train.jsonl"),
        1337,
    )
    short_train, short_val = split_examples(
        load_instruction_jsonl(short_instruct_path),
        1338,
    )
    if not short_train:
        raise RuntimeError(f"no valid short-instruct examples loaded from {short_instruct_path}")

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
        layer.attn.q_proj = HiRALinear(layer.attn.q_proj)
        layer.attn.k_proj = HiRALinear(layer.attn.k_proj)
        layer.attn.v_proj = HiRALinear(layer.attn.v_proj)
        layer.ffn.w2 = HiRALinear(layer.ffn.w2)
        layer.ffn.w3 = HiRALinear(layer.ffn.w3)

    for param in model.parameters():
        param.requires_grad = False
    model.token_embeddings.embedding_weights.requires_grad = True
    model.lm_head.W.requires_grad = True

    def embedding_grad_hook(grad):
        grad = grad.clone()
        grad[:32000] = 0
        return grad

    def lm_head_grad_hook(grad):
        grad = grad.clone()
        grad[:, :32000] = 0
        return grad

    model.token_embeddings.embedding_weights.register_hook(embedding_grad_hook)
    model.lm_head.W.register_hook(lm_head_grad_hook)
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True

    resume_obj = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(resume_obj["model"])
    resume_iter = int(resume_obj.get("it", fallback_resume_iter))
    start_iter = resume_iter + 1
    max_iters = resume_iter + extra_iters
    cosine_cycle_iters = max_iters - start_iter + 1
    model.to(device)
    model.train()

    optimizer = modules.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    with open("./pswd.json", encoding="utf-8") as file:
        os.environ["WANDB_API_KEY"] = json.load(file)["wandb-api-key"]
    wandb.init(
        project="gpt2vision_sft_chat_templetes",
        config={
            "resume_ckpt_path": ckpt_path,
            "target_iter": max_iters,
            "batch_size": batch_size,
            "lr": max_learning_rate,
        },
        name="[sharegpt-only] HiRA-r16 continue "
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        save_code=False,
    )

    print("device:", device)
    print("resume_ckpt_path:", ckpt_path)
    print("resume_iter:", resume_iter, "start_iter:", start_iter, "target_iter:", max_iters)
    print("sharegpt train:", len(sharegpt_train), "val:", len(sharegpt_val))
    print("short train:", len(short_train), "val:", len(short_val), "ratio:", short_ratio)
    print("initial update_ratio:", get_hira_update_ratio(model))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable={trainable:,} total={total:,} ratio={100 * trainable / total:.4f}%")

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, examples in (
            ("train", sharegpt_train),
            ("val", sharegpt_val),
            ("short_val", short_val),
        ):
            losses = torch.zeros(eval_iters, device=device)
            weighted_losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                x, y = make_batch(examples, batch_size, context_length, device, tokenizer)
                with autocast:
                    logits = model(x)
                    losses[k] = modules.run_cross_entropy_for_gem(
                        logits,
                        y,
                        ignore_index=-666,
                    )
                    weighted_losses[k] = sft_loss(
                        logits,
                        y,
                        ignore_index=-666,
                        first_token_weight=first_token_weight,
                    )
            out[split] = {
                "loss": losses.mean().item(),
                "weighted_loss": weighted_losses.mean().item(),
            }
        model.train()
        return out

    previous_ckpt_path = ""
    tokens_per_iter = batch_size * context_length
    t0 = time.time()

    for it in range(start_iter, max_iters + 1):
        lr = modules.run_get_lr_cosine_schedule(
            it=it - start_iter + 1,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = make_batch(
            sharegpt_train,
            batch_size,
            context_length,
            device,
            tokenizer,
            short_examples=short_train,
            short_ratio=short_ratio,
        )
        with autocast:
            loss = sft_loss(
                model(x),
                y,
                ignore_index=-666,
                first_token_weight=first_token_weight,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        modules.run_gradient_clipping(model.parameters(), max_l2_norm=max_grad_norm)
        optimizer.step()

        if it % log_interval == 0:
            dt = time.time() - t0
            a_norm, b_norm = get_hira_ab_norm(model)
            update_ratio = get_hira_update_ratio(model)
            print(
                f"iter {it} loss {loss.item():.4f} lr {lr:.6e} | "
                f"tok/s {tokens_per_iter * log_interval / dt:.0f} | "
                f"update_ratio {update_ratio:.6f}"
            )
            wandb.log({"loss": loss.item()}, step=it)
            t0 = time.time()

        if it % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"[eval] iter {it:8d} | "
                f"sharegpt train {losses['train']['loss']:.4f} "
                f"train_w {losses['train']['weighted_loss']:.4f} | "
                f"sharegpt val {losses['val']['loss']:.4f} "
                f"val_w {losses['val']['weighted_loss']:.4f} | "
                f"short val {losses['short_val']['loss']:.4f}"
            )
            wandb.log({"val_loss": losses["val"]["loss"]}, step=it)

        if it % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_save_path = f"checkpoints/sharegpt_only_hira_continue_iter_{it}.pt"
            if previous_ckpt_path:
                os.remove(previous_ckpt_path)
            modules.run_save_checkpoint(model, optimizer, it, ckpt_save_path)
            previous_ckpt_path = ckpt_save_path
            print(f"saved checkpoint to {ckpt_save_path}")

    torch.save(
        {"model": model.state_dict(), "it": max_iters},
        "./data/sharegpt_only_hira_continue_final.pt",
    )
    print("saved final weights to ./data/sharegpt_only_hira_continue_final.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
