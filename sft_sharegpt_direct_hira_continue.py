import contextlib
import datetime
import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import wandb

import modules
from sft_chat_templetes import (
    HiRALinear,
    append_jsonl,
    encode_chat_example,
    get_hira_ab_norm,
    get_hira_update_ratio,
    get_next_token_diagnostics,
    greedy_generate_ids,
    load_tokenizer,
    run_sft_loss,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _checkpoint_iter(path):
    try:
        name = os.path.basename(path)
        return int(name.rsplit("_iter_", 1)[1].rsplit(".pt", 1)[0])
    except (IndexError, ValueError):
        return -1


def default_resume_ckpt_path():
    patterns = [
        "./checkpoints/RE_sft_EvolSft_HiRA_r_16-essay-gpt2med_iter_*.pt",
        "./checkpoints/RE_sft_EvolSft_HiRA_r_16_gpt2med_iter_*.pt",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths), key=_checkpoint_iter)
    if paths:
        return paths[-1]
    return "./checkpoints/RE_sft_EvolSft_HiRA_r_16-essay-gpt2med_iter_13000.pt"


SHAREGPT_DATA_PATH = os.environ.get("SHAREGPT_DATA_PATH", "./data/train.jsonl")
DIRECT_DATA_PATH = os.environ.get("DIRECT_DATA_PATH", "")
USE_SYNTHETIC_DIRECT = os.environ.get("USE_SYNTHETIC_DIRECT", "1") == "1"
DIRECT_DATA_RATIO = float(os.environ.get("DIRECT_DATA_RATIO", "0.30"))

SFT_RESUME_CKPT_PATH = os.environ.get("SFT_RESUME_CKPT_PATH", default_resume_ckpt_path())
SFT_RESUME_OPTIM = os.environ.get("SFT_RESUME_OPTIM", "0") == "1"
SFT_RESET_LR_SCHEDULE = os.environ.get("SFT_RESET_LR_SCHEDULE", "1") == "1"
SFT_EXTRA_ITERS = int(os.environ.get("SFT_EXTRA_ITERS", "5000"))
SFT_TARGET_ITER = os.environ.get("SFT_TARGET_ITER", "")

SFT_CKPT_PREFIX = os.environ.get(
    "SFT_CKPT_PREFIX",
    "sharegpt_direct_hira_r16_from_current",
)
SFT_EVAL_LOG_PATH = os.environ.get(
    "SFT_EVAL_LOG_PATH",
    f"./logs/{SFT_CKPT_PREFIX}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
)
SFT_BEST_CKPT_PATH = os.environ.get(
    "SFT_BEST_CKPT_PATH",
    f"./checkpoints/{SFT_CKPT_PREFIX}_best_diag.pt",
)

FIRST_TOKEN_EVAL_N = int(os.environ.get("FIRST_TOKEN_EVAL_N", "100"))
DIAG_TOP_K = int(os.environ.get("DIAG_TOP_K", "10"))


def load_json_or_jsonl(path):
    if path == "":
        return []
    if path.endswith(".jsonl"):
        items = []
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_sharegpt_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations", [])
            for i in range(0, len(conv) - 1, 2):
                user = conv[i]
                assistant = conv[i + 1]
                if user.get("user") != "human" or assistant.get("user") != "gpt":
                    continue
                instruction = str(user.get("text", "")).strip()
                output = str(assistant.get("text", "")).strip()
                if instruction and output:
                    pairs.append({"instruction": instruction, "output": output})
    return pairs


def build_synthetic_direct_examples():
    examples = []

    colors = [
        "Blue", "Red", "Green", "Yellow", "Black", "White", "Purple",
        "Orange", "Pink", "Brown", "Gray", "Cyan", "Magenta", "Violet",
    ]
    color_templates = [
        "Say one word: {x}",
        "Answer with one word: {x}",
        "Repeat exactly one word: {x}",
        "Return only this word: {x}",
        "What word should you say? {x}",
    ]
    for color in colors:
        for template in color_templates:
            examples.append({"instruction": template.format(x=color), "output": color})

    for a in range(0, 21):
        for b in range(0, 21):
            examples.append({"instruction": f"{a}+{b}=?", "output": str(a + b)})
            if a >= b:
                examples.append({"instruction": f"{a}-{b}=?", "output": str(a - b)})

    for a in range(0, 13):
        for b in range(0, 13):
            examples.append({"instruction": f"{a}*{b}=?", "output": str(a * b)})

    short_qa = [
        ("what should i prepare for a math test?", "Review formulas, practice problems, bring pencils, and sleep well."),
        ("Name the capital of France.", "Paris"),
        ("What color is the sky on a clear day?", "Blue"),
        ("Answer yes or no: is water wet?", "Yes"),
        ("Answer yes or no: is fire cold?", "No"),
        ("Give one word for a young dog.", "Puppy"),
        ("Give one word for frozen water.", "Ice"),
        ("What is 2+2? Answer only the number.", "4"),
        ("What is 7 minus 3? Answer only the number.", "4"),
        ("What is 5 times 6? Answer only the number.", "30"),
    ]
    for instruction, output in short_qa:
        examples.append({"instruction": instruction, "output": output})

    return examples


def normalize_direct_examples(items):
    examples = []
    for item in items:
        if "instruction" in item and "output" in item:
            instruction = str(item["instruction"]).strip()
            output = str(item["output"]).strip()
        elif "prompt" in item and "response" in item:
            instruction = str(item["prompt"]).strip()
            output = str(item["response"]).strip()
        else:
            continue
        if instruction and output:
            examples.append({"instruction": instruction, "output": output})
    return examples


def split_examples(examples, seed):
    examples = list(examples)
    rng = random.Random(seed)
    rng.shuffle(examples)
    valid_end = max(1, math.floor(len(examples) * 0.1)) if examples else 0
    return examples[valid_end:], examples[:valid_end]


def make_batch_from_examples(
    examples,
    batch_size,
    context_length,
    device,
    tokenizer,
    ignore_index=-666,
):
    if not examples:
        raise ValueError("cannot sample from an empty dataset")

    pad_token_id = tokenizer.vocab_inv["<|pad|>".encode("utf-8")]
    x = torch.full(
        (batch_size, context_length),
        pad_token_id,
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
        for attempt in range(10000):
            example = random.choice(examples)
            prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)
            if len(full_tokens) <= context_length:
                break
        else:
            raise RuntimeError(
                f"failed to sample an example within context_length={context_length}"
            )

        labels = [ignore_index] * len(full_tokens)
        for i in range(len(prompt_tokens), len(full_tokens)):
            labels[i] = full_tokens[i]

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


def make_mixed_batch(
    sharegpt_examples,
    direct_examples,
    direct_ratio,
    batch_size,
    context_length,
    device,
    tokenizer,
    ignore_index=-666,
):
    pad_token_id = tokenizer.vocab_inv["<|pad|>".encode("utf-8")]
    x = torch.full(
        (batch_size, context_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    y = torch.full(
        (batch_size, context_length),
        ignore_index,
        dtype=torch.long,
        device=device,
    )
    counts = {
        "sharegpt": 0,
        "direct": 0,
        "sharegpt_label_tokens": 0,
        "direct_label_tokens": 0,
    }

    for b in range(batch_size):
        use_direct = bool(direct_examples) and random.random() < direct_ratio
        source = "direct" if use_direct else "sharegpt"
        pool = direct_examples if use_direct else sharegpt_examples
        counts[source] += 1

        for attempt in range(10000):
            example = random.choice(pool)
            prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)
            if len(full_tokens) <= context_length:
                break
        else:
            raise RuntimeError(
                f"failed to sample a {source} example within context_length={context_length}"
            )

        labels = [ignore_index] * len(full_tokens)
        for i in range(len(prompt_tokens), len(full_tokens)):
            labels[i] = full_tokens[i]
        counts[f"{source}_label_tokens"] += len(full_tokens) - len(prompt_tokens)

        x[b, : len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long, device=device)
        y[b, : len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)

    return x, y, counts


@torch.no_grad()
def estimate_loss_for_examples(
    model,
    examples,
    tokenizer,
    batch_size,
    context_length,
    eval_iters,
    first_token_weight,
    device,
):
    if not examples:
        return None

    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    weighted_losses = torch.zeros(eval_iters, device=device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    for k in range(eval_iters):
        x, y = make_batch_from_examples(
            examples=examples,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            tokenizer=tokenizer,
        )
        with autocast:
            logits = model(x)
            loss = modules.run_cross_entropy_for_gem(logits, y, ignore_index=-666)
            weighted_loss = run_sft_loss(
                logits,
                y,
                ignore_index=-666,
                first_token_weight=first_token_weight,
            )
        losses[k] = loss
        weighted_losses[k] = weighted_loss

    model.train()
    return {
        "loss": losses.mean().item(),
        "weighted_loss": weighted_losses.mean().item(),
    }


@torch.no_grad()
def estimate_first_token_rank_examples(
    model,
    examples,
    tokenizer,
    context_length,
    n=100,
):
    if not examples:
        return None

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ranks = []

    for example in examples:
        prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)
        if len(full_tokens) > context_length:
            continue
        if len(prompt_tokens) >= len(full_tokens):
            continue

        x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        target = full_tokens[len(prompt_tokens)]
        rank = int((logits > logits[target]).sum().item()) + 1
        ranks.append(rank)
        if len(ranks) >= n:
            break

    if was_training:
        model.train()

    if not ranks:
        return None

    return {
        "rank": sum(ranks) / len(ranks),
        "top1": sum(rank == 1 for rank in ranks) / len(ranks),
        "n": len(ranks),
    }


def build_model(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
    model = modules.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    for layer in model.layers:
        layer.attn.q_proj = HiRALinear(layer.attn.q_proj)
        layer.attn.k_proj = HiRALinear(layer.attn.k_proj)
        layer.attn.v_proj = HiRALinear(layer.attn.v_proj)
        layer.ffn.w2 = HiRALinear(layer.ffn.w2)
        layer.ffn.w3 = HiRALinear(layer.ffn.w3)
    return model


def enable_only_adapter_and_special_tokens(model, base_vocab_size=32000):
    for param in model.parameters():
        param.requires_grad = False

    model.token_embeddings.embedding_weights.requires_grad = True
    model.lm_head.W.requires_grad = True

    def embedding_grad_hook(grad):
        grad = grad.clone()
        grad[:base_vocab_size] = 0
        return grad

    def lm_head_grad_hook(grad):
        grad = grad.clone()
        grad[:, :base_vocab_size] = 0
        return grad

    model.token_embeddings.embedding_weights.register_hook(embedding_grad_hook)
    model.lm_head.W.register_hook(lm_head_grad_hook)

    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = 0
    world_size = 1

    context_length = int(os.environ.get("SFT_CONTEXT_LENGTH", "1024"))
    batch_size = int(os.environ.get("SFT_BATCH_SIZE", "8"))
    d_model = 1024
    num_layers = 24
    num_heads = 16
    d_ff = 2752
    rope_theta = 10000

    eval_interval = int(os.environ.get("SFT_EVAL_INTERVAL", "250"))
    eval_iters = int(os.environ.get("SFT_EVAL_ITERS", "30"))
    log_interval = int(os.environ.get("SFT_LOG_INTERVAL", "50"))
    checkpoint_interval = int(os.environ.get("SFT_CHECKPOINT_INTERVAL", "500"))
    max_learning_rate = float(os.environ.get("SFT_MAX_LR", "1e-4"))
    min_learning_rate = float(os.environ.get("SFT_MIN_LR", "2e-5"))
    warmup_iters = int(os.environ.get("SFT_WARMUP_ITERS", "100"))
    weight_decay = float(os.environ.get("SFT_WEIGHT_DECAY", "0.0"))
    betas = (0.9, 0.95)
    eps = 1e-8
    max_grad_norm = float(os.environ.get("SFT_MAX_GRAD_NORM", "1.0"))
    first_token_weight = float(os.environ.get("SFT_FIRST_TOKEN_WEIGHT", "30.0"))

    seed = int(os.environ.get("SFT_SEED", "1337")) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer, vocab_size = load_tokenizer()
    assistant_token_id = tokenizer.vocab_inv["<|assistant|>".encode("utf-8")]

    sharegpt_pairs = load_sharegpt_pairs(SHAREGPT_DATA_PATH)
    sharegpt_train, sharegpt_val = split_examples(sharegpt_pairs, seed)

    direct_examples = []
    if USE_SYNTHETIC_DIRECT:
        direct_examples.extend(build_synthetic_direct_examples())
    direct_examples.extend(normalize_direct_examples(load_json_or_jsonl(DIRECT_DATA_PATH)))
    direct_train, direct_val = split_examples(direct_examples, seed + 1)

    model = build_model(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    enable_only_adapter_and_special_tokens(model)

    resume_obj = torch.load(SFT_RESUME_CKPT_PATH, map_location="cpu")
    model.load_state_dict(resume_obj["model"])
    resume_iter = int(resume_obj.get("it", -1))
    start_iter = resume_iter + 1
    if SFT_TARGET_ITER.strip():
        max_iters = int(SFT_TARGET_ITER)
    else:
        max_iters = resume_iter + SFT_EXTRA_ITERS
    cosine_cycle_iters = max_iters - start_iter + 1 if SFT_RESET_LR_SCHEDULE else max_iters

    model.to(device)

    optimizer = modules.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    if SFT_RESUME_OPTIM and "optim" in resume_obj:
        optimizer.load_state_dict(resume_obj["optim"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("./pswd.json", encoding="utf-8") as file:
        pswds = json.load(file)
    os.environ["WANDB_API_KEY"] = pswds["wandb-api-key"]
    wandb.init(
        project="gpt2vision_sft_chat_templetes",
        config={
            "script": os.path.basename(__file__),
            "resume_ckpt_path": SFT_RESUME_CKPT_PATH,
            "resume_optimizer": SFT_RESUME_OPTIM,
            "reset_lr_schedule": SFT_RESET_LR_SCHEDULE,
            "start_iter": start_iter,
            "target_iter": max_iters,
            "sharegpt_data_path": SHAREGPT_DATA_PATH,
            "sharegpt_pairs": len(sharegpt_pairs),
            "direct_data_path": DIRECT_DATA_PATH,
            "synthetic_direct": USE_SYNTHETIC_DIRECT,
            "direct_examples": len(direct_examples),
            "direct_data_ratio": DIRECT_DATA_RATIO,
            "eval_log_path": SFT_EVAL_LOG_PATH,
            "ckpt_prefix": SFT_CKPT_PREFIX,
            "best_ckpt_path": SFT_BEST_CKPT_PATH,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_iters": warmup_iters,
            "max_grad_norm": max_grad_norm,
            "first_token_weight": first_token_weight,
        },
        name=f"[sharegpt-direct] HiRA-r16 + {nowtime}",
        save_code=True,
    )

    if rank == 0:
        print("device:", device)
        print("resume_ckpt_path:", SFT_RESUME_CKPT_PATH)
        print("resume_iter:", resume_iter, "start_iter:", start_iter, "target_iter:", max_iters)
        print("sharegpt_pairs:", len(sharegpt_pairs), "train:", len(sharegpt_train), "val:", len(sharegpt_val))
        print("direct_examples:", len(direct_examples), "train:", len(direct_train), "val:", len(direct_val))
        print("direct_data_ratio:", DIRECT_DATA_RATIO)
        print("eval_log_path:", SFT_EVAL_LOG_PATH)
        print("ckpt_prefix:", SFT_CKPT_PREFIX)
        print("initial update_ratio:", get_hira_update_ratio(model))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable={trainable:,} total={total:,} ratio={100 * trainable / total:.4f}%")

    diagnostic_prompts = [
        {"name": "one_word_blue", "prompt": "<|user|>\nSay one word: Blue\n<|assistant|>\n", "target": "Blue"},
        {"name": "math_test", "prompt": "<|user|>\nwhat should i prepare for a math test?\n<|assistant|>\n", "target": None},
        {"name": "two_plus_two", "prompt": "<|user|>\n2+2=?\n<|assistant|>\n", "target": "4"},
    ]

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    tokens_per_iter = batch_size * context_length * world_size
    best_diag_rank = float("inf")
    prev_ckpt_path = ""
    model.train()

    import time
    t0 = time.time()

    for it in range(start_iter, max_iters + 1):
        schedule_it = it - start_iter + 1 if SFT_RESET_LR_SCHEDULE else it
        lr = modules.run_get_lr_cosine_schedule(
            it=schedule_it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y, source_counts = make_mixed_batch(
            sharegpt_examples=sharegpt_train,
            direct_examples=direct_train,
            direct_ratio=DIRECT_DATA_RATIO,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            tokenizer=tokenizer,
        )

        with autocast:
            logits = model(x)
            loss = run_sft_loss(
                logits,
                y,
                ignore_index=-666,
                first_token_weight=first_token_weight,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        modules.run_gradient_clipping(model.parameters(), max_l2_norm=max_grad_norm)
        optimizer.step()

        hira_update_ratio = get_hira_update_ratio(model)

        if rank == 0 and it % log_interval == 0:
            dt = time.time() - t0
            tokens_processed = it * tokens_per_iter
            iter_since_resume = it - start_iter + 1
            tokens_since_resume = iter_since_resume * tokens_per_iter
            tok_s = tokens_per_iter * log_interval / dt
            a_norm, b_norm = get_hira_ab_norm(model)
            train_record = {
                "event": "train",
                "time": datetime.datetime.now().isoformat(),
                "iter": it,
                "iter_since_resume": iter_since_resume,
                "tokens": tokens_processed,
                "tokens_since_resume": tokens_since_resume,
                "lr": lr,
                "loss": loss.item(),
                "tok_s": tok_s,
                "hira_update_ratio": hira_update_ratio,
                "adapter_A_norm": a_norm,
                "adapter_B_norm": b_norm,
                "source_counts": source_counts,
                "direct_label_token_ratio": (
                    source_counts["direct_label_tokens"]
                    / max(1, source_counts["direct_label_tokens"] + source_counts["sharegpt_label_tokens"])
                ),
            }
            append_jsonl(SFT_EVAL_LOG_PATH, train_record)
            print(
                f"iter {it} loss {loss.item():.4f} lr {lr:.6e} | "
                f"tok/s {tok_s:.0f} | update_ratio {hira_update_ratio:.6f} | "
                f"src {source_counts}"
            )
            wandb.log({
                "iter": it,
                "lr": lr,
                "loss": loss.item(),
                "tok/s": tok_s,
                "adapter/update_ratio": hira_update_ratio,
                "adapter/A_norm": a_norm,
                "adapter/B_norm": b_norm,
                "data/direct_batch_count": source_counts["direct"],
                "data/sharegpt_batch_count": source_counts["sharegpt"],
                "data/direct_label_token_ratio": source_counts["direct_label_tokens"]
                / max(1, source_counts["direct_label_tokens"] + source_counts["sharegpt_label_tokens"]),
                "data/direct_label_tokens": source_counts["direct_label_tokens"],
                "data/sharegpt_label_tokens": source_counts["sharegpt_label_tokens"],
            })
            t0 = time.time()

        if it % eval_interval == 0:
            sharegpt_train_loss = estimate_loss_for_examples(
                model, sharegpt_train, tokenizer, batch_size, context_length,
                eval_iters, first_token_weight, device,
            )
            sharegpt_val_loss = estimate_loss_for_examples(
                model, sharegpt_val, tokenizer, batch_size, context_length,
                eval_iters, first_token_weight, device,
            )
            direct_train_loss = estimate_loss_for_examples(
                model, direct_train, tokenizer, batch_size, context_length,
                eval_iters, first_token_weight, device,
            ) if direct_train else None
            direct_val_loss = estimate_loss_for_examples(
                model, direct_val, tokenizer, batch_size, context_length,
                eval_iters, first_token_weight, device,
            ) if direct_val else None
            sharegpt_first = estimate_first_token_rank_examples(
                model, sharegpt_val, tokenizer, context_length, n=FIRST_TOKEN_EVAL_N,
            )
            direct_first = estimate_first_token_rank_examples(
                model, direct_val, tokenizer, context_length, n=FIRST_TOKEN_EVAL_N,
            ) if direct_val else None

            assistant_embedding_norm = model.token_embeddings.embedding_weights[
                assistant_token_id
            ].norm().item()
            eval_greedy_table = wandb.Table(
                columns=["iter", "name", "prompt", "target", "target_rank", "target_prob", "generated", "generated_len"]
            )
            greedy_records = []
            for diagnostic in diagnostic_prompts:
                prompt = diagnostic["prompt"]
                target_text = diagnostic["target"]
                generated = greedy_generate_ids(model, tokenizer, prompt, context_length, max_new_tokens=48)
                generated_text = tokenizer.decode(generated)
                next_diag = get_next_token_diagnostics(
                    model, tokenizer, prompt, target_text=target_text, top_k=DIAG_TOP_K,
                )
                target_diag = next_diag["target"]
                greedy_records.append({
                    "name": diagnostic["name"],
                    "prompt": prompt,
                    "target": target_text,
                    "generated": generated_text,
                    "generated_ids": generated,
                    "generated_len": len(generated),
                    "next_token": next_diag,
                })
                eval_greedy_table.add_data(
                    it,
                    diagnostic["name"],
                    prompt,
                    target_text,
                    None if target_diag is None else target_diag["rank"],
                    None if target_diag is None else target_diag["prob"],
                    generated_text,
                    len(generated),
                )

            diag_blue = next(
                (r["next_token"]["target"] for r in greedy_records if r["name"] == "one_word_blue"),
                None,
            )
            diag_rank = float("inf") if diag_blue is None else diag_blue["rank"]
            is_best = diag_rank < best_diag_rank
            best_rank_before_eval = None if math.isinf(best_diag_rank) else best_diag_rank

            eval_record = {
                "event": "eval",
                "time": datetime.datetime.now().isoformat(),
                "iter": it,
                "iter_since_resume": it - start_iter + 1,
                "lr": lr,
                "sharegpt_train_loss": sharegpt_train_loss,
                "sharegpt_val_loss": sharegpt_val_loss,
                "direct_train_loss": direct_train_loss,
                "direct_val_loss": direct_val_loss,
                "sharegpt_first_token": sharegpt_first,
                "direct_first_token": direct_first,
                "assistant_embedding_norm": assistant_embedding_norm,
                "hira_update_ratio": hira_update_ratio,
                "is_best_diag": is_best,
                "best_diag_rank_before_eval": best_rank_before_eval,
                "greedy": greedy_records,
            }
            append_jsonl(SFT_EVAL_LOG_PATH, eval_record)

            if rank == 0:
                print(
                    f"[eval] iter {it:8d} | "
                    f"sharegpt val {sharegpt_val_loss['loss']:.4f} "
                    f"sharegpt val_w {sharegpt_val_loss['weighted_loss']:.4f}"
                )
                if direct_val_loss is not None:
                    print(
                        f"[eval-direct] iter {it:8d} | "
                        f"direct val {direct_val_loss['loss']:.4f} "
                        f"direct val_w {direct_val_loss['weighted_loss']:.4f}"
                    )
                if sharegpt_first is not None:
                    print(
                        f"[first-token-sharegpt] iter {it:8d} | "
                        f"rank {sharegpt_first['rank']:.2f} | "
                        f"top1 {sharegpt_first['top1']:.4f} | n {sharegpt_first['n']}"
                    )
                if direct_first is not None:
                    print(
                        f"[first-token-direct] iter {it:8d} | "
                        f"rank {direct_first['rank']:.2f} | "
                        f"top1 {direct_first['top1']:.4f} | n {direct_first['n']}"
                    )
                for record in greedy_records:
                    print(f"[greedy] {record['name']} {record['prompt']!r} -> {record['generated']!r}")
                    target_diag = record["next_token"]["target"]
                    if target_diag is not None:
                        print(
                            f"[next-token-target] {record['name']} "
                            f"target={target_diag['first_token_text']!r} "
                            f"rank={target_diag['rank']} prob={target_diag['prob']:.6g}"
                        )
                print(f"[assistant-emb] norm {assistant_embedding_norm:.4f}")

                wandb_log = {
                    "eval/sharegpt_train_loss": sharegpt_train_loss["loss"],
                    "eval/sharegpt_train_weighted_loss": sharegpt_train_loss["weighted_loss"],
                    "eval/sharegpt_val_loss": sharegpt_val_loss["loss"],
                    "eval/sharegpt_val_weighted_loss": sharegpt_val_loss["weighted_loss"],
                    "eval/assistant_embedding_norm": assistant_embedding_norm,
                    "eval/hira_update_ratio": hira_update_ratio,
                    "eval/greedy_outputs": eval_greedy_table,
                    "eval/is_best_diag": int(is_best),
                }
                if direct_val_loss is not None:
                    wandb_log.update({
                        "eval/direct_train_loss": direct_train_loss["loss"],
                        "eval/direct_train_weighted_loss": direct_train_loss["weighted_loss"],
                        "eval/direct_val_loss": direct_val_loss["loss"],
                        "eval/direct_val_weighted_loss": direct_val_loss["weighted_loss"],
                    })
                if sharegpt_first is not None:
                    wandb_log.update({
                        "first_token/sharegpt_rank": sharegpt_first["rank"],
                        "first_token/sharegpt_top1": sharegpt_first["top1"],
                    })
                if direct_first is not None:
                    wandb_log.update({
                        "first_token/direct_rank": direct_first["rank"],
                        "first_token/direct_top1": direct_first["top1"],
                    })
                for record in greedy_records:
                    name = record["name"]
                    wandb_log[f"greedy/{name}/generated"] = record["generated"]
                    target_diag = record["next_token"]["target"]
                    if target_diag is not None:
                        wandb_log[f"greedy/{name}/target_rank"] = target_diag["rank"]
                        wandb_log[f"greedy/{name}/target_prob"] = target_diag["prob"]
                wandb.log(wandb_log)

                if is_best:
                    best_diag_rank = diag_rank
                    best_dir = os.path.dirname(SFT_BEST_CKPT_PATH)
                    if best_dir:
                        os.makedirs(best_dir, exist_ok=True)
                    modules.run_save_checkpoint(model, optimizer, it, SFT_BEST_CKPT_PATH)
                    print(f"saved best diagnostic checkpoint to {SFT_BEST_CKPT_PATH} rank={best_diag_rank:.2f}")

        if rank == 0 and it > 0 and it % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/{SFT_CKPT_PREFIX}_iter_{it}.pt"
            if prev_ckpt_path:
                os.remove(prev_ckpt_path)
            prev_ckpt_path = ckpt_path
            modules.run_save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    final_path = f"./data/{SFT_CKPT_PREFIX}_final.pt"
    torch.save({"model": model.state_dict()}, final_path)
    print(f"saved final weights to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
