import contextlib
import datetime
import glob
import json
import os
import random
import time

import numpy as np
import torch

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


CLEAN_WORDS = [
    "cat", "dog", "sun", "hat", "run", "red", "hot", "ice",
    "apple", "happy", "moon", "star", "fish", "book", "tree", "cloud",
    "rain", "fire", "green", "river", "ocean", "stone", "light", "smile",
    "garden", "window", "bridge", "summer", "winter", "animal", "puzzle", "rocket",
]

CLEAN_PHRASES = [
    "Hello world",
    "Good morning",
    "Thank you",
    "How are you",
    "I am fine",
    "Well done",
    "Nice to meet you",
    "Keep going",
    "Open the door",
    "Read the book",
    "Draw a star",
    "Find the answer",
]

CLEAN_ONSETS = [
    "", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w",
    "br", "cl", "fl", "gr", "pl", "pr", "sl", "st", "tr", "ch", "sh",
]

CLEAN_NUCLEI = ["a", "e", "i", "o", "u", "ai", "ee", "oa", "oo", "ou", "ay"]

CLEAN_CODAS = ["", "n", "m", "s", "t", "r", "l", "d", "k", "p", "ck", "nd", "st", "ng"]

ADD_TEMPLATES = [
    "{a} + {b} =",
    "What is {a} plus {b}?",
    "Calculate {a} + {b}",
    "Add {a} and {b}",
    "Find the sum of {a} and {b}",
]

SUB_TEMPLATES = [
    "{a} - {b} =",
    "What is {a} minus {b}?",
    "Calculate {a} - {b}",
    "Subtract {b} from {a}",
    "Find the difference between {a} and {b}",
]

MUL_TEMPLATES = [
    "{a} * {b} =",
    "What is {a} times {b}?",
    "Calculate {a} * {b}",
    "Multiply {a} and {b}",
    "Find the product of {a} and {b}",
]

SAY_TEMPLATES = [
    "Say the word: {w}",
    "Repeat after me: {w}",
    "Just say {w}",
    "Echo: {w}",
    "Output the word: {w}",
]

COPY_TEMPLATES = [
    "Copy this: {p}",
    "Repeat this exactly: {p}",
    "Output the following: {p}",
    "Just copy: {p}",
]

SPELL_TEMPLATES = [
    "Spell the word '{w}'",
    "How do you spell '{w}'?",
    "Give me the spelling of '{w}'",
    "Spell '{w}' for me",
]

COUNT_TEMPLATES = [
    "How many letters are in '{w}'?",
    "Count the letters in '{w}'",
    "What is the length of the word '{w}'?",
    "How many characters in '{w}'?",
]


def infer_task(instruction: str) -> str:
    text = instruction.lower()
    if "spell" in text or "spelling" in text:
        return "spell"
    if "how many" in text or "count the letters" in text or "length of the word" in text:
        return "count"
    if any(key in text for key in ["copy", "repeat", "say", "echo", "output the word", "output the following"]):
        return "copy"
    return "arithmetic"


def make_clean_pseudoword(rng: random.Random, min_len: int = 4, max_len: int = 9):
    for _ in range(100):
        syllables = rng.randint(1, 3)
        word = ""
        for _ in range(syllables):
            word += rng.choice(CLEAN_ONSETS)
            word += rng.choice(CLEAN_NUCLEI)
            if rng.random() < 0.65:
                word += rng.choice(CLEAN_CODAS)
        if min_len <= len(word) <= max_len and word.isalpha():
            return word

    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(letters) for _ in range(rng.randint(min_len, max_len)))


def build_clean_word_pool(total_examples: int, seed: int):
    rng = random.Random(seed)
    target_words = max(128, total_examples // 2)
    words = set(CLEAN_WORDS)

    while len(words) < target_words:
        words.add(make_clean_pseudoword(rng))

    return sorted(words)


def build_clean_phrase_pool(words: list[str], total_examples: int, seed: int):
    rng = random.Random(seed)
    target_phrases = max(128, total_examples // 3)
    phrases = set(CLEAN_PHRASES)

    while len(phrases) < target_phrases:
        n = rng.randint(2, 4)
        phrase_words = rng.sample(words, n)
        phrase = " ".join(phrase_words)
        phrases.add(phrase[0].upper() + phrase[1:])

    return sorted(phrases)


def build_clean_diagnostic_examples(total: int = 128, seed: int = 1337, exclude_keys=None):
    rng = random.Random(seed)
    words = build_clean_word_pool(total, seed + 1)
    phrases = build_clean_phrase_pool(words, total, seed + 2)
    examples = []
    seen = set()
    exclude_keys = exclude_keys or set()

    def add(instruction, output, task):
        key = (instruction, output)
        if key in seen or key in exclude_keys:
            return False
        seen.add(key)
        examples.append({"instruction": instruction, "output": output, "task": task})
        return True

    targets = {
        "math": round(total * 0.40),
        "copy": round(total * 0.40),
        "spell": round(total * 0.10),
    }
    targets["count"] = total - sum(targets.values())

    def fill(kind, target, generator):
        made = 0
        attempts = 0
        max_attempts = max(1000, target * 500)
        while made < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"failed to generate enough unique {kind} examples: "
                    f"made={made}, target={target}, pool_words={len(words)}, pool_phrases={len(phrases)}"
            )
            sample = generator()
            task = "arithmetic" if kind == "math" else kind
            if add(sample["instruction"], sample["output"], task):
                made += 1

    def math_sample():
        op = rng.choice(["add", "sub", "mul"])
        if op == "add":
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            return {"instruction": rng.choice(ADD_TEMPLATES).format(a=a, b=b), "output": str(a + b)}
        if op == "sub":
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            if a < b:
                a, b = b, a
            return {"instruction": rng.choice(SUB_TEMPLATES).format(a=a, b=b), "output": str(a - b)}
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        return {"instruction": rng.choice(MUL_TEMPLATES).format(a=a, b=b), "output": str(a * b)}

    def copy_sample():
        if rng.random() < 0.6:
            word = rng.choice(words)
            return {"instruction": rng.choice(SAY_TEMPLATES).format(w=word), "output": word}
        phrase = rng.choice(phrases)
        return {"instruction": rng.choice(COPY_TEMPLATES).format(p=phrase), "output": phrase}

    def spell_sample():
        word = rng.choice(words)
        return {"instruction": rng.choice(SPELL_TEMPLATES).format(w=word), "output": "-".join(word)}

    def count_sample():
        word = rng.choice(words)
        return {"instruction": rng.choice(COUNT_TEMPLATES).format(w=word), "output": str(len(word))}

    fill("math", targets["math"], math_sample)
    fill("copy", targets["copy"], copy_sample)
    fill("spell", targets["spell"], spell_sample)
    fill("count", targets["count"], count_sample)

    rng.shuffle(examples)
    return examples


def build_generalization_split(train_total: int, val_total: int, seed: int = 1337):
    train_examples = build_clean_diagnostic_examples(train_total, seed=seed)
    train_keys = {(ex["instruction"], ex["output"]) for ex in train_examples}
    val_examples = build_clean_diagnostic_examples(val_total, seed=seed + 1, exclude_keys=train_keys)
    return train_examples, val_examples


def count_tasks(examples):
    counts = {"arithmetic": 0, "copy": 0, "spell": 0, "count": 0}
    for example in examples:
        task = example.get("task", infer_task(example["instruction"]))
        counts[task] = counts.get(task, 0) + 1
    return counts


def format_task_accuracy(acc):
    parts = []
    for task in ["arithmetic", "copy", "spell", "count"]:
        values = acc["by_task"].get(task)
        if values is None:
            continue
        parts.append(f"{task}:{values['tf_exact']:.3f}/{values['n']}")
    return " ".join(parts)


def load_simple_jsonl(pattern: str):

    reject_phrases_lower = []
    
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


def make_fixed_batch(examples, context_length, device, tokenizer, ignore_index=-666):
    pad_token_id = tokenizer.vocab_inv["<|pad|>".encode("utf-8")]
    x = torch.full((len(examples), context_length), pad_token_id, dtype=torch.long, device=device)
    y = torch.full((len(examples), context_length), ignore_index, dtype=torch.long, device=device)

    for b, example in enumerate(examples):
        prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)
        if len(full_tokens) > context_length:
            raise RuntimeError(f"example too long for context_length={context_length}: {example}")

        labels = [ignore_index] * len(full_tokens)
        labels[len(prompt_tokens):] = full_tokens[len(prompt_tokens):]

        x[b, :len(full_tokens)] = torch.tensor(full_tokens, dtype=torch.long, device=device)
        y[b, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)

    return x, y


@torch.no_grad()
def estimate_teacher_forced_accuracy(
    model,
    examples,
    batch_size,
    context_length,
    device,
    tokenizer,
    autocast,
    max_examples=128,
    ignore_index=-666,
):
    model.eval()
    eval_examples = examples[:max_examples]
    exact = 0
    first = 0
    total = 0
    by_task = {}

    for start in range(0, len(eval_examples), batch_size):
        chunk = eval_examples[start:start + batch_size]
        x, y = make_fixed_batch(chunk, context_length, device, tokenizer, ignore_index=ignore_index)
        with autocast:
            logits = model(x)

        pred = logits[:, :-1, :].argmax(dim=-1)
        target = y[:, 1:]
        mask = target != ignore_index

        for b in range(len(chunk)):
            positions = torch.nonzero(mask[b], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            total += 1
            first_pos = positions[0].item()
            first_ok = int(pred[b, first_pos].item() == target[b, first_pos].item())
            exact_ok = int(torch.equal(pred[b, positions], target[b, positions]))
            first += first_ok
            exact += exact_ok

            task = chunk[b].get("task", infer_task(chunk[b]["instruction"]))
            if task not in by_task:
                by_task[task] = {"exact": 0, "first": 0, "n": 0}
            by_task[task]["exact"] += exact_ok
            by_task[task]["first"] += first_ok
            by_task[task]["n"] += 1

    model.train()
    if total == 0:
        return {"tf_first": 0.0, "tf_exact": 0.0, "n": 0, "by_task": {}}
    task_summary = {
        task: {
            "tf_first": values["first"] / values["n"],
            "tf_exact": values["exact"] / values["n"],
            "n": values["n"],
        }
        for task, values in sorted(by_task.items())
    }
    return {
        "tf_first": first / total,
        "tf_exact": exact / total,
        "n": total,
        "by_task": task_summary,
    }


@torch.no_grad()
def print_greedy_samples(model, examples, tokenizer, context_length, max_token=16, max_examples=8):
    model.eval()
    end_token = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]
    for example in examples[:max_examples]:
        prompt = f"<|user|>\n{example['instruction']}\n<|assistant|>\n"
        output_ids = modules.generating(
            model=model,
            enc_user_prompt=tokenizer.encode(prompt),
            end_token=end_token,
            context_len=context_length,
            max_token=max_token,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        pred = tokenizer.decode(output_ids.tolist() if hasattr(output_ids, "tolist") else output_ids).strip()
        print(f"[sample] Q: {example['instruction']}")
        print(f"         gold={example['output']!r} pred={pred!r}")
    model.train()


def main():
    # ---------- 配置 ----------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diagnostic_clean_overfit = False
    train_total = 2048
    val_total = 512
    eval_exact_examples = 512
    context_length = 128
    batch_size = 16
    eval_interval = 100
    eval_iters = 10
    log_interval = 50
    checkpoint_interval = 500
    max_learning_rate = 1e-4
    min_learning_rate = 1e-5
    warmup_iters = 200
    max_iters = 3000
    start_iter = 1
    cosine_cycle_iters = max_iters - start_iter + 1
    max_grad_norm = 1.0
    data_pattern = "./data/synthetic_skills_v3.jsonl"
    final_ckpt = "./data/demo_clean_generalization_hira_lmhead_final.pt"

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    tokenizer, vocab_size = load_tokenizer()

    # 加载数据
    if diagnostic_clean_overfit:
        print("Building clean overfit examples...", flush=True)
        all_examples = build_clean_diagnostic_examples(128, seed=1337)
    else:
        print(f"Building clean generalization split: train={train_total}, val={val_total}...", flush=True)
        train_examples, val_examples = build_generalization_split(train_total, val_total, seed=1337)
        all_examples = train_examples
    if not all_examples:
        raise RuntimeError(f"No examples found in {data_pattern}")
    if diagnostic_clean_overfit:
        train_examples = all_examples
        val_examples = all_examples

    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")
    print(f"train task counts: {count_tasks(train_examples) if 'count_tasks' in globals() else 'n/a'}")
    print(f"val task counts: {count_tasks(val_examples) if 'count_tasks' in globals() else 'n/a'}")

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
        layer.attn.o_proj = HiRALinear(layer.attn.o_proj, r=32, alpha=32)
        layer.ffn.w1 = HiRALinear(layer.ffn.w1, r=32, alpha=32)
        layer.ffn.w2 = HiRALinear(layer.ffn.w2, r=32, alpha=32)
        layer.ffn.w3 = HiRALinear(layer.ffn.w3, r=32, alpha=32)

    # 冻结所有参数，然后选择性解冻
    for param in model.parameters():
        param.requires_grad = False

    trainable_embedding_token_ids = [
        tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")],
        tokenizer.vocab_inv["<|user|>".encode("utf-8")],
        tokenizer.vocab_inv["<|assistant|>".encode("utf-8")],
        tokenizer.vocab_inv["<|pad|>".encode("utf-8")],
    ]

    # 2. 只训练新增 chat special token 的 embedding；lm_head 全量放开用于诊断。
    model.token_embeddings.embedding_weights.requires_grad = True
    model.lm_head.W.requires_grad = True

    def embedding_grad_hook(grad):
        grad = grad.clone()
        keep = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
        keep[trainable_embedding_token_ids] = True
        grad[~keep] = 0
        return grad

    model.token_embeddings.embedding_weights.register_hook(embedding_grad_hook)

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

    start_iter = 1

    lr = max_learning_rate



    print("device:", device)
    print("diagnostic_clean_overfit:", diagnostic_clean_overfit)
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
            t0 = time.time()

        # 评估
        if it % eval_interval == 0:
            losses = estimate_loss()
            train_acc = estimate_teacher_forced_accuracy(
                model=model,
                examples=train_examples,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                tokenizer=tokenizer,
                autocast=autocast,
                max_examples=eval_exact_examples,
            )
            val_acc = estimate_teacher_forced_accuracy(
                model=model,
                examples=val_examples,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                tokenizer=tokenizer,
                autocast=autocast,
                max_examples=eval_exact_examples,
            )
            print(
                f"[eval] iter {it:8d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"train_tf {train_acc['tf_exact']:.3f} | "
                f"val_tf {val_acc['tf_exact']:.3f} | n {val_acc['n']}"
            )
            print(f"[eval/train tasks] {format_task_accuracy(train_acc)}")
            print(f"[eval/val tasks]   {format_task_accuracy(val_acc)}")
            if diagnostic_clean_overfit and it % 500 == 0:
                print_greedy_samples(model, train_examples, tokenizer, context_length)

        # 保存 checkpoint
        if it % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/simple_hira_iter_{it}.pt"
            modules.run_save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    # 最终保存
    torch.save({"model": model.state_dict(), "it": max_iters}, final_ckpt)
    print(f"Saved final model to {final_ckpt}")
    if diagnostic_clean_overfit:
        print_greedy_samples(model, train_examples, tokenizer, context_length)


if __name__ == "__main__":
    main()
