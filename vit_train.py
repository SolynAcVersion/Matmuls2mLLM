import datetime
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import wandb
from transformers import CLIPImageProcessor

import modules
import vit_modules
from sft_chat_templetes import encode_chat_example, format_chat_prompt, load_tokenizer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


VALID_TASKS = {"scene", "left_object", "right_object"}
COLOR_WORDS = ["red", "yellow", "blue", "green", "white", "black"]
SHAPE_WORDS = ["rectangle", "ellipse", "triangle"]


def load_filtered_samples(items, tokenizer, max_text_len):
    samples = []
    for item in items:
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        task = str(item.get("task", "unknown")).strip() or "unknown"
        if not instruction or not output or task not in VALID_TASKS:
            continue
        _, full_tokens = encode_chat_example({"instruction": instruction, "output": output}, tokenizer)
        if len(full_tokens) <= max_text_len:
            samples.append({"fn": item["fn"], "instruction": instruction, "output": output, "task": task})
    return samples


def load_flickr30k_samples(csv_path, image_dir, tokenizer, max_text_len):
    samples = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = str(row.get("filename", "")).strip()
            split = str(row.get("split", "")).strip()
            if not filename or split not in {"train", "val", "test"}:
                continue
            raw_captions = json.loads(row["raw"])
            for caption_index, caption in enumerate(raw_captions):
                instruction = random.choice(
                    [
                        "describe the image",
                        "what does the image show?",
                        "what do you see?",
                        "describe this picture",
                        "caption this image",
                    ]
                )
                output = str(caption).strip()
                if not output:
                    continue
                _, full_tokens = encode_chat_example({"instruction": instruction, "output": output}, tokenizer)
                if len(full_tokens) > max_text_len:
                    continue
                samples.append(
                    {
                        "fn": os.path.join(image_dir, filename),
                        "instruction": instruction,
                        "output": output,
                        "task": "caption",
                        "split": split,
                        "img_id": str(row.get("img_id", "")).strip(),
                        "filename": filename,
                        "caption_index": caption_index,
                    }
                )
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    val_samples = [sample for sample in samples if sample["split"] in {"val", "test"}]
    return train_samples, val_samples


def extract_single_label(text, candidates):
    normalized = " " + re.sub(r"[^a-z]+", " ", text.lower()) + " "
    for candidate in candidates:
        if f" {candidate} " in normalized:
            return candidate
    return None


def extract_side_objects(text):
    normalized = " " + re.sub(r"[^a-z]+", " ", text.lower()) + " "
    objects = []
    for side in ["left", "right"]:
        side_match = re.search(rf"\bon\s+the\s+{side}\b", normalized)
        if side_match is None:
            continue
        window_start = max(0, side_match.start() - 48)
        window_text = normalized[window_start : side_match.end()]
        color = None
        shape = None
        for candidate in COLOR_WORDS:
            if f" {candidate} " in window_text:
                color = candidate
                break
        for candidate in SHAPE_WORDS:
            if f" {candidate} " in window_text:
                shape = candidate
                break
        if color is not None and shape is not None:
            item = (side, color, shape)
            if item not in objects:
                objects.append(item)
    return objects


def score_prediction(task, gold, pred):
    if task == "scene":
        gold_objects = sorted(extract_side_objects(gold))
        pred_objects = sorted(extract_side_objects(pred))
        return float(bool(gold_objects) and pred_objects == gold_objects)
    side = "left" if task == "left_object" else "right"
    gold_objects = [item for item in extract_side_objects(gold) if item[0] == side]
    pred_objects = [item for item in extract_side_objects(pred) if item[0] == side]
    return float(bool(gold_objects) and bool(pred_objects) and pred_objects[0] == gold_objects[0])


def generate_prediction(model, processor, tokenizer, eos_token, sample, max_generate_tokens):
    pred_tokens = vit_modules.generate_with_image(
        model=model,
        processor=processor,
        img_path=sample["fn"],
        text_tokens=tokenizer.encode(format_chat_prompt(sample["instruction"])),
        eos_token=eos_token,
        max_tokens=max_generate_tokens,
    )
    return tokenizer.decode(pred_tokens.tolist()).split("<|endoftext|>", 1)[0].strip()


def compute_batch_loss(model, processor, samples, batch_size, tokenizer, max_text_len, ignore_index, device):
    pixel_values, input_ids, labels = vit_modules.make_multimodal_batch(
        samples=samples,
        processor=processor,
        batch_size=batch_size,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        ignore_index=ignore_index,
    )
    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    if device.type == "cuda":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ):
            logits = model(pixel_values, input_ids)
            loss = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=ignore_index)
    else:
        logits = model(pixel_values, input_ids)
        loss = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=ignore_index)
    return loss, input_ids.numel()


def evaluate_val_loss(model, processor, samples, batch_size, tokenizer, max_text_len, ignore_index, device, eval_iters):
    losses = torch.zeros(eval_iters, device=device)
    for idx in range(eval_iters):
        losses[idx], _ = compute_batch_loss(
            model=model,
            processor=processor,
            samples=samples,
            batch_size=batch_size,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            ignore_index=ignore_index,
            device=device,
        )
    return losses.mean().item()


def evaluate_greedy(model, processor, tokenizer, eos_token, eval_by_task, max_generate_tokens):
    metrics = {}
    sample_table = wandb.Table(columns=["task", "instruction", "gold", "pred", "gold_parse", "pred_parse"])
    for task, examples in sorted(eval_by_task.items()):
        correct = 0.0
        for example in examples:
            pred = generate_prediction(model, processor, tokenizer, eos_token, example, max_generate_tokens)
            gold = example["output"].strip()
            correct += score_prediction(task, gold, pred)
            if sample_table.data is None or len(sample_table.data) < 8:
                sample_table.add_data(
                    task,
                    example["instruction"],
                    gold,
                    pred,
                    f"color={extract_single_label(gold, COLOR_WORDS)}, shape={extract_single_label(gold, SHAPE_WORDS)}, objects={extract_side_objects(gold)}",
                    f"color={extract_single_label(pred, COLOR_WORDS)}, shape={extract_single_label(pred, SHAPE_WORDS)}, objects={extract_side_objects(pred)}",
                )
        metrics[task] = correct / len(examples)
    metrics["overall"] = sum(metrics[task] * len(examples) for task, examples in eval_by_task.items()) / sum(
        len(examples) for examples in eval_by_task.values()
    )
    return metrics, sample_table


def evaluate_greedy_flickr(model, processor, tokenizer, eos_token, samples, max_generate_tokens):
    sample_table = wandb.Table(columns=["instruction", "gold", "pred", "filename"])
    for sample in samples[:16]:
        pred = generate_prediction(model, processor, tokenizer, eos_token, sample, max_generate_tokens)
        sample_table.add_data(sample["instruction"], sample["output"], pred, sample["filename"])
    return {"preview_count": float(min(len(samples), 16))}, sample_table


def save_projector_checkpoint(model, optimizer, it, path):
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(
        {
            "checkpoint_format": "vit_projector_stage3j",
            "it": it,
            "projector": model.projector.state_dict(),
            "optim": optimizer.state_dict(),
        },
        path,
    )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_version: {torch.version.cuda}")
    print(f"device: {device}")
    if device.type != "cuda" and os.environ.get("ALLOW_CPU", "0") != "1":
        raise RuntimeError(
            "CUDA is unavailable, so vit_train.py would run on CPU only. "
            "The warning above indicates that the installed PyTorch build requires a newer NVIDIA driver "
            "than the server currently provides. Update the driver or reinstall a PyTorch build compiled "
            "for an older CUDA runtime. If you intentionally want CPU training, rerun with ALLOW_CPU=1."
        )

    batch_size = 4
    eval_batch_size = 4
    max_text_len = 128
    max_iters = 12000
    dataset_name = "flickr30k"
    init_ckpt_path = "./data/shengoovlei_assistant_sft_stage3j_filtered_mix_final_merged.pt"
    init_projector_ckpt_path = "./data/vit_projector_stage3_stage3j_final.pt"

    random.seed(2007)
    np.random.seed(2007)
    torch.manual_seed(2007)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2007)

    tokenizer, vocab_size = load_tokenizer()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    if dataset_name == "flickr30k":
        train_samples, val_samples = load_flickr30k_samples(
            csv_path="./data/flickr30k/flickr_annotations_30k.csv",
            image_dir="./data/flickr30k/flickr30k-images",
            tokenizer=tokenizer,
            max_text_len=max_text_len,
        )
    else:
        with open("./data/vit_colors_shapes/train_data.json", encoding="utf-8") as f:
            train_raw = json.load(f)
        with open("./data/vit_colors_shapes/val_data.json", encoding="utf-8") as f:
            val_raw = json.load(f)
        train_samples = load_filtered_samples(train_raw, tokenizer, max_text_len)
        val_samples = load_filtered_samples(val_raw, tokenizer, max_text_len)

    if not train_samples:
        raise RuntimeError("no usable train samples found in ./data/vit_colors_shapes/train_data.json")
    if not val_samples:
        raise RuntimeError("no usable val samples found in ./data/vit_colors_shapes/val_data.json")
    if not os.path.exists(init_ckpt_path):
        raise FileNotFoundError(
            f"merged checkpoint not found: {init_ckpt_path}. "
            "Run `python ./fold_stage3j_hira_to_dense.py` first."
        )

    if os.path.exists("./pswd.json"):
        with open("./pswd.json", encoding="utf-8") as f:
            payload = json.load(f)
        if str(payload.get("wandb-api-key", "")).strip():
            wandb.login(key=str(payload["wandb-api-key"]).strip())
    else:
        print("pswd.json not found, wandb will use the existing login state")

    wandb.init(
        project="gpt2vision",
        name="vit_projector_stage3j_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        mode="online",
        save_code=True,
        config={
            "dataset_name": dataset_name,
            "train_data_path": "./data/flickr30k/flickr_annotations_30k.csv" if dataset_name == "flickr30k" else "./data/vit_colors_shapes/train_data.json",
            "val_data_path": "./data/flickr30k/flickr_annotations_30k.csv" if dataset_name == "flickr30k" else "./data/vit_colors_shapes/val_data.json",
            "init_ckpt_path": init_ckpt_path,
            "init_projector_ckpt_path": init_projector_ckpt_path,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "log_interval": 20,
            "eval_interval": 200,
            "eval_iters": 20,
            "greedy_eval_interval": 500,
            "greedy_examples_per_task": 32,
            "sample_interval": 200,
            "checkpoint_interval": 1000,
            "max_learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "warmup_iters": 100,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "max_text_len": max_text_len,
            "max_generate_tokens": 64,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
        },
    )

    model = vit_modules.MultiModalPrefixLM(
        vocab_size=vocab_size,
        context_length=max_text_len + 1,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=2752,
        rope_theta=10000,
        model_name="openai/clip-vit-base-patch16",
        freeze=True,
        vision_dim=768,
        text_dim=1024,
        hidden_dim=1536,
        device=device,
    ).to(device)

    model.transformer.load_state_dict(
        torch.load(
            init_ckpt_path,
            map_location="cpu",
        )["model"],
        strict=True,
    )
    if os.path.exists(init_projector_ckpt_path):
        projector_obj = torch.load(init_projector_ckpt_path, map_location="cpu")
        model.projector.load_state_dict(projector_obj["projector"], strict=True)
        print(f"continued_projector_from: {init_projector_ckpt_path}")
    else:
        print(f"projector checkpoint not found, training projector from random init: {init_projector_ckpt_path}")
    model.to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.projector.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    eos_token = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]

    val_eval_by_task = defaultdict(list)
    if dataset_name == "flickr30k":
        seen_val_images = set()
        for sample in val_samples:
            key = sample["filename"]
            if key in seen_val_images:
                continue
            seen_val_images.add(key)
            val_eval_by_task["caption"].append(sample)
            if len(val_eval_by_task["caption"]) >= 32:
                break
    else:
        for sample in val_samples:
            if len(val_eval_by_task[sample["task"]]) < 32:
                val_eval_by_task[sample["task"]].append(sample)

    print(f"train_size: {len(train_samples)}")
    print(f"val_size: {len(val_samples)}")
    print(f"train_task_mix: {dict(sorted(Counter(sample['task'] for sample in train_samples).items()))}")
    print(f"val_task_mix: {dict(sorted(Counter(sample['task'] for sample in val_samples).items()))}")
    print(
        "params: total="
        + f"{sum(param.numel() for param in model.parameters()):,}"
        + " trainable="
        + f"{sum(param.numel() for param in model.parameters() if param.requires_grad):,}"
    )

    model.train()
    model.encoder.eval()
    tokens_since_log = 0
    t0 = time.time()

    for it in range(1, max_iters + 1):
        lr = modules.run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=100,
            cosine_cycle_iters=max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss, token_count = compute_batch_loss(
            model=model,
            processor=processor,
            samples=train_samples,
            batch_size=batch_size,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            ignore_index=-666,
            device=device,
        )
        tokens_since_log += token_count

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        for param in model.projector.parameters():
            if param.grad is not None:
                grad_norm += param.grad.detach().float().norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        modules.run_gradient_clipping(model.projector.parameters(), max_l2_norm=1.0)
        optimizer.step()

        if it % 20 == 0 or it == 1:
            dt = max(time.time() - t0, 1e-6)
            tok_s = tokens_since_log / dt
            projector_param_norm = 0.0
            for param in model.projector.parameters():
                projector_param_norm += param.detach().float().norm().item() ** 2
            projector_param_norm = projector_param_norm ** 0.5
            print(
                f"iter {it:6d} | loss {loss.item():.6f} | lr {lr:.2e} | "
                f"tok/s {tok_s:.0f} | grad_norm {grad_norm:.4f} | projector_norm {projector_param_norm:.4f}"
            )
            wandb.log(
                {
                    "train/iter": it,
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tok_s": tok_s,
                    "train/grad_norm": grad_norm,
                    "train/projector_param_norm": projector_param_norm,
                },
                step=it,
            )
            tokens_since_log = 0
            t0 = time.time()

        if it % 200 == 0:
            model.eval()
            model.encoder.eval()
            val_loss = evaluate_val_loss(model, processor, val_samples, eval_batch_size, tokenizer, max_text_len, -666, device, 20)
            print(f"[eval] iter {it:6d} | val_loss {val_loss:.6f}")
            wandb.log({"eval/val_loss": val_loss}, step=it)
            model.train()
            model.encoder.eval()

        if it % 200 == 0:
            sample = random.choice(val_samples)
            pred = generate_prediction(model, processor, tokenizer, eos_token, sample, 48)
            print(
                f"[sample] prompt={sample['instruction']!r} "
                f"gold={sample['output']!r} "
                f"pred={pred!r}"
            )
            model.train()
            model.encoder.eval()

        if it % 500 == 0:
            model.eval()
            model.encoder.eval()
            if dataset_name == "flickr30k":
                metrics, sample_table = evaluate_greedy_flickr(model, processor, tokenizer, eos_token, val_eval_by_task["caption"], 64)
            else:
                metrics, sample_table = evaluate_greedy(model, processor, tokenizer, eos_token, val_eval_by_task, 48)
            print("[greedy/eval] " + " ".join(f"{task}:{score:.3f}" for task, score in sorted(metrics.items())))
            wandb.log(
                {**{f"greedy_eval/{task}": score for task, score in metrics.items()}, "greedy_eval/samples": sample_table},
                step=it,
            )
            model.train()
            model.encoder.eval()

        if it % 1000 == 0:
            save_projector_checkpoint(model, optimizer, it, f"./checkpoints/vit_projector_stage3j_iter_{it}.pt")
            print(f"saved checkpoint to ./checkpoints/vit_projector_stage3j_iter_{it}.pt")

    model.eval()
    model.encoder.eval()
    final_val_loss = evaluate_val_loss(model, processor, val_samples, eval_batch_size, tokenizer, max_text_len, -666, device, 20)
    print(f"[final] val_loss {final_val_loss:.6f}")
    wandb.log({"eval/final_val_loss": final_val_loss}, step=max_iters)

    if dataset_name == "flickr30k":
        metrics, sample_table = evaluate_greedy_flickr(model, processor, tokenizer, eos_token, val_eval_by_task["caption"], 64)
    else:
        metrics, sample_table = evaluate_greedy(model, processor, tokenizer, eos_token, val_eval_by_task, 48)
    print("[final greedy/eval] " + " ".join(f"{task}:{score:.3f}" for task, score in sorted(metrics.items())))
    wandb.log(
        {**{f"greedy_eval_final/{task}": score for task, score in metrics.items()}, "greedy_eval_final/samples": sample_table},
        step=max_iters,
    )

    save_projector_checkpoint(model, optimizer, max_iters, "./checkpoints/vit_projector_stage3j_final.pt")
    print("saved checkpoint to ./checkpoints/vit_projector_stage3j_final.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
