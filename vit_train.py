import datetime
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import wandb
from transformers import CLIPImageProcessor

import modules
import vit_modules
from sft_chat_templetes import HiRALinear, encode_chat_example, format_chat_prompt, load_tokenizer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(2007)
    np.random.seed(2007)
    torch.manual_seed(2007)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2007)

    with open("./data/vit_colors_shapes/train_data.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open("./data/vit_colors_shapes/val_data.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    tokenizer, vocab_size = load_tokenizer()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    train_samples = []
    val_samples = []
    for item in train_raw:
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        if not instruction or not output:
            continue
        _, full_tokens = encode_chat_example(
            {"instruction": instruction, "output": output},
            tokenizer,
        )
        if len(full_tokens) > 128:
            continue
        train_samples.append(
            {
                "fn": item["fn"],
                "instruction": instruction,
                "output": output,
                "task": str(item.get("task", "unknown")).strip() or "unknown",
            }
        )

    for item in val_raw:
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        if not instruction or not output:
            continue
        _, full_tokens = encode_chat_example(
            {"instruction": instruction, "output": output},
            tokenizer,
        )
        if len(full_tokens) > 128:
            continue
        val_samples.append(
            {
                "fn": item["fn"],
                "instruction": instruction,
                "output": output,
                "task": str(item.get("task", "unknown")).strip() or "unknown",
            }
        )

    if not train_samples:
        raise RuntimeError("no usable train samples found in ./data/vit_colors_shapes/train_data.json")
    if not val_samples:
        raise RuntimeError("no usable val samples found in ./data/vit_colors_shapes/val_data.json")

    if os.path.exists("./pswd.json"):
        with open("./pswd.json", encoding="utf-8") as f:
            payload = json.load(f)
        if str(payload.get("wandb-api-key", "")).strip():
            wandb.login(key=str(payload["wandb-api-key"]).strip())
    else:
        print("pswd.json not found, wandb will use the existing login state")

    run = wandb.init(
        project="gpt2vision",
        name="vit_projector_stage3j_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        mode="online",
        save_code=True,
        config={
            "train_data_path": "./data/vit_colors_shapes/train_data.json",
            "val_data_path": "./data/vit_colors_shapes/val_data.json",
            "init_ckpt_path": "./data/shengoovlei_assistant_sft_stage3j_filtered_mix_final.pt",
            "batch_size": 32,
            "max_iters": 5000,
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
            "max_text_len": 128,
            "max_generate_tokens": 16,
            "hira_r": 32,
            "hira_alpha": 32,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
        },
    )

    model = vit_modules.MultiModalPrefixLM(
        vocab_size=vocab_size,
        context_length=129,
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

    for layer in model.transformer.layers:
        layer.attn.q_proj = HiRALinear(layer.attn.q_proj, r=32, alpha=32)
        layer.attn.k_proj = HiRALinear(layer.attn.k_proj, r=32, alpha=32)
        layer.attn.v_proj = HiRALinear(layer.attn.v_proj, r=32, alpha=32)
        layer.attn.o_proj = HiRALinear(layer.attn.o_proj, r=32, alpha=32)
        layer.ffn.w1 = HiRALinear(layer.ffn.w1, r=32, alpha=32)
        layer.ffn.w2 = HiRALinear(layer.ffn.w2, r=32, alpha=32)
        layer.ffn.w3 = HiRALinear(layer.ffn.w3, r=32, alpha=32)

    model.transformer.load_state_dict(
        torch.load(
            "./data/shengoovlei_assistant_sft_stage3j_filtered_mix_final.pt",
            map_location="cpu",
        )["model"],
        strict=True,
    )
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

    for it in range(1, 5000 + 1):
        lr = modules.run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=100,
            cosine_cycle_iters=5000,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        pixel_values, input_ids, labels = vit_modules.make_multimodal_batch(
            samples=train_samples,
            processor=processor,
            batch_size=32,
            tokenizer=tokenizer,
            max_text_len=128,
            ignore_index=-666,
        )
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        tokens_since_log += input_ids.numel()

        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            ):
                logits = model(pixel_values, input_ids)
                loss = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=-666)
        else:
            logits = model(pixel_values, input_ids)
            loss = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=-666)

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
            losses = torch.zeros(20, device=device)
            for k in range(20):
                pixel_values, input_ids, labels = vit_modules.make_multimodal_batch(
                    samples=val_samples,
                    processor=processor,
                    batch_size=32,
                    tokenizer=tokenizer,
                    max_text_len=128,
                    ignore_index=-666,
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
                        losses[k] = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=-666)
                else:
                    logits = model(pixel_values, input_ids)
                    losses[k] = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=-666)
            val_loss = losses.mean().item()
            print(f"[eval] iter {it:6d} | val_loss {val_loss:.6f}")
            wandb.log({"eval/val_loss": val_loss}, step=it)
            model.train()
            model.encoder.eval()

        if it % 200 == 0:
            sample = random.choice(val_samples)
            pred_tokens = vit_modules.generate_with_image(
                model=model,
                processor=processor,
                img_path=sample["fn"],
                text_tokens=tokenizer.encode(format_chat_prompt(sample["instruction"])),
                eos_token=eos_token,
                max_tokens=16,
            )
            print(
                f"[sample] prompt={sample['instruction']!r} "
                f"gold={sample['output']!r} "
                f"pred={tokenizer.decode(pred_tokens.tolist()).split('<|endoftext|>', 1)[0].strip()!r}"
            )
            model.train()
            model.encoder.eval()

        if it % 500 == 0:
            model.eval()
            model.encoder.eval()
            metrics = {}
            sample_table = wandb.Table(columns=["task", "instruction", "gold", "pred"])
            for task, examples in sorted(val_eval_by_task.items()):
                correct = 0
                for example in examples:
                    pred_tokens = vit_modules.generate_with_image(
                        model=model,
                        processor=processor,
                        img_path=example["fn"],
                        text_tokens=tokenizer.encode(format_chat_prompt(example["instruction"])),
                        eos_token=eos_token,
                        max_tokens=16,
                    )
                    pred = tokenizer.decode(pred_tokens.tolist()).split("<|endoftext|>", 1)[0].strip()
                    gold = example["output"].strip()
                    correct += int(pred == gold)
                    if sample_table.data is None or len(sample_table.data) < 8:
                        sample_table.add_data(task, example["instruction"], gold, pred)
                metrics[task] = correct / len(examples)
            metrics["overall"] = sum(
                metrics[task] * len(examples) for task, examples in val_eval_by_task.items()
            ) / sum(len(examples) for examples in val_eval_by_task.values())
            print("[greedy/eval] " + " ".join(f"{task}:{score:.3f}" for task, score in sorted(metrics.items())))
            wandb.log(
                {**{f"greedy_eval/{task}": score for task, score in metrics.items()}, "greedy_eval/samples": sample_table},
                step=it,
            )
            model.train()
            model.encoder.eval()

        if it % 1000 == 0:
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save(
                {
                    "checkpoint_format": "vit_projector_stage3j",
                    "it": it,
                    "projector": model.projector.state_dict(),
                    "optim": optimizer.state_dict(),
                },
                f"./checkpoints/vit_projector_stage3j_iter_{it}.pt",
            )
            print(f"saved checkpoint to ./checkpoints/vit_projector_stage3j_iter_{it}.pt")

    model.eval()
    model.encoder.eval()
    losses = torch.zeros(20, device=device)
    for k in range(20):
        pixel_values, input_ids, labels = vit_modules.make_multimodal_batch(
            samples=val_samples,
            processor=processor,
            batch_size=32,
            tokenizer=tokenizer,
            max_text_len=128,
            ignore_index=-666,
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
                losses[k] = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=-666)
        else:
            logits = model(pixel_values, input_ids)
            losses[k] = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=-666)
    print(f"[final] val_loss {losses.mean().item():.6f}")
    wandb.log({"eval/final_val_loss": losses.mean().item()}, step=5000)

    metrics = {}
    sample_table = wandb.Table(columns=["task", "instruction", "gold", "pred"])
    for task, examples in sorted(val_eval_by_task.items()):
        correct = 0
        for example in examples:
            pred_tokens = vit_modules.generate_with_image(
                model=model,
                processor=processor,
                img_path=example["fn"],
                text_tokens=tokenizer.encode(format_chat_prompt(example["instruction"])),
                eos_token=eos_token,
                max_tokens=16,
            )
            pred = tokenizer.decode(pred_tokens.tolist()).split("<|endoftext|>", 1)[0].strip()
            gold = example["output"].strip()
            correct += int(pred == gold)
            if sample_table.data is None or len(sample_table.data) < 8:
                sample_table.add_data(task, example["instruction"], gold, pred)
        metrics[task] = correct / len(examples)
    metrics["overall"] = sum(
        metrics[task] * len(examples) for task, examples in val_eval_by_task.items()
    ) / sum(len(examples) for examples in val_eval_by_task.values())
    print("[final greedy/eval] " + " ".join(f"{task}:{score:.3f}" for task, score in sorted(metrics.items())))
    wandb.log(
        {**{f"greedy_eval_final/{task}": score for task, score in metrics.items()}, "greedy_eval_final/samples": sample_table},
        step=5000,
    )

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(
        {
            "checkpoint_format": "vit_projector_stage3j",
            "it": 5000,
            "projector": model.projector.state_dict(),
            "optim": optimizer.state_dict(),
        },
        "./checkpoints/vit_projector_stage3j_final.pt",
    )
    print("saved checkpoint to ./checkpoints/vit_projector_stage3j_final.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
