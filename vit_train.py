import json
import random
import os
import numpy as np
import torch
from transformers import CLIPImageProcessor

import vit_modules
from sft_chat_templetes import PRETRAIN_CKPT_PATH, load_tokenizer


def load_shape_samples(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        samples.append(
            {
                "fn": item["fn"],
                "desc": item["desc"],
                "prompt": item["prompt"],
            }
        )
    return samples


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = 16
    max_iters = 400
    log_interval = 20
    sample_interval = 100
    learning_rate = 1e-3
    max_text_len = 64
    ignore_index = -666

    tokenizer, vocab_size = load_tokenizer()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    samples = load_shape_samples("./data/vit_colors_shapes/smoke_test_data.json")

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

    ckpt = torch.load(PRETRAIN_CKPT_PATH, map_location="cpu")
    model.transformer.load_state_dict(ckpt["model"])

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=learning_rate)
    eos_token = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]

    print(f"device: {device}")
    print(f"samples: {len(samples)}")
    print(f"vocab_size: {vocab_size}")
    print(f"pretrain_ckpt: {PRETRAIN_CKPT_PATH}")

    model.train()
    for it in range(1, max_iters + 1):
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

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values, input_ids)
        loss = vit_modules.multimodal_ce_loss(logits, labels, ignore_index=ignore_index)
        loss.backward()
        optimizer.step()

        if it % log_interval == 0 or it == 1:
            print(f"iter {it:04d} | loss {loss.item():.6f}")

        if it % sample_interval == 0:
            sample = random.choice(samples)
            prompt_tokens = tokenizer.encode(sample["prompt"])
            gen_tokens = vit_modules.generate_with_image(
                model=model,
                processor=processor,
                img_path=sample["fn"],
                text_tokens=prompt_tokens,
                eos_token=eos_token,
                max_tokens=8,
            )
            gen_text = tokenizer.decode(gen_tokens.tolist())
            print(f"sample prompt: {sample['prompt']}")
            print(f"sample target: {sample['desc'][0]}")
            print(f"sample pred: {gen_text}")


if __name__ == "__main__":
    main()
