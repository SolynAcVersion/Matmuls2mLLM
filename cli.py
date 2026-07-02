import os

import torch
from transformers import CLIPImageProcessor

import modules
import vit_modules
from sft_chat_templetes import format_chat_prompt, load_tokenizer


def load_projector_state(model, checkpoint_path):
    obj = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(obj, dict) and "projector" in obj and isinstance(obj["projector"], dict):
        model.projector.load_state_dict(obj["projector"], strict=True)
        return "projector"

    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
        projector_keys = {
            key[len("projector."):]: value
            for key, value in state_dict.items()
            if key.startswith("projector.")
        }
        if projector_keys:
            model.projector.load_state_dict(projector_keys, strict=True)
            return "model.projector"

    if isinstance(obj, dict) and {"l1.weight", "l1.bias", "l2.weight", "l2.bias"} <= set(obj.keys()):
        model.projector.load_state_dict(obj, strict=True)
        return "raw_projector_state_dict"

    raise ValueError(
        f"unsupported checkpoint format: {checkpoint_path}. "
        "Expected a vit projector checkpoint with a `projector` key, "
        "a full model checkpoint with `projector.*` keys, or a raw projector state_dict."
    )


def build_model(device):
    tokenizer, vocab_size = load_tokenizer()
    model = vit_modules.MultiModalPrefixLM(
        vocab_size=vocab_size,
        context_length=65,
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

    transformer_ckpt_path = "./data/shengoovlei_assistant_sft_stage3j_filtered_mix_final_merged.pt"
    transformer_obj = torch.load(transformer_ckpt_path, map_location="cpu")
    model.transformer.load_state_dict(transformer_obj["model"], strict=True)
    model.to(device)
    model.eval()
    model.encoder.eval()
    return model, tokenizer, transformer_ckpt_path


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model, tokenizer, transformer_ckpt_path = build_model(device)
    eos_token = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]

    default_projector_ckpt = "./checkpoints/vit_projector_stage3j_final.pt"
    print(f"device: {device}")
    print(f"transformer_ckpt: {transformer_ckpt_path}")
    projector_ckpt_path = input(f"vit checkpoint path [{default_projector_ckpt}]: ").strip()
    if not projector_ckpt_path:
        projector_ckpt_path = default_projector_ckpt
    projector_ckpt_path = projector_ckpt_path.strip().strip('"').strip("'")

    if not os.path.exists(projector_ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {projector_ckpt_path}")

    loaded_from = load_projector_state(model, projector_ckpt_path)
    print(f"loaded_projector_from: {projector_ckpt_path} ({loaded_from})")
    print("max_tokens: 64")
    print("type `exit` or `quit` at the image path prompt to stop")

    while True:
        image_path = input("image path (blank for text-only): ").strip()
        if image_path.lower() in {"exit", "quit"}:
            break
        image_path = image_path.strip('"').strip("'")

        text = input("text: ").strip()
        if not text:
            print("empty text, skipped")
            continue

        prompt_tokens = tokenizer.encode(format_chat_prompt(text))

        with torch.no_grad():
            if image_path:
                if not os.path.exists(image_path):
                    print(f"image not found: {image_path}")
                    continue
                output_ids = vit_modules.generate_with_image(
                    model=model,
                    processor=processor,
                    img_path=image_path,
                    text_tokens=prompt_tokens,
                    eos_token=eos_token,
                    max_tokens=64,
                )
            else:
                output_ids = modules.generating(
                    model=model.transformer,
                    enc_user_prompt=prompt_tokens,
                    end_token=eos_token,
                    context_len=model.transformer.context_length,
                    max_token=64,
                    do_sample=False,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                )

        output_text = tokenizer.decode(
            output_ids.tolist() if hasattr(output_ids, "tolist") else output_ids
        ).split("<|endoftext|>", 1)[0].strip()
        print(f"output: {output_text}")


if __name__ == "__main__":
    main()
