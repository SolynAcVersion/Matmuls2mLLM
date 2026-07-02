import os

import torch

import modules
from sft_chat_templetes import load_tokenizer


def fold_hira_weight(state_dict, prefix, transpose_output, scale):
    w0 = state_dict[f"{prefix}.W_0"]
    a = state_dict[f"{prefix}.A"]
    b = state_dict[f"{prefix}.B"]
    dense_weight = w0 + scale * (w0 * (a @ b))
    if transpose_output:
        return dense_weight.T.contiguous()
    return dense_weight.contiguous()


def main():
    input_path = "./data/shengoovlei_assistant_sft_stage3j_filtered_mix_final.pt"
    output_path = "./data/shengoovlei_assistant_sft_stage3j_filtered_mix_final_merged.pt"
    hira_r = 32
    hira_alpha = 32
    scale = hira_alpha / hira_r

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"checkpoint not found: {input_path}")

    tokenizer, vocab_size = load_tokenizer()
    dense_model = modules.TransformerLM(
        vocab_size=vocab_size,
        context_length=1024,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=2752,
        rope_theta=10000,
    )
    dense_state = dense_model.state_dict()

    src_obj = torch.load(input_path, map_location="cpu")
    src_state = src_obj["model"]

    for key in list(dense_state.keys()):
        if key in src_state:
            dense_state[key] = src_state[key].detach().clone()
            continue

        if key.endswith(".weight"):
            prefix = key[:-7]
            dense_state[key] = fold_hira_weight(
                state_dict=src_state,
                prefix=prefix,
                transpose_output=True,
                scale=scale,
            )
            continue

        if key.endswith(".W"):
            prefix = key[:-2]
            dense_state[key] = fold_hira_weight(
                state_dict=src_state,
                prefix=prefix,
                transpose_output=False,
                scale=scale,
            )
            continue

        raise KeyError(f"no source mapping for dense key: {key}")

    dense_model.load_state_dict(dense_state, strict=True)

    out_obj = {
        "checkpoint_format": "dense_transformerlm_from_stage3j_hira",
        "model": dense_state,
        "it": src_obj.get("it"),
        "source_checkpoint": input_path,
        "source_checkpoint_format": src_obj.get("checkpoint_format"),
        "hira_r": hira_r,
        "hira_alpha": hira_alpha,
    }
    torch.save(out_obj, output_path)

    print(f"source_checkpoint: {input_path}")
    print(f"merged_checkpoint: {output_path}")
    print(f"vocab_size: {vocab_size}")
    print(f"saved_keys: {len(dense_state)}")


if __name__ == "__main__":
    main()
