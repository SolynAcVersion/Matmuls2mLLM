import random

import einops
import torch.nn as nn
import torch
import torchvision.transforms as transforms

import modules

from PIL import Image

from transformers import CLIPVisionModel, CLIPImageProcessor

class VisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze=True):
        super().__init__()

        self.vision_model = CLIPVisionModel.from_pretrained(model_name)

        if freeze:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            self.vision_model.eval()


    def forward(self, pixel_values):
        # pixel_values: [B, 3, 224, 224]
        # output: [B, 197, 768]
        # patch_embeds: [B, 197, 768]
        # pooled_feature: [B, 768]
        outputs = self.vision_model(pixel_values)
        patch_embeds = outputs.last_hidden_state
        pooled_feature = patch_embeds[:, 0, :]
        return patch_embeds, pooled_feature
        

class VisionProjector(nn.Module):
    def __init__(self, vision_dim=768, text_dim=1024, hidden_dim=1536):
        super().__init__()
        self.l1 = nn.Linear(vision_dim, hidden_dim, bias=True)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, text_dim, bias=True)

    def forward(self, pooled_feature):
        pooled_feature = self.l1(pooled_feature)
        pooled_feature = self.gelu(pooled_feature)
        pooled_feature = self.l2(pooled_feature)

        visual_prefix = pooled_feature.unsqueeze(1)

        return visual_prefix

class MultiModalPrefixLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 context_length,
                 d_model,
                 num_layers,
                 num_heads,
                 d_ff,
                 rope_theta,
                 model_name,
                 freeze,
                 vision_dim=768,
                 text_dim=1024,
                 hidden_dim=1536,
                 eps: float = 1e-5,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ):
        super().__init__()
        self.encoder = VisionEncoder(model_name, freeze)
        self.projector = VisionProjector(vision_dim, text_dim, hidden_dim)
        self.transformer = modules.TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, eps, device, dtype)


    def forward(self, img_tensor, text_tokens):
        # img_tensor: [B, 3, 224, 224]
        # text_tokens: [B, T]
        # output: [B, K + T, V]

        patch_embeds, pooled_feature = self.encoder(img_tensor)
        # patch_embeds: [B, 197, 768]
        # pooled_feature: [B, 768]
        visual_prefix = self.projector(pooled_feature)
        # visual_prefix: [B, 1, 1024]
        text_embed = self.transformer.token_embeddings(text_tokens)

        total_embed = torch.cat([visual_prefix, text_embed], 1)

        B, S, _ = total_embed.shape

        token_positions = torch.arange(S, device = total_embed.device)
        token_positions = einops.repeat(token_positions, 'S -> B S', B=B)



        for layer in self.transformer.layers:
            total_embed = layer(total_embed, token_positions=token_positions)

        total_embed = self.transformer.ln_final(total_embed)
        output = self.transformer.lm_head(total_embed)

        return output


def make_multimodal_batch(
    samples: list,
    processor: CLIPImageProcessor,
    batch_size: int,
    tokenizer: modules.Tokenizer | modules.FastTokenizerOWTHighPerformance,
    max_text_len=1024,
    ignore_index=-666,
    max_attempt=2048
):
    """
    sample[0]:
    {'fn': './flickr30k-images/158898445.jpg',
     'desc': ['A young Asian man ...', ...],
     'prompt': 'describe the image:'
    }
    """

    pixel_values = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32)
    input_ids = torch.full((batch_size, max_text_len), 32002, dtype=torch.long)
    labels = torch.full((batch_size, max_text_len + 1), ignore_index, dtype=torch.long)
    # <|pad|> = [32002]
    start = random.randint(0, len(samples) - 1)
    idx = 0
    attempts = 1

    while idx < batch_size and attempts < max_attempt:

        attempts += 1

        s = samples[(idx + start) % len(samples)]

        try:
            img = Image.open(s['fn'])
            pixel_values[idx] = processor(images=img)['pixel_values'][0]
        except Exception as e:
            print(f"{s['fn']} 路径不存在！")
            start += 1
            continue

        s_prompt = tokenizer.encode(s['prompt'])
        s_desc = tokenizer.encode(s['desc'][0])
        s_prompt = torch.tensor(s_prompt, dtype=torch.long)
        s_desc = torch.tensor(s_desc, dtype=torch.long)
        full_text = torch.cat([s_prompt, s_desc])

        if len(full_text) > max_text_len:
            start += 1
            continue

        input_ids[idx, :len(full_text)] = full_text

        label = torch.cat([torch.full((1 + len(s_prompt), ), ignore_index, dtype=torch.long),
                           s_desc])

        labels[idx, :len(label)] = label

        idx += 1

    return pixel_values, input_ids, labels


def multimodal_ce_loss(logits, labels, ignore_index=-666):
    assert logits.shape[1] == labels.shape[1]
    return modules.run_cross_entropy_for_gem(logits, labels, ignore_index)


@torch.no_grad()
def generate_with_image(model, processor, img_path, text_tokens, eos_token=31999, max_tokens=256):
    model.eval()

    device = next(model.parameters()).device

    input_ids = torch.tensor([text_tokens], dtype=torch.long, device=device)
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    gen_start_idx = input_ids.shape[1]
    prefix_len = 1
    text_context_len = model.transformer.context_length - prefix_len
    if text_context_len <= 0:
        raise ValueError("transformer.context_length must be larger than the visual prefix length")

    for _ in range(max_tokens):
        input_window = input_ids[:, -text_context_len:]
        logits = model(pixel_values, input_window)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        if next_token.item() == eos_token:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids[0][gen_start_idx:]
