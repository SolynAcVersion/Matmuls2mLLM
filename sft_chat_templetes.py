import contextlib
import os
import glob
import random
import time
import numpy as np
import torch
import datetime
import wandb

import math

import modules
import torch.nn as nn
import torch.nn.functional as F

import json

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_VOCAB_SIZE = 32000
VOCAB_PATH = './data/owt_train_32004.pickle'
MERGES_PATH = './data/owt_train_32000_merges.pickle'
PRETRAIN_CKPT_PATH = os.environ.get(
    "PRETRAIN_CKPT_PATH",
    './data/pretrain_gpt2med_iter_390000_chatvocab32003.pt',
)
SFT_RESUME_CKPT_PATH = os.environ.get(
    "SFT_RESUME_CKPT_PATH",
    './checkpoints/RE_sft_EvolSft_HiRA_r_16_gpt2med_iter_10000.pt',
)
SFT_RESUME_OPTIM = os.environ.get("SFT_RESUME_OPTIM", "0") == "1"

CHAT_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad|>",
]


def get_hira_update_ratio(model):
    ratios = []
    for module in model.modules():
        if isinstance(module, HiRALinear):
            delta = module.scale * (module.W_0 * (module.A @ module.B))
            ratio = delta.float().norm() / module.W_0.float().norm()
            ratios.append(ratio.item())
    return sum(ratios) / len(ratios)

def get_hira_ab_norm(model):
    A_norm = 0
    B_norm = 0
    cnt = 0
    for m in model.modules():
        if isinstance(m, HiRALinear):
            A_norm += m.A.norm().item()
            B_norm += m.B.norm().item()
            cnt += 1
    return (A_norm / cnt, B_norm / cnt)

def load_tokenizer():
    base_vocab = modules.load_with_pickle(VOCAB_PATH)
    vocab = {i: base_vocab[i] for i in range(BASE_VOCAB_SIZE)}
    token_to_id = {token: token_id for token_id, token in vocab.items()}
    next_id = BASE_VOCAB_SIZE
    for token in CHAT_SPECIAL_TOKENS:
        token = token.encode("utf-8")
        if token not in token_to_id:
            token_to_id[token] = next_id
            vocab[next_id] = token
            next_id += 1

    merges = modules.load_with_pickle(MERGES_PATH)
    tokenizer = modules.FastTokenizerOWTHighPerformance(
        vocab,
        merges,
        CHAT_SPECIAL_TOKENS,
    )
    return tokenizer, len(vocab)

def format_chat_prompt(instruction):
    instruction = str(instruction).strip()
    return f"<|user|>\n{instruction}\n<|assistant|>\n"

def format_chat_example(example):
    return format_chat_prompt(example["instruction"]) + str(example["output"]).strip() + "<|endoftext|>"

def encode_chat_example(example, tokenizer):
    prompt_tokens = tokenizer.encode(format_chat_prompt(example["instruction"]))
    output_tokens = tokenizer.encode(str(example["output"]).strip())
    eot_token_id = tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")]
    return prompt_tokens, prompt_tokens + output_tokens + [eot_token_id]

def get_linear_weight_and_bias(base):
    if hasattr(base, "W"):
        weight = base.W
    elif hasattr(base, "weight"):
        weight = base.weight.T
    else:
        raise TypeError(f"unsupported linear module: {type(base)!r}")

    bias = getattr(base, "bias", None)
    return weight, bias

def get_batch_from_json(
    json_data,
    batch_size,
    context_length,
    device,
    tokenizer,
    for_valid=False,
    ignore_index=-666,
):
    pad_token_id = tokenizer.vocab_inv['<|pad|>'.encode("utf-8")]

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

    valid_end = math.floor(len(json_data) * 0.1)

    if for_valid:
        data = json_data[: valid_end]
    else:
        data = json_data[valid_end: ]

    for b in range(batch_size):
        attempts = 0
        while True:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError(
                    f"failed to sample an SFT example within context_length={context_length}"
                )
            example = random.choice(data)
            x_token_part, x_token_full = encode_chat_example(example, tokenizer)
            if len(x_token_full) <= context_length:
                break


        labels = [ignore_index] * len(x_token_full)

        for i in range(len(x_token_part), len(x_token_full)):
            labels[i] = x_token_full[i]

        x[b, : len(x_token_full)] = torch.tensor(
            x_token_full,
            dtype=torch.long,
            device=device,
        )

        y[b, : len(labels)] = torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )

    return x, y


@torch.no_grad()
def estimate_first_token_rank(
    model,
    json_data,
    tokenizer,
    context_length,
    n=50,
    for_valid=True,
):
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    valid_end = math.floor(len(json_data) * 0.1)
    data = json_data[: valid_end] if for_valid else json_data[valid_end:]

    ranks = []

    for example in data:
        prompt_tokens, full_tokens = encode_chat_example(example, tokenizer)

        if len(full_tokens) > context_length:
            continue
        if len(prompt_tokens) >= len(full_tokens):
            continue

        x = torch.tensor(
            [prompt_tokens],
            dtype=torch.long,
            device=device,
        )
        logits = model(x)[0, -1].float()
        target = full_tokens[len(prompt_tokens)]
        rank = int((logits > logits[target]).sum().item()) + 1
        ranks.append(rank)

        if len(ranks) >= n:
            break

    if was_training:
        model.train()

    if len(ranks) == 0:
        return None

    return {
        "rank": sum(ranks) / len(ranks),
        "top1": sum(rank == 1 for rank in ranks) / len(ranks),
        "n": len(ranks),
    }



class HiRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=8):
        super().__init__()
        weight, bias = get_linear_weight_and_bias(base)
        self.W_0 = nn.Parameter(weight.detach().clone(), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None


        self.d_in, self.d_out = self.W_0.shape
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.A = nn.Parameter(torch.randn(self.d_in, r) * 0.001, requires_grad=True)
        self.B = nn.Parameter(torch.zeros(r, self.d_out), requires_grad=True)

    def forward(self, x):
        delta = self.W_0 * (self.A @ self.B)
        ret = x @ self.W_0 + self.scale * (x @ delta)
        if self.bias is not None:
            ret = ret + self.bias
        return ret


def sft():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context_length = 1024
    batch_size = 8
    eval_interval = 250
    eval_iters = 30
    log_interval = 50
    checkpoint_interval = 500
    max_learning_rate = 2e-5
    min_learning_rate = 5e-6
    warmup_iters = 200
    max_iters = 20100
    start_iter = 1
    cosine_cycle_iters = max_iters - start_iter + 1
    max_grad_norm = 0.5
    data_path = "./data/alpaca_evol_instruct_70k.json"
    final_path = "./data/evol_sft_hira_from_pretrain_final.pt"

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    tokenizer, vocab_size = load_tokenizer()
    trainable_token_ids = [
        tokenizer.vocab_inv["<|endoftext|>".encode("utf-8")],
        tokenizer.vocab_inv["<|user|>".encode("utf-8")],
        tokenizer.vocab_inv["<|assistant|>".encode("utf-8")],
        tokenizer.vocab_inv["<|pad|>".encode("utf-8")],
    ]

    with open(data_path, encoding="utf-8") as file:
        examples = [
            example
            for example in json.load(file)
            if str(example.get("instruction", "")).strip()
            and str(example.get("output", "")).strip()
        ]
    if not examples:
        raise RuntimeError(f"no valid Evol-SFT examples loaded from {data_path}")
    random.Random(1337).shuffle(examples)
    valid_end = max(1, len(examples) // 10)
    train_examples = examples[valid_end:]
    val_examples = examples[:valid_end]

    def make_batch(examples):
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
            -666,
            dtype=torch.long,
            device=device,
        )

        for b in range(batch_size):
            for _ in range(10000):
                prompt_tokens, full_tokens = encode_chat_example(
                    random.choice(examples),
                    tokenizer,
                )
                if len(full_tokens) <= context_length:
                    break
            else:
                raise RuntimeError(
                    f"failed to sample an Evol-SFT example within context_length={context_length}"
                )

            labels = [-666] * len(full_tokens)
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
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True
    resume_obj = torch.load('./checkpoints/evol_sft_hira_from_pretrain_iter_13000.pt', map_location="cpu")
    model.load_state_dict(resume_obj["model"])
    resume_iter = resume_obj["it"]
    start_iter = resume_iter + 1

    model.to(device)
    model.train()

    optimizer = modules.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    optimizer.load_state_dict(resume_obj["optim"])

    with open("./pswd.json", encoding="utf-8") as file:
        os.environ["WANDB_API_KEY"] = json.load(file)["wandb-api-key"]
    wandb.init(
        project="RE_gpt2vision_sft_chat_templetes",
        config={
            "pretrain_ckpt_path": PRETRAIN_CKPT_PATH,
            "target_iter": max_iters,
            "batch_size": batch_size,
            "lr": max_learning_rate,
        },
        name="[evol-gem] HiRA-r16 from pretrain "
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        save_code=False,
    )

    print("device:", device)
    print("pretrain_ckpt_path:", PRETRAIN_CKPT_PATH)
    print("start_iter:", start_iter, "target_iter:", max_iters)
    print("evol train:", len(train_examples), "val:", len(val_examples))
    print("trainable_token_ids:", trainable_token_ids)
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
        for split, split_examples in (("train", train_examples), ("val", val_examples)):
            losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                x, y = make_batch(split_examples)
                with autocast:
                    losses[k] = modules.run_cross_entropy_for_gem(
                        model(x),
                        y,
                        ignore_index=-666,
                    )
            out[split] = losses.mean().item()
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

        x, y = make_batch(train_examples)
        with autocast:
            loss = modules.run_cross_entropy_for_gem(
                model(x),
                y,
                ignore_index=-666,
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
            wandb.log(
                {
                    "loss": loss.item(),
                    "adapter/A_norm": a_norm,
                    "adapter/B_norm": b_norm,
                    "adapter/update_ratio": update_ratio,
                },
                step=it,
            )
            t0 = time.time()

        if it % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"[eval] iter {it:8d} | "
                f"evol train {losses['train']:.4f} | "
                f"evol val {losses['val']:.4f}"
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


        if it % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_save_path = f"checkpoints/evol_sft_hira_from_pretrain_iter_{it}.pt"
            if previous_ckpt_path:
                os.remove(previous_ckpt_path)
            modules.run_save_checkpoint(model, optimizer, it, ckpt_save_path)
            previous_ckpt_path = ckpt_save_path
            print(f"saved checkpoint to {ckpt_save_path}")

    torch.save(
        {"model": model.state_dict(), "it": max_iters},
        final_path,
    )
    print(f"saved final weights to {final_path}")
    wandb.finish()


# ====== Bench: LoRA + Cross Entropy Error

class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        weight, bias = get_linear_weight_and_bias(base)
        self.W_0 = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None
        d_in, d_out = self.W_0.shape
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d_out))
    def forward(self, x):
        out = (x @ self.W_0 + self.scale * (x @ self.A @ self.B))
        if self.bias is not None:
            out = out + self.bias
        return out

import torch.nn.functional as F
    
def ce_loss(logits, targets, ignore_index=-666):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )

def get_lora_ab_norm(model):
    A_norm = 0.0
    B_norm = 0.0
    cnt = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            A_norm += m.A.norm().item()
            B_norm += m.B.norm().item()
            cnt += 1
    if cnt == 0:
        return 0.0, 0.0
    return A_norm / cnt, B_norm / cnt

def get_lora_update_ratio(model):
    ratios = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            delta = m.A @ m.B
            scale = m.scale
            update_norm = (scale * delta).norm()
            original_norm = m.W_0.norm()
            if original_norm > 0:
                ratios.append(update_norm.item() / original_norm.item())
    if len(ratios) == 0:
        return 0.0
    return sum(ratios) / len(ratios)

def sft_LoRA():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = 0
    world_size = 1

    context_length = 1024
    batch_size = 8

    d_model = 1024
    num_layers = 24
    num_heads = 16
    d_ff = 2752
    rope_theta = 10000

    max_iters = 42000
    eval_interval = 500
    eval_iters = 50
    log_interval = 50
    checkpoint_interval = 5000

    max_learning_rate = 5e-4
    min_learning_rate = 2e-4
    warmup_iters = 2000
    cosine_cycle_iters = max_iters

    weight_decay = 0.01
    betas = (0.9, 0.95)
    eps = 1e-8
    max_grad_norm = 1.0

    tokens_per_iter = batch_size * context_length * world_size

    tk, vocab_size = load_tokenizer()

    seed = 1337 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if rank == 0:
        print("单卡模式")
        print("rank:", rank)
        print("world_size:", world_size)
        print("device:", device)

    obj = torch.load(PRETRAIN_CKPT_PATH, map_location="cpu")
    state_dict = obj["model"]
    

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    
    with open('./pswd.json') as file:
        pswds = json.load(file)
    os.environ['WANDB_API_KEY'] = pswds["wandb-api-key"]
    
    wandb.init(
        project='RE_gpt2vision_sft_chat_templetes',
        config={
        },
        name='LoRA Bench-r-512 ' + nowtime,
        save_code=True
    )

    model = modules.TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
        )

    model.load_state_dict(state_dict)


    for l in model.layers:
        l.attn.q_proj = LoRALinear(l.attn.q_proj)
        l.attn.k_proj = LoRALinear(l.attn.k_proj)
        l.attn.v_proj = LoRALinear(l.attn.v_proj)
        l.attn.o_proj = LoRALinear(l.attn.o_proj)


    


    for _, param in model.named_parameters():
        param.requires_grad = False

    for n, p in model.named_parameters():
        if "A" in n or "B" in n:
            p.requires_grad = True

    model.to(device)

    print("===== LoRA replacements all done")
    model.train()

    optimizer = modules.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}

        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters, device=device)

            for k in range(eval_iters):
                x, y = get_batch_from_json(
                    json_data=json_data,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                    tokenizer=tk,
                    for_valid=(split == "val"),
                    ignore_index=-666
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    logits = model(x)
                    loss = modules.run_cross_entropy_for_gem(logits, y, ignore_index=-666)

                losses[k] = loss

            mean_loss = losses.mean()
            out[split] = mean_loss.item()

        model.train()
        return out

    json_data = []

    with open('./data/alpaca_evol_instruct_70k.json') as file:
        json_data = json.load(file) 


    prev_ckpt_path = ''

    import time

    t0 = time.time()    

    for it in range(int(max_iters)):
        lr = modules.run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch_from_json(
            json_data=json_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            tokenizer=tk
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits = model(x)

            loss = modules.run_cross_entropy_for_gem(logits, y, ignore_index=-666)

            

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        modules.run_gradient_clipping(
            model.parameters(),
            max_l2_norm=max_grad_norm,
        )

        optimizer.step()


        if rank == 0 and it % log_interval == 0:
            dt = time.time() - t0

            tokens_in_window = tokens_per_iter * log_interval

            tok_s = tokens_in_window / dt

            tokens_processed = it * tokens_per_iter

            print(
                f"iter {it} "
                f"loss {loss.item():.4f} "
                f"lr {lr:.6e} | "
                f"tok/s {tok_s:.0f} | "
                f"tokens {tokens_processed / 1e9:.3f}B"
            )

            wandb.log({
                "iter": it,
                "lr": lr,
                "loss": loss.item(),
                "tok/s": tok_s,
                "tokens_B": tokens_processed / 1e9
            })


            a_norm, b_norm = get_lora_ab_norm(model)

            wandb.log({
                "adapter/A_norm": a_norm,
                "adapter/B_norm": b_norm,
            })

            wandb.log({"adapter/update_ratio": get_lora_update_ratio(model)})



            t0 = time.time()

        if it % eval_interval == 0:
            losses = estimate_loss()
            if rank == 0:
                print(
                    f"[eval] iter {it:8d} | "
                    f"train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f}"
                )
                wandb.log({
                    "[eval] iter": it,
                    "[eval] train_loss": losses['train'],
                    "[eval] val_loss": losses['val']
                })

        if rank == 0 and it > 0 and it % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/sft_EvolSft_LoRA_BENCH_r-512_gpt2med_iter_{it}.pt"

            if prev_ckpt_path != '':
                os.remove(prev_ckpt_path)

            prev_ckpt_path = ckpt_path

            modules.run_save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=it,
                out=ckpt_path,
            )
            print(f"saved checkpoint to {ckpt_path}")

    obj = {}
    obj["model"] = model.state_dict()
    torch.save(obj, './data/weights-sft-1-EvolSft-LoRA-Bench-r-512-text.pt')

    wandb.finish()

if __name__ == "__main__":
    i = input("1 for LoRA Bench, 2 for HiRA: ")
    if i == "1":
        sft_LoRA()
    else:
        sft()
