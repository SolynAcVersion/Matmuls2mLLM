import os
import random
import re
import numpy as np
import torch
import datetime
import wandb

import math

import modules
import torch.nn as nn

import json

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_VOCAB_SIZE = 32000
VOCAB_PATH = './data/owt_train_32004.pickle'
MERGES_PATH = './data/owt_train_32000_merges.pickle'
SFT_RESUME_CKPT_PATH = os.environ.get(
    "SFT_RESUME_CKPT_PATH",
    './data/sharegpt_direct30_hira_r16_from15000_final.pt',
)
SFT_RESUME_ITER = os.environ.get("SFT_RESUME_ITER", "")
SFT_TARGET_ITER = os.environ.get("SFT_TARGET_ITER", "")
SFT_DEFAULT_TARGET_ITER = int(os.environ.get("SFT_DEFAULT_TARGET_ITER", "30000"))
SFT_EXTRA_ITERS = int(os.environ.get("SFT_EXTRA_ITERS", "5000"))
SFT_RESET_LR_SCHEDULE = os.environ.get("SFT_RESET_LR_SCHEDULE", "1") == "1"
SFT_RESUME_OPTIM = os.environ.get("SFT_RESUME_OPTIM", "0") == "1"
SFT_SHAREGPT_DATA_PATH = os.environ.get("SFT_SHAREGPT_DATA_PATH", './data/train.jsonl')
SFT_CKPT_PREFIX = os.environ.get(
    "SFT_CKPT_PREFIX",
    "sharegpt_only_hira_continue",
)
SFT_FINAL_PATH = os.environ.get(
    "SFT_FINAL_PATH",
    './data/sharegpt_only_hira_continue_final.pt',
)

CHAT_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad|>",
]


def checkpoint_iter_from_path(path):
    name = os.path.basename(path)
    patterns = [
        r"_iter_(\d+)\.pt$",
        r"_from(\d+)(?:_|\.pt$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match is not None:
            return int(match.group(1))
    return -1


def resolve_resume_iter(ckpt_obj, ckpt_path):
    if "it" in ckpt_obj:
        return int(ckpt_obj["it"])
    if SFT_RESUME_ITER.strip() != "":
        return int(SFT_RESUME_ITER)

    parsed_iter = checkpoint_iter_from_path(ckpt_path)
    if parsed_iter >= 0:
        return parsed_iter

    raise RuntimeError(
        "cannot infer checkpoint iteration. Set SFT_RESUME_ITER explicitly "
        "or use a checkpoint that contains an 'it' field / _iter_N filename."
    )


def resolve_target_iter(resume_iter):
    if SFT_TARGET_ITER.strip() != "":
        target_iter = int(SFT_TARGET_ITER)
    else:
        target_iter = max(SFT_DEFAULT_TARGET_ITER, resume_iter + SFT_EXTRA_ITERS)

    if target_iter <= resume_iter:
        raise RuntimeError(
            f"target_iter={target_iter} must be greater than resume_iter={resume_iter}"
        )
    return target_iter


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


def load_sharegpt_pairs(path):
    pairs = []
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            conversation = json.loads(line)["conversations"]
            for idx in range(0, len(conversation) - 1, 2):
                if conversation[idx].get("user") != "human":
                    continue
                if conversation[idx + 1].get("user") != "gpt":
                    continue
                instruction = conversation[idx].get("text", "").strip()
                output = conversation[idx + 1].get("text", "").strip()
                if instruction and output:
                    pairs.append({"instruction": instruction, "output": output})
    return pairs


def encode_sharegpt_pair(example, tk, ignore: int=-666) -> (list, list):
    front = f'<|user|>\n{example["instruction"]}\n<|assistant|>\n'
    back = f'{example["output"]}<|endoftext|>'
    front_tokens = tk.encode(front)
    back_tokens = tk.encode(back)
    return front_tokens + back_tokens, [ignore] * len(front_tokens) + back_tokens


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
                    f"failed to sample a ShareGPT pair within context_length={context_length}"
                )
            example = random.choice(data)
            ids, labels = encode_sharegpt_pair(example, tokenizer, ignore_index)
            if len(ids) <= context_length:
                break


        x[b, : len(ids)] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )

        y[b, : len(labels)] = torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )

    return x, y




def get_linear_weight_and_bias(base):
    if hasattr(base, "W"):
        weight = base.W
    elif hasattr(base, "weight"):
        weight = base.weight.T
    else:
        raise TypeError(f"unsupported linear module: {type(base)!r}")

    bias = getattr(base, "bias", None)
    return weight, bias





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
    # Evol-SFT & GEM
    # device = torch.device("cpu")
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

    max_iters = 10000
    eval_interval = 500
    eval_iters = 50
    log_interval = 50
    checkpoint_interval = 1000

    max_learning_rate = 2e-5
    min_learning_rate = 5e-6
    warmup_iters = 500
    cosine_cycle_iters = max_iters

    weight_decay = 0.01
    betas = (0.9, 0.95)
    eps = 1e-8
    max_grad_norm = 0.5
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

    with open('./pswd.json') as file:
        os.environ['WANDB_API_KEY'] = json.load(file)["wandb-api-key"]
    
    wandb.init(
        project='gpt2vision_sft_chat_templetes',
        config={
            "ckpt_path": './data/sharegpt_direct30_hira_r16_from15000_final.pt',
            "sharegpt_data_path": './data/train.jsonl',
            "max_iters": max_iters,
            "batch_size": batch_size,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_iters": warmup_iters,
            "max_grad_norm": max_grad_norm,
        },
        name='[sharegpt-only] HiRA continue ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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

    for l in model.layers:
        l.attn.q_proj = HiRALinear(l.attn.q_proj)
        l.attn.k_proj = HiRALinear(l.attn.k_proj)
        l.attn.v_proj = HiRALinear(l.attn.v_proj)
        l.ffn.w2 = HiRALinear(l.ffn.w2)
        l.ffn.w3 = HiRALinear(l.ffn.w3)


    


    for param in model.parameters():
        param.requires_grad = False

    base_vocab_size = 32000
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

    for n, p in model.named_parameters():
        if ".A" in n or ".B" in n:
            p.requires_grad = True

    obj = torch.load('./data/sharegpt_direct30_hira_r16_from15000_final.pt', map_location="cpu")
    model.load_state_dict(obj["model"])

    model.to(device)

    print("===== HiRA replacements all done")
    print("loaded checkpoint:", './data/sharegpt_direct30_hira_r16_from15000_final.pt')
    print("initial HiRA update_ratio", get_hira_update_ratio(model))

    trainable = 0
    total = 0

    for n,p in model.named_parameters():

        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()

    print(
        f"trainable={trainable:,}"
    )

    print(
        f"total={total:,}"
    )

    print(
        f"ratio={100*trainable/total:.4f}%"
    )

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

    json_data = load_sharegpt_pairs('./data/train.jsonl')
    random.shuffle(json_data)


    prev_ckpt_path = ''

    import time

    t0 = time.time()    

    for it in range(1, int(max_iters) + 1):
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

        hira_update_ratio = get_hira_update_ratio(model)
        # if hira_update_ratio > max_hira_update_ratio:
        #     raise RuntimeError(
        #         f"HiRA update_ratio too high: {hira_update_ratio:.6f} "
        #         f"> {max_hira_update_ratio}. Stop training and lower lr/alpha."
        #     )


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
                f"update_ratio {hira_update_ratio}"
            )

            a_norm, b_norm = get_hira_ab_norm(model)

            wandb.log({
                "iter": it,
                "lr": lr,
                "loss": loss.item(),
                "tok/s": tok_s,
                "tokens_B": tokens_processed / 1e9,
                "adapter/A_norm": a_norm,
                "adapter/B_norm": b_norm,
                "adapter/update_ratio": hira_update_ratio,
            })



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
            ckpt_path = f"checkpoints/sharegpt_only_hira_continue_iter_{it}.pt"

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
    torch.save(obj, './data/sharegpt_only_hira_continue_final.pt')

    wandb.finish()

if __name__ == "__main__":
    sft()
