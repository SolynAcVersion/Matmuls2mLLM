import os
import glob
import random
import numpy as np
import torch
import datetime
import wandb
import json
import re
import math

import modules

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_shards(pattern):
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No npy shards found: {pattern}")

    shards = [np.load(f, mmap_mode="r") for f in files]

    print(f"loaded {len(shards)} shards from {pattern}")
    for f, s in zip(files, shards):
        print(f"  {f}: {s.shape}")

    return shards


def find_latest_checkpoint(pattern):
    files = glob.glob(pattern)
    if len(files) == 0:
        return None

    def extract_iter(path):
        m = re.search(r"iter_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    files.sort(key=extract_iter)
    return files[-1]


def get_batch_from_shards(shards, batch_size, context_length, device):
    x = torch.empty(
        batch_size,
        context_length,
        dtype=torch.long,
        device=device,
    )

    y = torch.empty(
        batch_size,
        context_length,
        dtype=torch.long,
        device=device,
    )

    for b in range(batch_size):
        shard = random.choice(shards)

        max_start = shard.shape[0] - context_length - 1
        start = random.randint(0, max_start)

        x_np = shard[start : start + context_length]
        y_np = shard[start + 1 : start + context_length + 1]

        x[b] = torch.from_numpy(np.asarray(x_np, dtype=np.int64)).to(device)
        y[b] = torch.from_numpy(np.asarray(y_np, dtype=np.int64)).to(device)

    return x, y


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = 0
    world_size = 1

    vocab_size = 32000
    context_length = 1024
    batch_size = 8

    d_model = 1024
    num_layers = 24
    num_heads = 16
    d_ff = 2752
    rope_theta = 10000

    target_train_epochs = 1.0
    max_iters = None
    eval_interval = 1000
    eval_iters = 50
    log_interval = 50
    checkpoint_interval = 5000

    fresh_max_learning_rate = 2e-4
    resume_max_learning_rate = 5e-5
    max_learning_rate = fresh_max_learning_rate
    min_learning_rate = 2e-5
    warmup_iters = 2000

    weight_decay = 0.1
    betas = (0.9, 0.95)
    eps = 1e-8
    max_grad_norm = 1.0

    tokens_per_iter = batch_size * context_length * world_size

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
        print("batch_size:", batch_size)
        print("global_batch_size:", batch_size * world_size)
        print("tokens_per_iter:", tokens_per_iter)
    train_shards = load_shards("./data/wiki_npys-train--1-shard*.npy")
    valid_shards = load_shards("./data/wiki_npys-valid--1-shard0.npy")
    train_tokens = sum(shard.shape[0] for shard in train_shards)

    max_iters = math.ceil(train_tokens * target_train_epochs / tokens_per_iter)
    cosine_cycle_iters = max_iters
    resume_ckpt_path = find_latest_checkpoint("checkpoints/pretrain_gpt2med_iter_*.pt")
    if resume_ckpt_path is not None:
        max_learning_rate = resume_max_learning_rate

    if rank == 0:
        print("train_tokens:", train_tokens)
        print("target_train_epochs:", target_train_epochs)
        print("max_iters:", max_iters)
        print("fresh_max_learning_rate:", fresh_max_learning_rate)
        print("resume_max_learning_rate:", resume_max_learning_rate)
        print("max_learning_rate:", max_learning_rate)
        print("resume_ckpt_path:", resume_ckpt_path)

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    
    with open('./pswd.json') as file:
        pswds = json.load(file)
    os.environ['WANDB_API_KEY'] = pswds["wandb-api-key"]

    wandb.init(
        project='gpt2vision',
        config={
            "num_layers": num_layers,
            "weight_decay": weight_decay,
            "target_train_epochs": target_train_epochs,
            "train_tokens": train_tokens,
            "max_iters": max_iters,
            "batch_size": batch_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "context_length": context_length,
            "fresh_max_learning_rate": fresh_max_learning_rate,
            "resume_max_learning_rate": resume_max_learning_rate,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate
        },
        name='pretrain_wiki' + nowtime,
        save_code=True
    )

    raw_model = modules.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
    )
    raw_model.train()

    optimizer = modules.AdamW(
        raw_model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    model = raw_model

    prev_it = -1
    prev_ckpt_path = ""
    if resume_ckpt_path is not None:
        print(f"resuming from {resume_ckpt_path}")
        prev_it = modules.run_load_checkpoint(
            resume_ckpt_path,
            raw_model,
            optimizer,
        )
        prev_ckpt_path = resume_ckpt_path
        print(f"loaded checkpoint at iter {prev_it}")

    optimizer = modules.AdamW(
        raw_model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}

        for split, shards in [
            ("train", train_shards),
            ("val", valid_shards),
        ]:
            losses = torch.zeros(eval_iters, device=device)

            for k in range(eval_iters):
                x, y = get_batch_from_shards(
                    shards=shards,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )

                logits = model(x)

                loss = modules.run_cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                )

                losses[k] = loss

            mean_loss = losses.mean()
            out[split] = mean_loss.item()

        model.train()
        return out


    import time

    t0 = time.time()    

    for it in range(prev_it + 1, int(max_iters)):
        lr = modules.run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch_from_shards(
            shards=train_shards,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits = model(x)

            loss = modules.run_cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        modules.run_gradient_clipping(
            raw_model.parameters(),
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
                "tokens_B": tokens_processed / 1e9,
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
                    "[eval]iter": it,
                    "[eval]train": losses['train'],
                    "[eval]val": losses['val'],
                })

        if rank == 0 and it > 0 and it % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/pretrain_gpt2med_iter_{it}.pt"

            if prev_ckpt_path != '':
                os.remove(prev_ckpt_path)

            prev_ckpt_path = ckpt_path

            modules.run_save_checkpoint(
                model=raw_model,
                optimizer=optimizer,
                iteration=it,
                out=ckpt_path,
            )
            print(f"saved checkpoint to {ckpt_path}")

    obj = {}
    obj["model"] = model.state_dict()
    torch.save(obj, './data/weights-pretrain-0-wiki-text.pt')

    wandb.finish()


if __name__ == "__main__":
    main()
