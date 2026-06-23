import os
import glob

import pandas as pd

import modules


def train_bpe():
    # 在 owt 语料上训练一个 32000 词表的 BPE, 存成 pickle
    vocab, merges = modules.run_train_bpe(
        "./data/owt_train.txt",
        32000,
        ["<|endoftext|>"],
        chunk_size_mb=32,
        min_freq=2,
        max_pretokens=1_000_000,
        use_cpp=True,
        verbose=True,
    )
    modules.save_with_pickle(vocab, "./data/owt_train_32009.pickle")
    modules.save_with_pickle(merges, "./data/owt_train_32000_merges.pickle")


def decode_text(ids):
    vocab = modules.load_with_pickle("./data/owt_train_32009.pickle")
    merges = modules.load_with_pickle("./data/owt_train_32000_merges.pickle")
    tk = modules.Tokenizer(vocab, merges, ["<|endoftext|>"])
    return tk.decode(ids)


def main():
    # 把 wiki parquet 切成 train/valid 文本, 再用 BPE 编码成 npy 分片供预训练读取
    vocab = modules.load_with_pickle("./data/owt_train_32004.pickle")
    merges = modules.load_with_pickle("./data/owt_train_32000_merges.pickle")

    parquet_dir = "./data/dataset"
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    parquet_files = [f for f in parquet_files if not f.endswith(".parquet.parquet")]
    parquet_files = sorted(parquet_files)

    tk = modules.FastTokenizerOWTHighPerformance(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"],
        cache_size=1_000_000,
        use_cpp=True,
    )

    train_txt = "./data/wiki_temp_train.txt"
    valid_txt = "./data/wiki_temp_valid.txt"

    # 前 450 个 parquet 当训练集, 其余当验证集
    with open(train_txt, "w", encoding="utf-8") as f:
        for p in parquet_files[:450]:
            cont = pd.read_parquet(p)
            for text in cont["text"]:
                f.write(text + "\n")

    with open(valid_txt, "w", encoding="utf-8") as f:
        for p in parquet_files[450:]:
            cont = pd.read_parquet(p)
            for text in cont["text"]:
                f.write(text + "\n")

    print("=====txt all created")

    modules.make_npy_owt_high_performance(
        tk=tk,
        inputPath=(train_txt, valid_txt),
        outputPath="./data/wiki_npys",
        lines=-1,
        shard_size=500_000_000,
        batch_lines=2048,
        num_workers=8,
        use_cpp=True,
    )


if __name__ == "__main__":
    main()
