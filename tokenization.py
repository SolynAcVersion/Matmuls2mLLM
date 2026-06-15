import modules
import train

def a():

    import glob, os, pandas as pd

    vocab = modules.load_with_pickle("./data/owt_train_32004.pickle")
    merges = modules.load_with_pickle("./data/owt_train_32000_merges.pickle")

    output_dir = './data/my-npy'
    shard_size = 500_000_000

    parquet_dir = "./data/dataset"
    parquet_files_dirs = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    parquet_files_dirs = [f for f in parquet_files_dirs if not f.endswith(".parquet.parquet")]

    buffer = []
    shard_idx = 0
    eot_idx = 32000
    

    tk = modules.FastTokenizerOWTHighPerformance(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"],
        cache_size=1_000_000,
        use_cpp=True,
    )

    train_txt = "./data/wiki_temp_train.txt"
    valid_txt = "./data/wiki_temp_valid.txt"

    parquet_files_dirs = sorted(parquet_files_dirs)

    with open(train_txt, "w", encoding="utf-8") as file_train:
        for p in parquet_files_dirs[: 450]:
            cont = pd.read_parquet(p)
            for text in cont['text']:
                file_train.write(text + '\n')

    with open(valid_txt, "w", encoding="utf-8") as file_train:
        for p in parquet_files_dirs[450: ]:
            cont = pd.read_parquet(p)
            for text in cont['text']:
                file_train.write(text + '\n')

    print("=====txt all created")

    modules.make_npy_owt_high_performance(
        tk=tk,
        inputPath=(
            train_txt,
            valid_txt
        ),
        outputPath="./data/wiki_npys",
        lines=-1,
        shard_size=500_000_000,
        batch_lines=2048,
        num_workers=8,
        use_cpp=True,
    )



def decodeText(l):
    vocab = modules.load_with_pickle('./data/owt_train_32009.pickle')
    merges = modules.load_with_pickle('./data/owt_train_32000_merges.pickle')
    tk = modules.Tokenizer(vocab, merges, ["<|endoftext|>"])
    return tk.decode(l)

def train_bpe(path):
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
    modules.save_with_pickle(vocab, './data/owt_train_32009.pickle')
    modules.save_with_pickle(merges, './data/owt_train_32000_merges.pickle')

if __name__ == "__main__":
    a()