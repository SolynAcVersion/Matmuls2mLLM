import argparse
import json
import pickle
import random
import re
from pathlib import Path


DEFAULT_VOCAB_PATH = "./data/owt_train_32004.pickle"
DEFAULT_OUT_PATH = "./data/synthetic_skills_v3.jsonl"


SEED_WORDS = [
    "cat", "dog", "sun", "hat", "run", "big", "red", "hot", "ice", "fly",
    "book", "tree", "moon", "star", "fish", "rain", "fire", "cloud", "apple", "happy",
    "hello", "world", "green", "brown", "blue", "black", "white", "light", "smile", "heart",
    "river", "ocean", "stone", "grass", "plant", "flower", "garden", "shadow", "window", "bridge",
    "simple", "bright", "little", "quick", "quiet", "sweet", "friend", "summer", "winter", "animal",
    "mountain", "rainbow", "butterfly", "chocolate", "puzzle", "memory", "picture", "music", "answer", "letter",
    "number", "report", "season", "travel", "future", "present", "minute", "rocket", "basket", "ladder",
    "candle", "silver", "golden", "orange", "purple", "engine", "blanket", "school", "family", "market",
]


COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "get", "go", "had", "has", "have", "he", "her", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "may", "me", "more", "most", "my", "no", "not", "of",
    "on", "one", "or", "our", "out", "over", "said", "say", "she", "should", "so", "some", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "too", "under",
    "up", "us", "very", "was", "we", "well", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "yours",
    "con", "pro", "com", "comp", "inter", "trans", "sub", "pre", "dis", "re", "un", "non", "anti",
    "auto", "micro", "photo", "graph", "ment", "tion", "sion", "ness", "able", "ible", "less", "ful",
    "ism", "ist", "ity", "ive", "ous", "al", "ary", "ery", "ing", "ed", "ly", "er", "or", "est",
    "app", "int", "res", "rel", "rec", "sec", "sim", "spec", "str", "dev", "fin", "gov", "pol",
    "tech", "sys", "man", "col", "dec", "imp", "inc", "inv", "pro", "pub", "priv", "qual", "reg",
}


ADD_TEMPLATES = [
    "{a} + {b} =",
    "What is {a} plus {b}?",
    "Calculate {a} + {b}",
    "Can you solve {a} + {b}?",
    "Find the sum of {a} and {b}",
    "Add {a} and {b}",
    "Compute {a} + {b}",
    "How much is {a} + {b}?",
    "{a} added to {b}",
]

SUB_TEMPLATES = [
    "{a} - {b} =",
    "What is {a} minus {b}?",
    "Calculate {a} - {b}",
    "Can you solve {a} - {b}?",
    "Find the difference between {a} and {b}",
    "Subtract {b} from {a}",
    "Compute {a} - {b}",
    "How much is {a} - {b}?",
]

MUL_TEMPLATES = [
    "{a} * {b} =",
    "What is {a} times {b}?",
    "Calculate {a} * {b}",
    "Can you solve {a} * {b}?",
    "Find the product of {a} and {b}",
    "Multiply {a} and {b}",
    "Compute {a} * {b}",
    "How much is {a} * {b}?",
]

SAY_TEMPLATES = [
    "Say the word: {w}",
    "Repeat after me: {w}",
    "Just say {w}",
    "Echo: {w}",
    "Output the word: {w}",
    "Can you say {w}?",
    "Please say {w}",
]

COPY_TEMPLATES = [
    "Copy this: {p}",
    "Repeat this exactly: {p}",
    "Output the following: {p}",
    "Just copy: {p}",
]

SPELL_TEMPLATES = [
    "Spell the word '{w}'",
    "How do you spell '{w}'?",
    "Give me the spelling of '{w}'",
    "Spell '{w}' for me",
]

COUNT_TEMPLATES = [
    "How many letters are in '{w}'?",
    "Count the letters in '{w}'",
    "What is the length of the word '{w}'?",
    "How many characters in '{w}'?",
]


ONSETS = [
    "", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "z",
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw", "ch", "sh", "th",
]

NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "ie", "oa", "oo", "ou", "ue", "ay", "ow", "ui"]

CODAS = [
    "", "n", "m", "s", "t", "r", "l", "d", "k", "p", "g", "ck", "sh", "th", "nd", "nt", "mp",
    "st", "rd", "rn", "ld", "ng",
]


def load_vocab_words(vocab_path: str, min_len: int = 5, max_len: int = 12) -> list[str]:
    vocab = pickle.load(open(vocab_path, "rb"))
    words: set[str] = set()

    for token_bytes in vocab.values():
        if not isinstance(token_bytes, bytes):
            continue
        try:
            text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        word = text.strip().lower()
        if not re.fullmatch(r"[a-z]+", word):
            continue
        if not (min_len <= len(word) <= max_len):
            continue
        if word in COMMON_STOPWORDS:
            continue
        words.add(word)

    return sorted(words)


def build_word_pools(vocab_path: str):
    real_words = load_vocab_words(vocab_path)
    short_words = sorted({w for w in SEED_WORDS if 3 <= len(w) <= 4})
    medium_words = sorted({w for w in SEED_WORDS if 5 <= len(w) <= 6} | {w for w in real_words if 5 <= len(w) <= 6})
    long_words = sorted({w for w in SEED_WORDS if len(w) >= 7} | {w for w in real_words if len(w) >= 7})
    phrase_words = sorted({w for w in SEED_WORDS if 4 <= len(w) <= 10} | {w for w in real_words if 4 <= len(w) <= 10})

    if not short_words:
        short_words = ["cat", "dog", "sun", "hat", "run"]
    if not medium_words:
        medium_words = ["apple", "happy", "cloud", "river", "hello"]
    if not long_words:
        long_words = ["mountain", "rainbow", "butterfly", "chocolate"]
    if not phrase_words:
        phrase_words = short_words + medium_words + long_words

    return {
        "short": short_words,
        "medium": medium_words,
        "long": long_words,
        "phrase": phrase_words,
    }


def make_pseudoword(rng: random.Random, min_len: int = 4, max_len: int = 10) -> str:
    for _ in range(100):
        syllables = rng.randint(1, 3)
        parts = []
        for _ in range(syllables):
            onset = rng.choice(ONSETS)
            nucleus = rng.choice(NUCLEI)
            coda = rng.choice(CODAS) if rng.random() < 0.7 else ""
            parts.append(onset + nucleus + coda)
        word = "".join(parts).lower()
        if min_len <= len(word) <= max_len and re.fullmatch(r"[a-z]+", word):
            return word

    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    length = rng.randint(min_len, max_len)
    chars = []
    for i in range(length):
        chars.append(rng.choice(consonants if i % 2 == 0 else vowels))
    return "".join(chars)


def choose_word(rng: random.Random, pools: dict[str, list[str]], bucket: str, pseudo_ratio: float = 0.15) -> str:
    if bucket == "short":
        pool = pools["short"]
        pseudo_min, pseudo_max = 3, 5
    elif bucket == "medium":
        pool = pools["medium"]
        pseudo_min, pseudo_max = 5, 7
    else:
        pool = pools["long"]
        pseudo_min, pseudo_max = 7, 10

    if not pool or rng.random() < pseudo_ratio:
        return make_pseudoword(rng, pseudo_min, pseudo_max)
    return rng.choice(pool)


def make_phrase(rng: random.Random, pools: dict[str, list[str]]) -> str:
    pool = pools["phrase"]
    word_count = rng.randint(2, 4)
    if len(pool) >= word_count:
        words = rng.sample(pool, word_count)
    else:
        words = [rng.choice(pool) for _ in range(word_count)]
    phrase = " ".join(words)
    if phrase:
        phrase = phrase[0].upper() + phrase[1:]
    return phrase


def generate_math_sample(rng: random.Random) -> dict[str, str]:
    op = rng.choice(["add", "sub", "mul"])
    if op == "add":
        a, b = rng.randint(0, 200), rng.randint(0, 200)
        ans = a + b
        template = rng.choice(ADD_TEMPLATES)
    elif op == "sub":
        a, b = rng.randint(0, 200), rng.randint(0, 200)
        if a < b:
            a, b = b, a
        ans = a - b
        template = rng.choice(SUB_TEMPLATES)
    else:
        a, b = rng.randint(0, 25), rng.randint(0, 25)
        ans = a * b
        template = rng.choice(MUL_TEMPLATES)

    return {"instruction": template.format(a=a, b=b), "output": str(ans)}


def generate_copy_sample(rng: random.Random, pools: dict[str, list[str]]) -> dict[str, str]:
    if rng.random() < 0.55:
        word = choose_word(rng, pools, rng.choices(["short", "medium", "long"], weights=[3, 5, 2])[0], pseudo_ratio=0.10)
        template = rng.choice(SAY_TEMPLATES)
        return {"instruction": template.format(w=word), "output": word}

    phrase = make_phrase(rng, pools)
    template = rng.choice(COPY_TEMPLATES)
    return {"instruction": template.format(p=phrase), "output": phrase}


def generate_spell_sample(rng: random.Random, pools: dict[str, list[str]]) -> dict[str, str]:
    bucket = rng.choices(["short", "medium", "long"], weights=[2, 5, 3])[0]
    word = choose_word(rng, pools, bucket, pseudo_ratio=0.20)
    inst = rng.choice(SPELL_TEMPLATES).format(w=word)
    return {"instruction": inst, "output": "-".join(word.lower())}


def generate_count_sample(rng: random.Random, pools: dict[str, list[str]]) -> dict[str, str]:
    bucket = rng.choices(["short", "medium", "long"], weights=[1, 4, 5])[0]
    word = choose_word(rng, pools, bucket, pseudo_ratio=0.15)
    inst = rng.choice(COUNT_TEMPLATES).format(w=word)
    return {"instruction": inst, "output": str(len(word))}


def generate_dataset(total: int, seed: int, pools: dict[str, list[str]]):
    rng = random.Random(seed)
    targets = {
        "math": round(total * 0.40),
        "copy": round(total * 0.40),
        "spell": round(total * 0.10),
    }
    targets["count"] = total - sum(targets.values())

    samples = []
    seen = set()

    def add_sample(sample: dict[str, str]):
        key = (sample["instruction"], sample["output"])
        if key in seen:
            return False
        seen.add(key)
        samples.append(sample)
        return True

    generators = {
        "math": lambda: generate_math_sample(rng),
        "copy": lambda: generate_copy_sample(rng, pools),
        "spell": lambda: generate_spell_sample(rng, pools),
        "count": lambda: generate_count_sample(rng, pools),
    }

    kind_counts = {name: 0 for name in targets}
    for name in ["math", "copy", "spell", "count"]:
        target = targets[name]
        attempts = 0
        while kind_counts[name] < target:
            attempts += 1
            sample = generators[name]()
            if add_sample(sample):
                kind_counts[name] += 1
                continue
            if attempts > target * 100:
                raise RuntimeError(f"too many duplicate samples while generating {name}")

    rng.shuffle(samples)
    return samples, seen


def write_jsonl(rows, out_path: str):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic skills dataset.")
    parser.add_argument("--total", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    pools = build_word_pools(args.vocab)
    dataset, seen = generate_dataset(args.total, args.seed, pools)
    write_jsonl(dataset, args.out)

    print(f"wrote {len(dataset)} unique samples to {args.out}")
    print(f"short/medium/long/phrase pools: {len(pools['short'])}/{len(pools['medium'])}/{len(pools['long'])}/{len(pools['phrase'])}")
    print(f"unique pairs kept: {len(seen)}")


if __name__ == "__main__":
    main()
