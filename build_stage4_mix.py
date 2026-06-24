import json
import random
from collections import Counter
from pathlib import Path


def main():
    seed = 1337
    out = Path("./data/re_sft_stage4_mix.jsonl")
    base_candidates = [Path("./data/re_sft_stage3j_mix.jsonl"), Path("./data/train_stage3j_mix_106k.jsonl")]
    base = None
    for path in base_candidates:
        if path.exists():
            base = path
            break
    if base is None:
        raise RuntimeError("missing base stage3j mix")

    rows = []
    with open(base, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            instr = str(obj.get("instruction", "")).strip()
            output = str(obj.get("output", "")).strip()
            if not instr or not output:
                continue
            rows.append({
                "instruction": instr,
                "output": output,
                "task": str(obj.get("task", "assistant_qa")).strip() or "assistant_qa",
                "source": str(obj.get("source", "stage3j_base")).strip() or "stage3j_base",
                "split_key": str(obj.get("split_key", f"{instr}\t{output}")),
            })

    seen = set()
    base_rows = []
    for row in rows:
        key = (row["instruction"], row["output"])
        if key in seen:
            continue
        seen.add(key)
        base_rows.append(row)

    rng = random.Random(seed)

    repeat_templates = ["Repeat exactly: {}", "Copy this exactly: {}", "Echo exactly: {}", "Output exactly: {}"]
    common_payloads = "Hello Thanks Goodbye Yes No Maybe Red Blue Green Yellow Black White Apple Water Fire House Chair Table River Mountain Flower Music Small Large Quiet Bright Circle Square Cat Dog Bird".split()
    syllables = "ba be bi bo bu da de di do du fa fe fi fo fu ga ge gi go gu ha he hi ho hu ka ke ki ko ku la le li lo lu ma me mi mo mu na ne ni no nu pa pe pi po pu ra re ri ro ru sa se si so su ta te ti to tu va ve vi vo vu wa we wi wo wu ya ye yi yo yu".split()
    repeat_rows = []
    repeat_seen = set()
    tries = 0
    while len(repeat_rows) < 2000 and tries < 60000:
        tries += 1
        if rng.random() < 0.3:
            payload = rng.choice(common_payloads)
        else:
            payload = " ".join(
                "".join(rng.choice(syllables) for _ in range(rng.randint(2, 4)))
                for _ in range(rng.randint(1, 5))
            )
        if rng.random() < 0.15:
            payload = f"{rng.randint(10, 999)} {payload}"
        instruction = rng.choice(repeat_templates).format(payload)
        key = (instruction, payload)
        if key in repeat_seen:
            continue
        repeat_seen.add(key)
        repeat_rows.append({
            "instruction": instruction,
            "output": payload,
            "task": "repeat",
            "source": "synthetic_stage4_repeat",
            "split_key": f"{instruction}\t{payload}",
        })

    groups = [
        ("animal", ["cat", "dog", "bird", "fish", "horse", "rabbit", "bear", "lion", "tiger", "cow"]),
        ("fruit", ["apple", "banana", "orange", "pear", "grape", "peach", "mango", "lemon", "plum", "cherry"]),
        ("vehicle", ["car", "bus", "train", "bike", "boat", "truck", "plane", "ship", "scooter", "motorcycle"]),
        ("tool", ["hammer", "screwdriver", "wrench", "saw", "drill", "pliers", "shovel", "ladder", "broom", "rake"]),
        ("clothing item", ["shirt", "pants", "socks", "hat", "jacket", "shoes", "gloves", "scarf", "coat", "belt"]),
        ("musical instrument", ["piano", "guitar", "violin", "drum", "flute", "trumpet", "harp", "cello", "saxophone", "ukulele"]),
        ("color", ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "brown", "orange"]),
        ("body part", ["head", "hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg", "back"]),
        ("drink", ["water", "tea", "coffee", "milk", "juice", "soda", "lemonade", "broth", "smoothie", "cocoa"]),
        ("food", ["bread", "rice", "cheese", "soup", "salad", "pizza", "pasta", "cookie", "cake", "cereal"]),
    ]
    words = []
    for _, xs in groups:
        words.extend(xs)
    wraps = ["", "Answer yes or no: ", "Please answer yes or no: "]
    yesno_rows = []
    yesno_seen = set()
    for wrap in wraps:
        for word in words:
            aw = "an" if word[:1].lower() in "aeiou" else "a"
            for cat, xs in groups:
                ac = "an" if cat[:1].lower() in "aeiou" else "a"
                instruction = f"{wrap}Is {aw} {word} {ac} {cat}?"
                output = "yes" if word in xs else "no"
                key = (instruction, output)
                if key in yesno_seen:
                    continue
                yesno_seen.add(key)
                yesno_rows.append({
                    "instruction": instruction,
                    "output": output,
                    "task": "yesno",
                    "source": "synthetic_stage4_yesno_clean",
                    "split_key": f"{instruction}\t{output}",
                })
                if len(yesno_rows) >= 2000:
                    break
            if len(yesno_rows) >= 2000:
                break
        if len(yesno_rows) >= 2000:
            break

    ood_subjects = [
        "moon soup",
        "invisible contest",
        "weekday color",
        "exact temperature of a thought",
        "password to the ocean",
        "sound of a triangle",
        "age of a rumor",
        "north wind owner",
        "language of clouds",
        "third smell of rain",
        "shape of yesterday",
        "weight of silence",
        "capital of Atlantis",
        "number of invisible doors",
        "secret ingredient in moon soup",
        "color of a memory",
        "taste of a clock",
        "height of a question",
        "name of the last star",
        "recipe for a shadow",
    ]
    ood_wraps = [
        "What is the {}?",
        "Can you tell me the {}?",
        "How do I find the {}?",
        "Who knows the {}?",
        "Which city owns the {}?",
        "What is the password for the {}?",
        "Where can I buy the {}?",
        "Why does the {} matter?",
        "How many parts does the {} have?",
        "What color is the {}?",
    ]
    ood_rows = []
    for wrap in ood_wraps:
        for subject in ood_subjects:
            instruction = wrap.format(subject)
            ood_rows.append({
                "instruction": instruction,
                "output": "I don't know.",
                "task": "ood_refusal",
                "source": "synthetic_stage4_ood_refusal",
                "split_key": f"{instruction}\tI don't know.",
            })

    rng = random.Random(seed + 99)
    rows = base_rows + repeat_rows + yesno_rows + ood_rows
    rng.shuffle(rows)
    seen = set()
    out_rows = []
    for row in rows:
        key = (row["instruction"], row["output"])
        if key in seen:
            continue
        seen.add(key)
        out_rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"base={base}")
    print(f"wrote {len(out_rows)} rows to {out}")
    print(dict(sorted(Counter(row["task"] for row in out_rows).items())))


if __name__ == "__main__":
    main()
