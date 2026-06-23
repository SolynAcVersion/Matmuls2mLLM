import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


SEED = 1337
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
KEY_PATH = Path("pswd.json")
PART_DIR = Path("data/stage3_v4_parts")
FINAL_PATH = Path("data/re_sft_assistant_deepseek_v4.jsonl")
SOURCE = "deepseek_generated_v4"

TARGETS = {
    "assistant_qa": 6000,
    "short_qa": 6000,
    "yesno": 1500,
    "repeat": 5000,
    "identity": 200,
}

BATCH_SIZE = 40
SLEEP_SECONDS = 0.7
RESET_PARTS = False

INDEX_KINDS = {
    "1": "assistant_qa",
    "2": "short_qa",
    "3": "yesno",
    "4": "build",
}

DOMAINS = [
    "basic science explanations",
    "study and learning advice",
    "writing and communication help",
    "travel and daily planning",
    "food and cooking basics",
    "basic technology concepts",
    "workplace and social advice",
    "nature and everyday world knowledge",
    "home organization without cleaning hacks",
    "simple hobbies and personal routines",
]

SHORT_DOMAINS = [
    "geography",
    "biology",
    "chemistry",
    "physics",
    "history",
    "literature",
    "everyday objects",
    "animals",
    "food",
    "sports",
    "music",
    "basic categories",
]

ARITHMETIC_HINT = re.compile(r"(?i)(\d+\s*(?:\+|-|\*|/|=)\s*\d+|\b(?:plus|minus|times|multiply|divided|calculate|equation|solve|fraction|percent|arithmetic|count letters|number of letters)\b)")
CODE_HINT = re.compile(r"(?i)\b(?:code|python|java|javascript|typescript|function|class|sql|html|css|api|regex|program|algorithm|debug|compile|script)\b")
IDENTITY_HINT = re.compile(r"(?i)\b(?:your name|who are you|what are you called|assistant name|who created you|who made you|your creator|your developer|are you human|what are you)\b")
REPEAT_HINT = re.compile(r"(?i)^\s*(?:repeat(?: after me)?|copy(?: this)?|echo|say exactly|output exactly|say the word|say)\s*:")
GRAMMAR_HINT = re.compile(r"(?i)\b(?:rewrite|paraphrase|translate|summarize|correct grammar|spelling|comparative|superlative|morph|inflect|plural|singular)\b")
HIGH_STAKES_HINT = re.compile(r"(?i)\b(?:medical|diagnose|medicine|dosage|legal|lawsuit|contract|financial|invest|stock|tax|loan|insurance)\b")
OLD_CLUSTER_HINT = re.compile(r"(?i)\b(?:sticker|residue|baking soda|crayon|non-slip mat|cutting board|burnt pot|jar lid|cooking oil|peanut butter|magic eraser|splinter)\b")

STOP = set("a an and are as at be by can do does for from how i in is it me of on or should the this to what when where which who why with you your".split())


def clean(text):
    return re.sub(r"\s+", " ", str(text).strip())


def norm(text):
    text = clean(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return clean(text).split()


def word_count(text):
    return len(words(text))


def mostly_ascii(text):
    return bool(text) and sum(ord(ch) < 128 for ch in text) / max(1, len(text)) >= 0.98


def repeated_ngram(text, n=3):
    xs = [x.lower() for x in words(text)]
    grams = [" ".join(xs[i:i + n]) for i in range(max(0, len(xs) - n + 1))]
    return len(grams) != len(set(grams))


def split_key(task, instruction, output=""):
    if task == "repeat":
        return f"repeat:{norm(output)}"
    return f"{task}:{norm(instruction)}"


def bad_common(instruction, output):
    joined = f"{instruction} {output}"
    if not mostly_ascii(joined):
        return True
    if ARITHMETIC_HINT.search(joined):
        return True
    if CODE_HINT.search(joined):
        return True
    if HIGH_STAKES_HINT.search(joined):
        return True
    if OLD_CLUSTER_HINT.search(joined):
        return True
    if GRAMMAR_HINT.search(instruction):
        return True
    if repeated_ngram(output):
        return True
    lowered = output.lower()
    bad = ["as an ai", "i don't know", "i cannot", "i can't", "sorry", "not sure"]
    return any(x in lowered for x in bad)


def yesno(output):
    text = clean(output).lower().rstrip(".!")
    if text in {"yes", "y"}:
        return "yes"
    if text in {"no", "n"}:
        return "no"
    return ""


def item(task, instruction, output, source=SOURCE):
    return {
        "instruction": clean(instruction),
        "output": clean(output),
        "task": task,
        "source": source,
        "split_key": split_key(task, instruction, output),
    }


def valid_generated(obj, expected):
    if not isinstance(obj, dict):
        return None
    task = clean(obj.get("task", expected))
    instruction = clean(obj.get("instruction", ""))
    output = clean(obj.get("output", ""))
    if task != expected or not instruction or not output:
        return None
    if REPEAT_HINT.search(instruction) or IDENTITY_HINT.search(instruction):
        return None
    if bad_common(instruction, output):
        return None
    if task == "assistant_qa":
        if not (3 <= word_count(output) <= 80 and len(output) <= 600):
            return None
        if yesno(output):
            return None
    elif task == "short_qa":
        if yesno(output):
            return None
        if not (1 <= word_count(output) <= 6 and len(output) <= 80):
            return None
    elif task == "yesno":
        output = yesno(output)
        if not output:
            return None
    else:
        return None
    return item(task, instruction, output)


def valid_local(obj):
    if not isinstance(obj, dict):
        return None
    task = clean(obj.get("task", ""))
    instruction = clean(obj.get("instruction", ""))
    output = clean(obj.get("output", ""))
    if task == "repeat":
        if not REPEAT_HINT.search(instruction):
            return None
        if not output or len(output) > 80 or word_count(output) > 6:
            return None
        return item(task, instruction, output, clean(obj.get("source", "synthetic_repeat_v4")))
    if task == "identity":
        if output != "Shengoovlei":
            return None
        return item(task, instruction, output, clean(obj.get("source", "synthetic_identity_v4")))
    return valid_generated(obj, task)


def load_key():
    with open(KEY_PATH, encoding="utf-8") as f:
        obj = json.load(f)
    for key in ["deepseek-api-key", "deepseek_api_key", "DEEPSEEK_API_KEY"]:
        if obj.get(key):
            return obj[key]
    raise RuntimeError(f"missing DeepSeek key in {KEY_PATH}")


def call_api(api_key, prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 6000,
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_items(text):
    text = clean(text).removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    tries = [text]
    if "[" in text and "]" in text:
        tries.append(text[text.find("["):text.rfind("]") + 1])
    for x in tries:
        try:
            obj = json.loads(x)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    out = []
    for line in text.splitlines():
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            pass
    return out


def prompt_for(kind, n, rng):
    if kind == "assistant_qa":
        domain = rng.choice(DOMAINS)
        return f"""Return only a JSON array with exactly {n} objects.
Each object must have instruction, output, and task.
task must be "assistant_qa".
Domain: {domain}.
Write concise English assistant answers, 3 to 70 words each.
Use varied topics and varied first words.
Avoid household cleaning hacks, sticker residue, baking soda, crayon marks, non-slip mats, stuck jars, direct arithmetic, code, identity questions, medical, legal, and financial advice.
Examples:
{{"instruction":"Why do leaves change color in autumn?","output":"Leaves change color because chlorophyll breaks down, revealing yellow and orange pigments that were already present.","task":"assistant_qa"}}
{{"instruction":"What is a simple way to improve sleep quality?","output":"Keep a consistent bedtime and avoid bright screens shortly before sleeping.","task":"assistant_qa"}}"""
    if kind == "short_qa":
        domain = rng.choice(SHORT_DOMAINS)
        return f"""Return only a JSON array with exactly {n} objects.
Each object must have instruction, output, and task.
task must be "short_qa".
Domain: {domain}.
Outputs must be one to six words, no yes/no, no sentences, and no explanations.
Use direct fact questions or unambiguous category questions.
Avoid greetings, repeat/copy instructions, grammar transformations, direct arithmetic, identity questions, code, and vague prompts.
Examples:
{{"instruction":"What is the capital of France?","output":"Paris","task":"short_qa"}}
{{"instruction":"Classify this item: a spoon.","output":"Utensil","task":"short_qa"}}
{{"instruction":"What is the primary ingredient in guacamole?","output":"Avocado","task":"short_qa"}}"""
    return f"""Return only a JSON array with exactly {n} objects.
Each object must have instruction, output, and task.
task must be "yesno".
Outputs must be exactly "yes" or exactly "no", lowercase.
Use simple stable facts, everyday categories, and harmless preference-style questions.
Avoid arithmetic, code, identity questions, medical, legal, and financial advice.
Examples:
{{"instruction":"Is the sun a star?","output":"yes","task":"yesno"}}
{{"instruction":"Can a human breathe underwater without equipment?","output":"no","task":"yesno"}}"""


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append_jsonl(path, rows):
    PART_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_api(kind):
    path = PART_DIR / f"{kind}.jsonl"
    if RESET_PARTS and path.exists():
        path.unlink()
    existing = []
    seen = set()
    for obj in read_jsonl(path):
        row = valid_generated(obj, kind)
        if row:
            existing.append(row)
            seen.add(row["split_key"])
    target = TARGETS[kind]
    missing = max(0, target - len(existing))
    if missing == 0:
        print(f"{kind}: already_have={len(existing)} target={target}")
        return
    batch_count = len(existing) // BATCH_SIZE
    rng = random.Random(SEED + sum(ord(ch) for ch in kind) + batch_count + time.time_ns())
    api_key = load_key()
    made = 0
    tries = 0
    max_tries = max(20, ((missing + BATCH_SIZE - 1) // BATCH_SIZE + 10) * 4)
    while made < missing and tries < max_tries:
        tries += 1
        want = min(BATCH_SIZE, missing - made)
        try:
            raw = call_api(api_key, prompt_for(kind, want, rng))
            objs = parse_items(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print(f"api_error {type(e).__name__}: {e}")
            time.sleep(SLEEP_SECONDS * 3)
            continue
        rows = []
        for obj in objs:
            row = valid_generated(obj, kind)
            if row and row["split_key"] not in seen:
                seen.add(row["split_key"])
                rows.append(row)
        if rows:
            append_jsonl(path, rows)
            made += len(rows)
            print(f"{kind}: +{len(rows)} new={made} total={len(existing) + made}/{target} file={path}")
        time.sleep(SLEEP_SECONDS)
    print(f"{kind}: already_had={len(existing)} generated={made} target={target}")


SYLLABLES = "ba be bi bo bu da de di do du fa fe fi fo fu ga ge gi go gu ha he hi ho hu ka ke ki ko ku la le li lo lu ma me mi mo mu na ne ni no nu pa pe pi po pu ra re ri ro ru sa se si so su ta te ti to tu va ve vi vo vu wa we wi wo wu ya ye yi yo yu".split()
COMMON_PAYLOADS = "Hello Thanks Goodbye Yes No Maybe Red Blue Green Yellow Black White Apple Water Fire House Chair Table River Mountain Flower Music Small Large Quiet Bright Circle Square Cat Dog Bird".split()
REPEAT_TEMPLATES = ["Repeat after me: {}", "Copy this: {}", "Echo: {}", "Say exactly: {}", "Output exactly: {}"]
IDENTITY_PREFIXES = ["", "One-word answer: ", "No explanation: ", "Answer briefly: ", "Name only: ", "Please answer with your name only: ", "Give the fixed assistant name: ", "Keep it concise: "]
IDENTITY_QUESTIONS = ["What is your name?", "What are you called?", "Can you say your name?", "Please introduce your name.", "State your name.", "Tell me your name.", "Who are you?", "Give me your name.", "Your name?", "What should I call you?", "Identify yourself by name.", "Say the assistant name."]
IDENTITY_WRAPPERS = [
    "{}",
    "User asks: {}",
    "Assistant identity check: {}",
    "Fixed-name task: {}",
    "Respond with the assistant name only: {}",
]


def made_word(rng):
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4)))


def generate_repeat():
    rng = random.Random(SEED + 404 + time.time_ns())
    path = PART_DIR / "repeat.jsonl"
    if RESET_PARTS and path.exists():
        path.unlink()
    existing = []
    seen = set()
    for obj in read_jsonl(path):
        row = valid_local(obj)
        if row and row["task"] == "repeat":
            existing.append(row)
            seen.add(row["split_key"])
    missing = max(0, TARGETS["repeat"] - len(existing))
    if missing == 0:
        print(f"repeat: already_have={len(existing)} target={TARGETS['repeat']}")
        return
    rows = []
    attempts = 0
    while len(rows) < missing and attempts < missing * 20:
        attempts += 1
        if rng.random() < 0.25:
            payload = rng.choice(COMMON_PAYLOADS)
        else:
            payload = " ".join(made_word(rng) for _ in range(rng.randint(1, 5)))
        if rng.random() < 0.10:
            payload = f"{rng.randint(10, 999)} {payload}"
        instruction = rng.choice(REPEAT_TEMPLATES).format(payload)
        row = item("repeat", instruction, payload, "synthetic_repeat_v4")
        if row["split_key"] not in seen:
            seen.add(row["split_key"])
            rows.append(row)
    append_jsonl(path, rows)
    print(f"repeat: +{len(rows)} total={len(existing) + len(rows)}/{TARGETS['repeat']} file={path}")


def generate_identity():
    path = PART_DIR / "identity.jsonl"
    if RESET_PARTS and path.exists():
        path.unlink()
    existing = []
    seen = set()
    for obj in read_jsonl(path):
        row = valid_local(obj)
        if row and row["task"] == "identity":
            existing.append(row)
            seen.add(row["split_key"])
    missing = max(0, TARGETS["identity"] - len(existing))
    if missing == 0:
        print(f"identity: already_have={len(existing)} target={TARGETS['identity']}")
        return
    rows = []
    for wrapper in IDENTITY_WRAPPERS:
        for prefix in IDENTITY_PREFIXES:
            for question in IDENTITY_QUESTIONS:
                row = item("identity", wrapper.format(prefix + question), "Shengoovlei", "synthetic_identity_v4")
                if row["split_key"] not in seen:
                    seen.add(row["split_key"])
                    rows.append(row)
                if len(rows) >= missing:
                    append_jsonl(path, rows)
                    print(f"identity: +{len(rows)} total={len(existing) + len(rows)}/{TARGETS['identity']} file={path}")
                    return
    append_jsonl(path, rows)
    print(f"identity: +{len(rows)} total={len(existing) + len(rows)}/{TARGETS['identity']} file={path}")


def signature(row):
    xs = [w for w in re.findall(r"[a-z]+", norm(row["instruction"])) if len(w) > 3 and w not in STOP]
    return " ".join(xs[:4])


def build():
    rows = []
    for path in sorted(PART_DIR.glob("*.jsonl")):
        for obj in read_jsonl(path):
            row = valid_local(obj)
            if row:
                rows.append(row)
    rng = random.Random(SEED + 999)
    rng.shuffle(rows)
    seen = set()
    sigs = Counter()
    counts = Counter()
    out = []
    for row in rows:
        task = row["task"]
        if counts[task] >= TARGETS.get(task, 0):
            continue
        if row["split_key"] in seen:
            continue
        sig = (task, signature(row))
        if task == "assistant_qa" and sigs[sig] >= 2:
            continue
        if task == "short_qa" and sigs[sig] >= 1:
            continue
        seen.add(row["split_key"])
        sigs[sig] += 1
        counts[task] += 1
        out.append(row)
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows to {FINAL_PATH}")
    print(dict(sorted(counts.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", choices=["1", "2", "3", "4"])
    args = parser.parse_args()

    kind = INDEX_KINDS[args.index]
    if kind in {"assistant_qa", "short_qa", "yesno"}:
        generate_api(kind)
    else:
        generate_repeat()
        generate_identity()
        build()


if __name__ == "__main__":
    main()
