import argparse
import json
import os
import time
import urllib.error
import urllib.request


API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


def load_api_key(path):
    with open(path, encoding="utf-8") as file:
        obj = json.load(file)
    for key in ("deepseek-api-key", "deepseek_api_key", "DEEPSEEK_API_KEY"):
        if obj.get(key):
            return obj[key]
    raise RuntimeError(
        f"missing DeepSeek API key in {path}. Add one key named deepseek-api-key."
    )


def build_prompt(n, batch_id):
    return f"""Generate {n} single-turn instruction tuning examples for a small chat assistant.

Output only raw JSONL. Each line must be one valid JSON object with exactly these keys: "instruction" and "output".
Do not use markdown fences, code blocks, or any other formatting. Do not include any explanation.

The assistant's name is Shengoovlei. It must consistently use this identity when relevant.

CRITICAL DESIGN GOAL: DATA MUST BE BALANCED ACROSS 6 SKILL DOMAINS

You MUST generate a dataset covering the following categories evenly:

1. Arithmetic (16.7%)
Simple addition, subtraction, multiplication, division (0–100 range)
Format examples:
"What is 12+7?" → "19"
"8*6=?" → "48"
"100-37?" → "63"
2. Copy / Repeat (16.7%)
Exact string reproduction behavior
Must generalize beyond fixed words
Format:
"Repeat: apple" → "apple"
"Say: 48291" → "48291"
"Copy this: hello world" → "hello world"
3. Yes / No Logic (16.7%)
Binary reasoning questions
Include slight variation in phrasing
Format:
"Is water wet?" → "Yes."
"Is fire cold?" → "No."
"Can birds fly?" → "Yes."
4. Refusal / Uncertainty (16.7%)
Honest unknown-answer behavior
Must avoid hallucination
Format:
"Who will win tomorrow's game?" → "I don't know."
"What is inside my pocket?" → "I don't know."
"Predict the lottery number" → "I don't know."
5. Greeting / Social (16.7%)
Short conversational responses
Identity handling included
Format:
"Hello" → "Hello! How can I help?"
"How are you?" → "I'm fine, thanks."
"What is your name?" → "My name is Shengoovlei."
6. Identity / Instruction Following (16.7%)
Name, role, and direct instruction parsing
Must generalize across paraphrases
Format:
"Who are you?" → "I'm Shengoovlei."
"Tell me your name" → "My name is Shengoovlei."
"Say the word: sunshine" → "sunshine"

GLOBAL OUTPUT RULES

70% of outputs must be ≤10 words
20% may be 2–3 sentences
10% must be single-word or yes/no answers
Keep responses extremely concise and deterministic
Avoid explanations, reasoning, or extra text in outputs

STRICT PROHIBITIONS

Do NOT include explanations in output
Do NOT use refusal phrases like "As an AI", "I cannot", etc.
Do NOT generate long paragraphs
Do NOT deviate from instruction-output format
Do NOT leak formatting instructions into dataset

DIVERSITY REQUIREMENT

Each batch must vary:
wording of instructions
numerical ranges
entity names
phrasing of questions
Avoid repeated templates across samples

Diversity seed: batch-{batch_id}"""


def call_deepseek(api_key, prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_jsonl(text):
    good = []
    bad = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if not line.startswith("{"):
            bad += 1
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        instruction = str(obj.get("instruction", "")).strip()
        output = str(obj.get("output", "")).strip()
        if instruction and output:
            good.append({"instruction": instruction, "output": output})
        else:
            bad += 1
    return good, bad


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_valid_rows(path):
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("instruction", "")).strip() and str(obj.get("output", "")).strip():
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-json", default="./pswd.json")
    parser.add_argument("--out", default="./data/simple_logits_deepseek.jsonl")
    parser.add_argument("--target-total", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--per-batch", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=10000)
    args = parser.parse_args()

    api_key = load_api_key(args.key_json)
    existing = count_valid_rows(args.out)
    total = existing
    next_chunk_mark = ((total // args.chunk_size) + 1) * args.chunk_size
    print(f"existing valid rows: {existing}; target total: {args.target_total}")

    batch_id = 0
    while total < args.target_total:
        batch_id += 1
        need = min(args.per_batch, args.target_total - total)
        prompt = build_prompt(need, batch_id)
        for attempt in range(3):
            try:
                text = call_deepseek(api_key, prompt, args.max_tokens)
                rows, bad = parse_jsonl(text)
                append_jsonl(args.out, rows)
                total += len(rows)
                print(
                    f"batch {batch_id}: saved {len(rows)} rows, "
                    f"bad_lines {bad}, total {total}"
                )
                while total >= next_chunk_mark:
                    print(f"chunk complete: {next_chunk_mark}/{args.target_total}")
                    next_chunk_mark += args.chunk_size
                break
            except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError) as exc:
                if attempt == 2:
                    raise
                print(f"batch {batch_id}: retry after error: {exc}")
                time.sleep(3 * (attempt + 1))

    print(f"done: {total} valid rows in {args.out}")


if __name__ == "__main__":
    main()
