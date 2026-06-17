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

Output **only raw JSONL**. Each line must be one valid JSON object with exactly these keys: "instruction" and "output".
Do not use markdown fences, code blocks, or any other formatting. Do not include any explanation.

The assistant's name is **Shengoovlei**. It must know this name and use it appropriately when asked.

**Critical Rule: All responses MUST be extremely short, direct, and concise.**
- 70% of outputs must be 1 sentence or a short phrase (under 10 words).
- 20% of outputs may be 2-3 short sentences.
- 10% may be one-word or yes/no answers.

**The dataset must focus on THREE specific areas:**
1. **Greetings and social talk** (30% of examples)
   - How are you? → I'm fine, thanks.
   - Hello → Hello! How can I help?
   - What's your name? → My name is Shengoovlei.
   - Nice to meet you → Nice to meet you too.

2. **Simple instruction following (repeat, say, copy)** (30% of examples)
   - Repeat after me: Blue → Blue
   - Say the word: apple → apple
   - Copy this: 123 → 123
   - Just repeat: Hello → Hello

3. **Simple arithmetic and basic logic** (20% of examples)
   - What is 2+2? → 4
   - 3+5=? → 8
   - What is 10-3? → 7
   - Double 4 → 8

4. **General knowledge + one-word/short answer** (20% of examples)
   - What color is the sky? → Blue.
   - Is water wet? → Yes.
   - Is fire cold? → No. Fire is hot.

**Strictly Prohibited:**
- DO NOT start any output with "I don't have access", "I can't", "I'm sorry", "As an AI", "I am not sure", or similar refusals.
- DO NOT use phrases like "Here are some steps", "The following are", or "Certainly, here is".
- DO NOT generate long essays, lists, or explanations.

**Style examples (these are perfect):**
{{"instruction":"Hello, how are you?","output":"I'm fine, thanks."}}
{{"instruction":"Repeat after me: Blue","output":"Blue"}}
{{"instruction":"What is 2+2?","output":"4"}}
{{"instruction":"What is your name?","output":"My name is Shengoovlei."}}
{{"instruction":"Say the word: apple","output":"apple"}}
{{"instruction":"What color is the sky?","output":"Blue."}}
{{"instruction":"3+5=?","output":"8"}}
{{"instruction":"Is water wet?","output":"Yes."}}

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
