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

Output JSONL only. Each line must be one valid JSON object with exactly these keys: "instruction" and "output".
Do not use markdown fences. Do not put two JSON objects on one line. Do not include any explanation.

Requirements:
- Responses must be short, direct, and natural.
- 60% of outputs should be 1 sentence.
- 30% of outputs should be 2-3 sentences.
- 10% may be one-word or yes/no answers.
- Avoid long essays, roleplay, story continuation, code debugging, chapter outlines, and repeated examples.
- Avoid labels like "repeat" or "example".

Include a balanced mix:
- greetings and small talk
- yes/no questions
- one-word answers
- simple factual QA
- practical preparation advice
- answer-format following
- simple math
- classification
- brief rewrite
- one-sentence summary
- uncertainty for unknown/live data
- polite refusal for impossible requests

Style examples:
{{"instruction":"Is fire cold?","output":"No. Fire is hot."}}
{{"instruction":"Hello, how are you today?","output":"I'm doing well. How can I help you today?"}}
{{"instruction":"What should I prepare for a climbing tour?","output":"Bring sturdy shoes, water, snacks, layered clothing, sun protection, a first-aid kit, and a map or GPS."}}
{{"instruction":"Answer with one word: What color is the sky on a clear day?","output":"Blue"}}

Diversity seed: batch-{batch_id}
"""


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
    parser.add_argument("--out", default="./data/short_instruct_deepseek.jsonl")
    parser.add_argument("--target-total", type=int, default=30000)
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
