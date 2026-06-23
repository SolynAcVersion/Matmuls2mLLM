import json
import random
from collections import Counter


def main():
    rows = []
    seen = set()
    for line in open("data/re_sft_assistant_deepseek_v4.jsonl", encoding="utf-8"):
        obj = json.loads(line)
        task = obj.get("task", "")
        if task not in {"assistant_qa", "yesno", "identity"}:
            continue
        instruction = str(obj.get("instruction", "")).strip()
        output = str(obj.get("output", "")).strip()
        if not instruction or not output:
            continue
        key = obj.get("split_key") or f"{task}:{instruction}\t{output}"
        if (task, key) in seen:
            continue
        seen.add((task, key))
        rows.append({
            "instruction": instruction,
            "output": output,
            "task": task,
            "source": obj.get("source", "deepseek_generated_v4"),
            "split_key": key,
        })

    repeats = []
    repeat_seen = set()
    for line in open("data/_re_sft_short_tasks.jsonl", encoding="utf-8"):
        obj = json.loads(line)
        if obj.get("task") != "repeat":
            continue
        instruction = str(obj.get("instruction", "")).strip()
        output = str(obj.get("output", "")).strip()
        if not instruction or not output or len(output) > 80 or len(output.split()) > 5:
            continue
        key = obj.get("split_key") or f"repeat:{output.lower()}"
        if (instruction, output) in repeat_seen:
            continue
        repeat_seen.add((instruction, output))
        repeats.append({
            "instruction": instruction,
            "output": output,
            "task": "repeat",
            "source": obj.get("source", "stage2_repeat_replay"),
            "split_key": key,
        })

    random.Random(1337).shuffle(repeats)
    rows.extend(repeats)
    random.Random(1338).shuffle(rows)
    with open("data/re_sft_assistant_stage3_anchor_mix.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(dict(sorted(Counter(row["task"] for row in rows).items())))


if __name__ == "__main__":
    main()
