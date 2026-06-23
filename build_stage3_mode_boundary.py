import json
import random
from collections import Counter


def main():
    rows = []
    seen = set()
    for line in open("data/re_sft_assistant_deepseek_v4.jsonl", encoding="utf-8"):
        obj = json.loads(line)
        task = obj.get("task", "")
        instruction = str(obj.get("instruction", "")).strip()
        output = str(obj.get("output", "")).strip()
        if not instruction or not output:
            continue
        if task == "assistant_qa":
            instruction = f"Answer in one concise helpful sentence: {instruction}"
        elif task == "yesno":
            instruction = f"Answer yes or no only: {instruction}"
        elif task != "identity":
            continue
        key = f"{task}:mode:{obj.get('split_key') or instruction}"
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "instruction": instruction,
            "output": output,
            "task": task,
            "source": f"{obj.get('source', 'deepseek_generated_v4')}_mode_boundary",
            "split_key": key,
        })

    repeat_seen = set()
    for line in open("data/_re_sft_short_tasks.jsonl", encoding="utf-8"):
        obj = json.loads(line)
        if obj.get("task") != "repeat":
            continue
        instruction = str(obj.get("instruction", "")).strip()
        output = str(obj.get("output", "")).strip()
        if not instruction or not output or len(output) > 80 or len(output.split()) > 5:
            continue
        if (instruction, output) in repeat_seen:
            continue
        repeat_seen.add((instruction, output))
        rows.append({
            "instruction": instruction,
            "output": output,
            "task": "repeat",
            "source": obj.get("source", "stage2_repeat_replay"),
            "split_key": obj.get("split_key") or f"repeat:{output.lower()}",
        })

    random.Random(1339).shuffle(rows)
    with open("data/re_sft_assistant_stage3_mode_boundary.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(dict(sorted(Counter(row["task"] for row in rows).items())))


if __name__ == "__main__":
    main()
