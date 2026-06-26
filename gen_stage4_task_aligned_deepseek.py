import concurrent.futures
import hashlib
import json
import re
from pathlib import Path

import requests


def main():
    kind = 4
    api_key = json.loads(Path("./pswd.json").read_text(encoding="utf-8"))["deepseek-api-key"]
    url = "https://api.deepseek.com/chat/completions"
    model = "deepseek-chat"
    workers = 8
    per_job = 96
    batch_jobs = 8
    temperature = 1.1
    top_p = 0.95
    out_dir = Path("./data/re_sft_stage4_task_aligned_parts")

    if kind == 1:
        topic = "factoid_short"
        out_path = out_dir / "1_factoid_short.jsonl"
        target = 6000
        prompts = [
            "Create {n} standalone factoid QA items for a tiny assistant SFT dataset. Return only JSON with one key named items. Each item must have instruction and output. The instruction must be a standalone what/which/who/where/when question. The output must be a direct factual answer in 3 to 18 words, at most one sentence. Avoid yes/no questions, math, dates, prices, current events, politics, brands, CEOs, and live-data topics. Favor common science, geography, animals, language, food, tools, weather, and household facts.",
            "Generate {n} clean single-turn factoid questions and answers. Return JSON only: {\"items\": [...]} with instruction and output fields. Questions must begin with what, which, who, where, or when. Answers must be short, direct, one sentence max, and not padded with explanation. Avoid obscure trivia and anything requiring fresh information.",
            "Produce {n} short factual QA pairs in JSON only. Use items with instruction and output. Every instruction must be an independent what/which/who/where/when question. Every output must be concise and factual, 3 to 18 words. No lists, no code, no caveats, no live-data topics.",
            "Return JSON only with {n} standalone factoid QA items. Format: {\"items\":[{\"instruction\":\"...\",\"output\":\"...\"}]}. Questions must be simple factual prompts starting with what/which/who/where/when. Answers must be short and concrete. Avoid math, scheduling, local recommendations, biographies with dates, and creative writing."
        ]
    elif kind == 2:
        topic = "how_why_short"
        out_path = out_dir / "2_how_why_short.jsonl"
        target = 5000
        prompts = [
            "Create {n} standalone how/why assistant QA items. Return only JSON with key items, and each item must have instruction and output. Instructions must start with how or why. Outputs must be one concise helpful sentence, about 8 to 26 words. Avoid medical, legal, financial, crisis, self-harm, or dangerous advice. Avoid saying it depends, consult a professional, or I can't.",
            "Generate {n} single-turn practical how/why questions and short helpful answers. Return JSON only: {\"items\": [...]}. Each instruction must begin with how or why. Each output must be one concrete sentence with a direct explanation or suggestion. No lists, no hedging, no live-data topics.",
            "Produce {n} clean how/why QA pairs for SFT. Return JSON only with instruction and output. Focus on everyday practical topics like cleaning, cooking, study habits, sleep, communication, organizing, travel basics, and home care. Answers must be one short useful sentence.",
            "Return JSON only with {n} how/why items. Questions must be standalone and begin with how or why. Answers must be short, concrete, and helpful, not generic filler. Avoid high-stakes domains, code, creative writing, and multi-step lists."
        ]
    elif kind == 3:
        topic = "slot_following"
        out_path = out_dir / "3_slot_following.jsonl"
        target = 3000
        per_job = 40
        prompts = [
            "Create {n} single-turn slot-following QA items for assistant SFT. Return only JSON with key items, each item having instruction and output. Use only name, city, country, school, hometown, and workplace facts. Examples: My name is Nadia. What is my name? I am from Peru. Where am I from? The output must be only the exact slot value, usually 1 to 3 words, with no full sentence, no explanation, and no ending punctuation. Do not use common repeated values like John, Brown, Paris, London, blue, pizza, teacher, doctor, Tokyo, mango, or Shengoovlei. Use diverse values and do not repeat a value within the batch.",
            "Generate {n} standalone extraction-style QA items. Return JSON only: {\"items\": [...]}. Use only favorite facts such as favorite fruit, favorite animal, favorite book, favorite movie, favorite color, favorite drink, favorite subject, favorite singer, or favorite season. Each instruction must include the fact and then ask for the exact fact. Outputs must be bare slot values only, not sentences. Do not add a period or extra words. Use diverse less-common values and do not repeat values within the batch.",
            "Produce {n} short memory-and-copy QA pairs in JSON only. Use only pet names, car brands, job titles, hobbies, and foods. Every instruction must first state a fact and then ask for the stated fact. Every output must be the exact answer span only, 1 to 3 words, with no article unless it is part of the slot value. Avoid numbers, dates, prices, identity prompts, and repeated common values.",
            "Return JSON only with {n} slot-following items. Format: {\"items\":[{\"instruction\":\"...\",\"output\":\"...\"}]}. Use simple patterns like My favorite fruit is papaya. What is my favorite fruit? Keep outputs exact, minimal, and never phrased as sentences. Use varied values and avoid reusing the same names, cities, jobs, foods, or colors."
        ]
    elif kind == 4:
        topic = "short_form_answer"
        out_path = out_dir / "4_short_form_answer.jsonl"
        target = 3000
        per_job = 40
        prompts = [
            "Create {n} one-word-only QA items. Return only JSON with key items, each item having instruction and output. Every instruction must explicitly say one word only. Every output must be exactly one plain word with no punctuation. Use everyday topics like opposites, synonyms, object categories, materials, tastes, colors, weather words, and animal classes. Do not explain the answer.",
            "Generate {n} short-phrase-only QA items in JSON only: {\"items\": [...]}. Every instruction must explicitly say short phrase only. Every output must be 1 to 3 words, not a sentence, and not a definition. Use noun phrases and adjective phrases such as red fruit, cold weather, metal tool, spicy food, or flying animal. Never write The answer is, A good example, or This means.",
            "Produce {n} one-short-sentence-only QA items for SFT. Return JSON only with instruction and output. Every instruction must explicitly say one short sentence only. Every output must be one concrete sentence of 4 to 10 words. Use simple everyday topics like dogs, rain, cooking, books, plants, and rooms. Avoid creative writing and generic filler.",
            "Return JSON only with {n} short-form items. Mix one word only, short phrase only, and one short sentence only, but make every item explicit about the required form. Keep outputs exact, short, and non-explanatory."
        ]
    else:
        raise SystemExit("kind must be 1, 2, 3, or 4")

    rows = []
    seen = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            row = json.loads(line)
            if row["split_key"] in seen:
                continue
            seen.add(row["split_key"])
            rows.append(row)
    submitted = 0
    flat_rounds = 0

    def one_job(i):
        text = ""
        try:
            r = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You create compact high-quality SFT data. Return JSON only. No markdown fences. No extra text."
                        },
                        {
                            "role": "user",
                            "content": (prompts[0] if kind in (1, 2) else prompts[i % len(prompts)]).replace("{n}", str(per_job))
                        },
                    ],
                },
                timeout=180,
            )
            text = r.json()["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```json\s*|^```\s*|```$", "", text.strip(), flags=re.M).strip()
            obj = json.loads(text)
            items = obj.get("items", [])
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                return []
        except Exception:
            return []

        part = []
        for item in items:
            if not isinstance(item, dict):
                continue
            instruction = " ".join(str(item.get("instruction", "")).strip().split())
            output = " ".join(str(item.get("output", "")).strip().split())
            if not instruction or not output:
                continue
            if "\n" in instruction or "\n" in output:
                continue
            if len(instruction.split()) < 3 or len(instruction.split()) > 40:
                continue
            if output.endswith("?"):
                continue
            if "shengoovlei" in instruction.lower() or "shengoovlei" in output.lower():
                continue
            if "i don't know" in output.lower() or "i cant" in output.lower() or "i can't" in output.lower():
                continue

            if kind == 1:
                if not instruction.lower().startswith(("what ", "which ", "who ", "where ", "when ")):
                    continue
                if len(output.split()) < 2 or len(output.split()) > 18:
                    continue
                if output.lower().startswith(("yes", "no")):
                    continue
            elif kind == 2:
                if not instruction.lower().startswith(("how ", "why ")):
                    continue
                if len(output.split()) < 6 or len(output.split()) > 28:
                    continue
                if output.lower().startswith(("yes", "no")):
                    continue
            elif kind == 3:
                if not any(x in instruction.lower() for x in [
                    " my name is ", " her name is ", " his name is ", " i live in ", " i am from ",
                    " my favorite ", " i work at ", " my dog is named ", " my cat is named ",
                    " my hometown is ", " my school is ", " my job is ", " my car is ", " my city is "
                ]) and not instruction.lower().startswith((
                    "my name is ", "her name is ", "his name is ", "i live in ", "i am from ",
                    "my favorite ", "i work at ", "my dog is named ", "my cat is named ",
                    "my hometown is ", "my school is ", "my job is ", "my car is ", "my city is "
                )):
                    continue
                if len(output.split()) < 1 or len(output.split()) > 4:
                    continue
                if output.lower().startswith(("yes", "no")):
                    continue
                if re.search(r"[.!?,:;\"()]", output):
                    continue
                if re.search(r"\b(is|are|was|were|means|called|named|lives|works|stands)\b", output.lower()):
                    continue
            elif kind == 4:
                if not any(x in instruction.lower() for x in [
                    "one word only", "short phrase only", "one short sentence only"
                ]):
                    continue
                if any(x in output.lower() for x in [
                    "the word", "the phrase", "a good example", "an example", "the answer is",
                    "this means", "refers to", "is a word", "is a phrase", "for example"
                ]):
                    continue
                if "one word only" in instruction.lower():
                    if len(output.split()) != 1:
                        continue
                    if re.search(r"[.!?,:;\"()]", output):
                        continue
                if "short phrase only" in instruction.lower():
                    if not (1 <= len(output.split()) <= 4):
                        continue
                    if re.search(r"[.!?,:;\"()]", output):
                        continue
                    if re.search(r"\b(is|are|was|were|means|called|named|lives|works|stands|can|will|should)\b", output.lower()):
                        continue
                if "one short sentence only" in instruction.lower():
                    if not (4 <= len(output.split()) <= 10):
                        continue
                    if output.count("?") or output.count("!"):
                        continue

            split_key = hashlib.sha1(f"assistant_qa:{topic}:{instruction.lower()}".encode("utf-8")).hexdigest()
            part.append(
                {
                    "instruction": instruction,
                    "output": output,
                    "task": "assistant_qa",
                    "source": f"deepseek_stage4_{topic}",
                    "topic": topic,
                    "split_key": split_key,
                }
            )
        return part

    while len(rows) < target and flat_rounds < 12:
        before = len(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for part in ex.map(one_job, range(submitted, submitted + batch_jobs)):
                for row in part:
                    if row["split_key"] in seen:
                        continue
                    seen.add(row["split_key"])
                    rows.append(row)
                if len(rows) >= target:
                    break
        submitted += batch_jobs
        gained = len(rows) - before
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("round_done", submitted, len(rows), gained, flush=True)
        if gained < max(8, per_job // 2):
            flat_rounds += 1
        else:
            flat_rounds = 0

    rows = rows[:target]
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(out_path, len(rows), flush=True)
    if rows:
        print(json.dumps(rows[0], ensure_ascii=False), flush=True)
        print(json.dumps(rows[min(1, len(rows) - 1)], ensure_ascii=False), flush=True)

    if kind == 4:
        merged_path = Path("./data/re_sft_stage4_task_aligned_deepseek_all.jsonl")
        merged = []
        merged_seen = set()
        for name in [
            "1_factoid_short.jsonl",
            "2_how_why_short.jsonl",
            "3_slot_following.jsonl",
            "4_short_form_answer.jsonl",
        ]:
            path = out_dir / name
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if row["split_key"] in merged_seen:
                    continue
                merged_seen.add(row["split_key"])
                merged.append(row)
        with merged_path.open("w", encoding="utf-8") as f:
            for row in merged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(merged_path, len(merged), flush=True)


if __name__ == "__main__":
    main()
