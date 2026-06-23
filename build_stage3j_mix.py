import json
import re
import random

# 输出
OUT = "./data/train_stage3j_mix_106k.jsonl"

# ── 过滤规则（针对无 task 标签的原始数据）──────────────────────────
drop_shengo   = re.compile(r'shengoovlei', re.IGNORECASE)
noisy_yesno   = re.compile(r'^(Yes\.|No\.|YES\.|NO\.)(\s|$)')   # 句号结尾的 Yes./No.
pure_number   = re.compile(r'^-?\d+(\.\d+)?$')                  # 纯数字（数学题答案）

def is_clean(output):
    words = output.split()
    lo = output.lower().strip()
    if drop_shengo.search(output):      return False  # 含 Shengoovlei
    if noisy_yesno.match(output):       return False  # Yes./No. 格式噪声
    if '\n' in output:                  return False  # 多行（列表/代码）
    if pure_number.match(output):       return False  # 纯数字答案
    if len(words) > 60:                 return False  # 太长
    if len(words) == 1 and lo not in ('yes', 'no'):
        return False                                  # 单词非 yes/no
    return True


def load_raw(path):
    """无 task 字段的原始数据，过滤后打上 assistant_qa。"""
    examples = []
    drop = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except: continue
            instr  = str(obj.get('instruction', '')).strip()
            output = str(obj.get('output', '')).strip()
            if not instr or not output: continue
            if not is_clean(output):
                drop += 1
                continue
            examples.append({'instruction': instr, 'output': output, 'task': 'assistant_qa'})
    print(f"  {path}: kept={len(examples)} dropped={drop}")
    return examples


def load_tagged(path):
    """有 task 字段的数据，直接读，不过滤。"""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except: continue
            instr  = str(obj.get('instruction', '')).strip()
            output = str(obj.get('output', '')).strip()
            task   = str(obj.get('task', 'unknown')).strip() or 'unknown'
            if not instr or not output: continue
            examples.append({
                'instruction': instr,
                'output':      output,
                'task':        task,
                'split_key':   str(obj.get('split_key', f'{instr}\t{output}')),
                'source':      str(obj.get('source', 'unknown')),
            })
    print(f"  {path}: loaded={len(examples)}")
    return examples


print("── 原始混合数据（严格过滤）──")
raw1 = load_raw('./data/raw_short_instruct_deepseek_313k.jsonl')
raw2 = load_raw('./data/raw_short_instruct_deepseek_local.jsonl')

print("── 已标注 anchor 数据（直接掺）──")
anchor1 = load_tagged('./data/anchor_mode_boundary_eval.jsonl')   # yesno/repeat/identity
anchor2 = load_tagged('./data/anchor_short_tasks.jsonl')          # repeat/yesno/identity
anchor3 = load_tagged('./data/anchor_assistant_deepseek_v4.jsonl') # assistant_qa/yesno/repeat
anchor4 = load_tagged('./data/anchor_assistant_qa_clean.jsonl')    # assistant_qa

all_examples = raw1 + raw2 + anchor1 + anchor2 + anchor3 + anchor4

# 按 (instruction, output) 去重
seen = set()
deduped = []
for ex in all_examples:
    key = (ex['instruction'], ex['output'])
    if key not in seen:
        seen.add(key)
        deduped.append(ex)

random.seed(42)
random.shuffle(deduped)

with open(OUT, 'w') as f:
    for ex in deduped:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

from collections import Counter
task_dist = Counter(ex['task'] for ex in deduped)
print(f"\n── 输出 ──")
print(f"total: {len(deduped)}  (deduped from {len(all_examples)})")
print(f"task distribution: {dict(task_dist.most_common())}")
print(f"-> {OUT}")
