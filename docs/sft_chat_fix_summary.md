# SFT Chat 修复重点

## 1. 固定 chat vocab 方案

SFT 现在统一使用 4 个 special token：

```text
31999 <|endoftext|>
32000 <|user|>
32001 <|assistant|>
32002 <|pad|>
```

对应：

```text
vocab_size = 32003
```

代码位置：

```text
sft_chat_templetes.py
CHAT_SPECIAL_TOKENS
load_sft_tokenizer()
```

## 2. 不再手动扩容 checkpoint

旧逻辑会在 SFT 里把 checkpoint 的 embedding/lm_head 从旧 vocab size 手动 resize 到新 vocab size。

现在已删除该逻辑。

新逻辑是：

```text
checkpoint vocab_size 必须等于 tokenizer vocab_size
```

否则直接报错：

```text
SFT no longer resizes model weights.
Pretrain or continue-pretrain with the same chat vocab before running this script.
```

这意味着当前本地 `vocab_size=32000` 的 checkpoint 不能在无扩容方案下直接 SFT。
需要使用从一开始就是 `vocab_size=32003` 的预训练或继续预训练 checkpoint。

如果必须保留已有 `vocab_size=32000` 权重，不要在 SFT 训练入口里临时扩容。
使用一次性迁移脚本生成 canonical `vocab_size=32003` checkpoint：

```bash
.venv/bin/python migrate_checkpoint_to_chat_vocab.py \
  --src ./data/pretrain_gpt2med_iter_390000.pt \
  --out ./data/pretrain_gpt2med_iter_390000_chatvocab32003.pt \
  --drop-optim
```

迁移后运行 SFT 时指定：

```bash
PRETRAIN_CKPT_PATH=./data/pretrain_gpt2med_iter_390000_chatvocab32003.pt \
  .venv/bin/python sft_chat_templetes.py
```

这个迁移只做一次。
之后 SFT 脚本仍然要求 checkpoint vocab size 与 tokenizer vocab size 完全一致。

## 3. 修复 SFT label 起点错位

旧 batch 构造问题：

```text
x_text_part = "<|user|> ... <|assistant|> "
x_text_full = x_text_part + output + " <|endoftext|>"
```

因为 `x_text_part` 单独 encode 时尾部空格是独立 token，但完整文本 encode 时这个空格可能和回答第一个词合并成一个 BPE token，导致：

```text
第一个 assistant 输出 token 被 mask 掉
labels 从第二个输出 token 才开始
```

新逻辑：

```text
prompt_tokens = encode(prompt)
output_tokens = encode(output)
full_tokens = prompt_tokens + output_tokens + [eos]
labels 从 len(prompt_tokens) 开始
```

这样不会再受 prompt/output 边界 BPE 合并影响。

## 4. 固定训练和推理模板

训练模板：

```python
prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
full = prompt + output + "<|endoftext|>"
```

推理时也必须使用同样的 prompt 格式：

```python
prompt = "<|user|>\nhow are you today?\n<|assistant|>\n"
```

不要使用空格版：

```python
"<|user|> how are you today? <|assistant|> "
```

推理 prompt 中不要加入 `<|endoftext|>`。
只把 `<|endoftext|>` 的 id `31999` 作为停止 token。

## 5. 修复 HiRA/LoRA wrapper 兼容性

本地 `modules.Linear` 的权重字段是：

```text
W
```

不是 PyTorch 标准 `weight`。

因此 `HiRALinear` / `LoRALinear` 现在通过 `get_linear_weight_and_bias()` 同时兼容：

```text
modules.Linear.W
torch.nn.Linear.weight
```

## 6. 已验证

已运行：

```bash
.venv/bin/python -m py_compile sft_chat_templetes.py
```

batch 对齐检查结果：

```text
<|assistant|>
\n
第一个输出 token
```

第一个被监督的目标现在是第一个输出 token。

HiRA/LoRA 小模型替换 forward 通过。

当前本地 checkpoint 检查结果：

```text
checkpoint vocab_size=32000
SFT tokenizer vocab_size=32003
```

所以新版 SFT 会拒绝当前 checkpoint，这是预期行为。
