# 错误与修正时间线

## 1. 先怀疑是“生成塌缩到特殊 token”
**现象**
- 生成时大量偏向 `\n` / `space` / `<|endoftext|>`。
- teacher forcing 正常，但 free generation 很差。

**最初判断**
- 以为是 generation dynamics collapse，或 SFT 学歪了起始分布。

## 2. 排除了“loss 对齐错位”这个猜测
**检查**
- 读了 `modules.py` 里的 `run_cross_entropy_for_gem()`。

**结果**
- 它内部已经做了 shift：
  - `logits[:, :-1, :]`
  - `labels[:, 1:]`

**修正**
- 说明 HiRA/SFT 分支不是“标签整体错一位”的 bug。
- 但 LoRA bench 分支原来用了未 shift 的 `ce_loss`，这是另一个问题。

## 3. 发现 vocab 文件本身不干净
**检查**
- `owt_train_32004.pickle` 不是连续 vocab：
  - `<|endoftext|>` 有两个 id：`31999` 和 `32000`
  - `32002` 是空洞 id
- 所以：
  - `len(vocab) = 32004`
  - `max_id + 1 = 32005`

**错误**
- 不能直接拿 `max_id + 1` 当真实 vocab size。
- 也不能把这个坏 vocab 直接当训练/推理唯一依据。

**修正**
- 在 notebook / 训练脚本里重建连续 vocab：
  - 保留 base vocab `0..31999`
  - 按训练顺序追加 special tokens
- HiRA 修正后 vocab 变成 `32003`：
  - `<|endoftext|>` -> `31999`
  - `<|user|>` -> `32000`
  - `<|assistant|>` -> `32001`
  - `<|pad|>` -> `32002`

## 4. 训练脚本里有硬编码 id
**错误**
- eval 里还写死了 `32004` 去看 `assistant` embedding norm。
- vocab 修正后这会越界，因为 `assistant` 已经不是 `32004`。

**修正**
- 改成动态读取：
  - `assistant_token_id = tk.vocab_inv[b"<|assistant|>"]`
- 同时把梯度 hook 里的 `32000` 改成变量 `old_vocab`，避免语义混乱。

## 5. 发现真正影响生成的边界问题：prompt 末尾空格
**错误**
- 原模板是：
  - `"<|user|> ... <|assistant|> "`
- 这让最后一个上下文 token 是裸空格。

**验证**
- 带空格时，top-1 常常是 `<|endoftext|>`，top-2 是 `\n`。
- 去掉空格后，top-1 变成 `\n`，说明模型学到的是“assistant 后面要换行”，不是“assistant 后面接空格”。

**修正**
- 把模板改成：
  - `"<|user|> ... <|assistant|>\n"`
- 训练和推理必须完全一致。

## 6. 生成开始恢复正常
**结果**
- 用新模板后，prompt：
  - `<|user|> hello， how are you today? <|assistant|>\n`
- 生成结果开始变成正常英文句子，而不是特殊 token 风暴。

**结论**
- 问题已经从“模型塌缩”变成“模板边界不一致”。
- 现在模型至少能稳定进入自然语言分布。

## 7. LoRA 分支也顺手修了
**错误**
- LoRA bench 原来：
  - `vocab_size = 32009` 硬编码
  - `ce_loss` 没做 shift

**修正**
- 改成和 HiRA 一样：
  - vocab 动态构建
  - `vocab_size = max_id + 1`
  - loss 用 `run_cross_entropy_for_gem()`

## 最终结论
### 主要错误顺序
1. 先误判为“生成动力学塌缩”
2. 再排除 loss shift bug
3. 发现坏 vocab 文件
4. 修正 vocab/id 映射
5. 修正硬编码 id
6. 最后定位到 prompt 末尾空格导致的边界偏置
7. 改成 `"<|assistant|>\n"` 后恢复正常

### 现在的正确做法
- 训练和推理统一用：
  - `<|user|> ... <|assistant|>\n`
- special token id 从 tokenizer 动态取
- 不再使用坏的原始 vocab 顺序
