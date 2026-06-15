# sft_chat_for_sharegpt_hira.py 对话总结

日期：2026-06-15

本文记录本次围绕 `sft_chat_for_sharegpt_hira.py` 的完整对话和代码调整过程。重点包括：脚本恢复、代码结构变化、预训练 checkpoint 词表兼容、数据加载验证，以及最后按要求改成硬编码直跑版。

## 1. 初始状态

对话开始时，`sft_chat_for_sharegpt_hira.py` 已经处于删除状态：

```text
 D sft_chat_for_sharegpt_hira.py
```

仓库当前目录是：

```text
/home/yldeveloper/learning/CS336/ai
```

当时先检查了仓库文件，确认还存在这些相关文件：

```text
modules.py
sft_chat_templetes.py
data/train.jsonl
data/short_instruct_deepseek.jsonl
data/pretrain_gpt2med_iter_390000.pt
```

因为目标脚本已经被删除，所以先通过：

```bash
git show HEAD:sft_chat_for_sharegpt_hira.py
```

读取了 Git 中的旧版本内容，作为恢复和重写的依据。

## 2. 旧脚本的主要功能

旧版 `sft_chat_for_sharegpt_hira.py` 主要做以下事情：

1. 从 `./data/train.jsonl` 读取 ShareGPT 风格数据。
2. 从 `./data/short_instruct_deepseek.jsonl` 读取短指令数据。
3. 将 ShareGPT 多轮 conversation 拆成单轮 `{instruction, output}` 样本。
4. 使用 `sft_chat_templetes.py` 中的 chat tokenizer 和 `encode_chat_example()`。
5. 构建 `modules.TransformerLM`：

```text
vocab_size = tokenizer vocab size
context_length = 1024
d_model = 1024
num_layers = 24
num_heads = 16
d_ff = 2752
rope_theta = 10000
```

6. 将部分线性层替换成 `HiRALinear`：

```text
attention q_proj
attention k_proj
attention v_proj
ffn w2
ffn w3
```

7. 冻结大部分模型参数，只训练：

```text
HiRA A/B 参数
新增 chat special token 的 embedding 行
新增 chat special token 的 lm_head 列
```

8. 对 embedding 和 lm_head 注册梯度 hook，让 base vocab 前 32000 个 token 的梯度为 0。
9. 使用带第一枚 assistant 输出 token 加权的 SFT loss。
10. 训练时混合 ShareGPT 样本和短指令样本，短指令比例原来是 `0.70`。
11. 定期 eval、打印日志、保存 checkpoint。
12. 旧版还使用 wandb 记录训练过程。

## 3. 第一次重建：参数化版本

第一次重建时，先做了一个比较工程化的版本。这个版本加入了：

```text
argparse
dataclass ModelConfig
dataclass TrainConfig
可配置路径
可配置 batch / lr / interval / wandb
可配置 resume checkpoint
```

当时这样做的原因是：

1. 旧脚本大量硬编码，难以在不同 checkpoint 和数据路径之间切换。
2. 需要明确区分从 pretrain checkpoint 初始化和从 SFT checkpoint resume。
3. 需要把 wandb 变成可选项，避免默认导入或默认联网。
4. 需要让 checkpoint、output、eval interval 等行为更清晰。

这个版本中保留或新增了以下关键函数：

```text
load_sharegpt_pairs()
load_instruction_jsonl()
split_examples()
make_batch()
sft_loss()
build_model()
add_hira_adapters()
freeze_for_hira()
load_model_weights()
load_wandb_key()
parse_args()
```

## 4. 发现并修正的关键技术问题

### 4.1 初始化顺序问题

一开始参数化版本里曾经先替换 HiRA 层，再加载 pretrain checkpoint。这样会有一个严重问题：

```text
pretrain checkpoint 里是普通线性层权重
HiRA 层里需要的是冻结基座 W_0
如果先替换 HiRA，再用 strict=False 加载 pretrain，很容易导致 W_0 没有正确来自预训练权重
```

因此后来修正为：

```text
构建 base TransformerLM
先加载 pretrain checkpoint
再把目标线性层替换成 HiRALinear
再冻结参数并启用 HiRA A/B 训练
```

这保证了 HiRA 的 `W_0` 来自真实预训练权重，而不是随机初始化权重。

### 4.2 checkpoint vocab size 不一致

检查 `data/pretrain_gpt2med_iter_390000.pt` 后确认：

```text
token_embeddings.embedding_weights: (32000, 1024)
lm_head.W: (1024, 32000)
```

而 `sft_chat_templetes.py` 中的 chat tokenizer 实际 vocab size 是：

```text
32003
```

原因是 base vocab 前 32000 个 token 中已经包含 `<|endoftext|>`，所以只额外新增：

```text
<|user|>
<|assistant|>
<|pad|>
```

因此 SFT 模型需要：

```text
token_embeddings.embedding_weights: (32003, 1024)
lm_head.W: (1024, 32003)
```

`torch.nn.Module.load_state_dict(strict=False)` 不能自动忽略同名参数的 shape mismatch，所以如果直接加载，会失败。

为此加入了专门的 pretrain 权重加载逻辑：

```text
普通参数：shape 必须完全一致
token_embeddings.embedding_weights：复制 checkpoint 中前 32000 行到新模型
lm_head.W：复制 checkpoint 中前 32000 列到新模型
新增的 3 个 chat token 参数保留模型初始化值
其他 shape mismatch 直接报错
```

实际验证输出为：

```text
resized token_embeddings.embedding_weights: (32000, 1024) -> (32003, 1024)
resized lm_head.W: (1024, 32000) -> (1024, 32003)
```

### 4.3 本地 Python 环境问题

系统 Python 没有安装 torch，直接运行时出现：

```text
ModuleNotFoundError: No module named 'torch'
```

后来确认仓库使用 `.venv`：

```text
.venv -> /home/yldeveloper/learning/CS336/assignment1-basics-main/.venv
```

之后所有验证命令都改用：

```bash
./.venv/bin/python
```

该环境中的 PyTorch 版本是：

```text
2.11.0+cu130
```

### 4.4 短指令 JSONL 中存在非严格 JSONL 行

读取 `./data/short_instruct_deepseek.jsonl` 时发现某一行不是严格的一行一个 JSON object：

```text
JSONDecodeError: Extra data
```

旧脚本对坏行比较宽松，而第一次新写的 loader 过于严格，导致直接报错。

后来改成使用：

```python
json.JSONDecoder().raw_decode(...)
```

支持从同一行中连续解析多个 JSON object。实在无法解析的行才跳过，并打印汇总提示。

最终数据加载验证结果为：

```text
sharegpt 391655
sharegpt split [352490, 39165]
short 31602
short split [28442, 3160]
```

## 5. 用户后续要求：不要可配置

在完成参数化版本之后，用户明确提出：

```text
不要可配置！直接写死进去，不要在最开始设置一堆变量，字符串直接硬编码进入函数，最小化代码块，减少不必要的函数直接写入应用的地方。
```

因此后续又把脚本重构为“直接运行版”。

这次重构删除了：

```text
argparse
dataclass ModelConfig
dataclass TrainConfig
wandb 支持
resume checkpoint 支持
build_model() 包装函数
add_hira_adapters() 包装函数
freeze_for_hira() 包装函数
load_wandb_key()
format_count()
```

训练路径、checkpoint 路径、保存路径和超参数都直接写入使用位置。

例如：

```python
sharegpt_train, sharegpt_val = split_examples(load_sharegpt_pairs("./data/train.jsonl"), 1337)
```

预训练 checkpoint 路径直接写在 `load_pretrain_weights()` 里：

```python
obj = torch.load("./data/pretrain_gpt2med_iter_390000.pt", map_location="cpu")
```

最终权重保存路径直接写在 `finally` 里：

```python
torch.save({"model": model.state_dict(), "it": last_iter}, "./data/sharegpt_only_hira_final.pt")
```

## 6. 当前最终脚本结构

当前 `sft_chat_for_sharegpt_hira.py` 只保留这些顶层函数：

```text
iter_jsonl_objects(path)
load_sharegpt_pairs(path)
load_instruction_jsonl(path)
split_examples(examples, seed)
load_pretrain_weights(model)
make_batch(examples, tokenizer, device, short_examples=None)
sft_loss(logits, labels)
main()
```

其中真正训练流程集中在 `main()` 中，主要顺序是：

```text
设置随机种子
加载 tokenizer
加载 ShareGPT 数据
加载 short instruct 数据
构建 TransformerLM
加载 32000 vocab pretrain checkpoint，并 resize 到 32003 vocab
替换 HiRA 层
冻结 base 参数
只开放 HiRA A/B 和新增 token embedding/lm_head 参数
创建 AdamW optimizer
定期训练、打印、eval、保存 checkpoint
finally 中保存最终权重
```

当前写死的训练参数包括：

```text
context_length = 1024
batch_size = 8
max_iters = 10000
eval_interval = 250
checkpoint_interval = 500
log_interval = 50
max_learning_rate = 3e-5
min_learning_rate = 5e-6
warmup_iters = 200
max_grad_norm = 0.5
first assistant token weight = 5.0
short instruct ratio = 0.70
HiRA r = 16
HiRA alpha = 8
```

当前写死的路径包括：

```text
./data/train.jsonl
./data/short_instruct_deepseek.jsonl
./data/pretrain_gpt2med_iter_390000.pt
./checkpoints/sharegpt_hira_iter_{it}.pt
./data/sharegpt_only_hira_final.pt
```

## 7. 当前训练逻辑细节

### 7.1 Batch 构造

`make_batch()` 固定生成：

```text
x shape = (8, 1024)
y shape = (8, 1024)
```

`x` 使用 `<|pad|>` token 填充。

`y` 使用 `-666` 作为 ignore index。

每条样本的 label 逻辑是：

```text
prompt 部分 label = -666
assistant output + <|endoftext|> 部分参与监督
```

训练时有 70% 概率从 short instruct 数据池采样，否则从 ShareGPT 数据池采样。

### 7.2 Loss

当前 `sft_loss()` 固定做 causal shift：

```text
logits = logits[:, :-1, :]
labels = labels[:, 1:]
```

并对每条样本第一枚被监督的 assistant token 加权：

```text
weight = 5.0
```

这延续了前面针对“自由生成第一步容易退回预训练续写模式”的修复思路。

### 7.3 HiRA 参数训练

当前只替换：

```text
q_proj
k_proj
v_proj
ffn.w2
ffn.w3
```

没有替换：

```text
o_proj
ffn.w1
```

这与本次恢复的旧版 `sft_chat_for_sharegpt_hira.py` 保持一致。

冻结策略：

```text
默认冻结全部参数
开放 token_embeddings.embedding_weights
开放 lm_head.W
开放所有 HiRA .A / .B 参数
```

但 embedding 和 lm_head 通过 hook 屏蔽 base vocab 梯度：

```text
embedding grad[:32000] = 0
lm_head grad[:, :32000] = 0
```

所以实际只训练新增 chat token 的 embedding 行和 lm_head 列。

## 8. 已执行验证

### 8.1 Python 编译检查

已运行：

```bash
./.venv/bin/python -m py_compile sft_chat_for_sharegpt_hira.py
```

结果：通过。

### 8.2 数据加载检查

已运行轻量检查，结果：

```text
sharegpt 391655 [352490, 39165]
short 31602 [28442, 3160]
```

说明当前两个数据源都能被脚本读取并切分。

### 8.3 预训练 checkpoint 加载检查

已实例化完整 24 层模型并调用 `load_pretrain_weights(model)`。

验证输出：

```text
FastTokenizerOWTHighPerformance: using C++ bpe_fast
resized token_embeddings.embedding_weights: (32000, 1024) -> (32003, 1024)
resized lm_head.W: (1024, 32000) -> (1024, 32003)
vocab_size 32003
embedding (32003, 1024)
lm_head (1024, 32003)
```

说明当前脚本能从 `32000` vocab 的 pretrain checkpoint 初始化到 `32003` vocab 的 chat SFT 模型。

## 9. 未执行的事项

本次没有启动完整训练。

原因是本次任务重点是恢复和重构脚本，并验证关键加载路径。完整训练会占用较长时间和 GPU 资源，因此只做了语法、数据、checkpoint 三类 smoke test。

## 10. 当前工作区状态

当前主要改动是：

```text
M sft_chat_for_sharegpt_hira.py
```

本文件是本次新增文档：

```text
docs/sft_chat_for_sharegpt_hira_conversation_summary.md
```

## 11. 运行方式

当前脚本已经没有命令行参数，直接运行：

```bash
./.venv/bin/python sft_chat_for_sharegpt_hira.py
```

输出 checkpoint：

```text
./checkpoints/sharegpt_hira_iter_{it}.pt
```

最终权重：

```text
./data/sharegpt_only_hira_final.pt
```

