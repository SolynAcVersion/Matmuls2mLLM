# HiRA SFT 继续训练改动记录

日期：2026-06-14

本文记录本次对 `sft_chat_templetes.py` 的训练逻辑改动。目标是从下面这个
`r=16` HiRA SFT checkpoint 继续训练：

```text
./checkpoints/RE_sft_EvolSft_HiRA_r_16_gpt2med_iter_10000.pt
```

## 背景

之前的诊断结果说明：

```text
tokenizer / special token id 是正确的
HiRA checkpoint 加载是正确的
adapter 对 logits 有明显影响
single-example overfit 可以成功
```

所以问题不是明显的代码硬 bug。更可能是原来的全量 SFT 训练太弱，尤其是
第一枚 assistant 输出 token 没学好，导致自由生成第一步经常回到预训练式续写：

```text
The following is a list of episodes...
```

此前测到的 first-token 指标大致是：

```text
mean rank around 10474
top1 around 0.05
```

也就是说，真实第一枚回答 token 经常排在非常靠后的位置。

## 主要改动

### 1. 默认从 SFT checkpoint 继续训练

新增：

```python
SFT_RESUME_CKPT_PATH = os.environ.get(
    "SFT_RESUME_CKPT_PATH",
    './checkpoints/RE_sft_EvolSft_HiRA_r_16_gpt2med_iter_10000.pt',
)
```

现在 `sft()` 默认不再从 pretrain checkpoint 重新开始，而是从这个 SFT
checkpoint 继续。

加载顺序改为：

```text
先构建 base TransformerLM
再把 q/k/v/o 和 FFN w1/w2/w3 wrap 成 HiRALinear
再把 model 移到 device
再加载 SFT checkpoint 的 state_dict
最后从 ckpt["it"] + 1 开始继续训练
```

这样做的原因是：SFT checkpoint 里已经包含 HiRA 的 `.W_0`、`.A`、`.B`
等参数，所以必须在模型完成 HiRA wrap 之后再加载。

### 2. 默认继续跑到 30000 iter

新增环境变量：

```python
max_iters = int(os.environ.get("SFT_TARGET_ITER", "30000"))
```

默认行为：

```text
resume_iter = 10000
start_iter = 10001
target_iter = 30000
```

也就是从 `10000` 的 checkpoint 继续跑到 `30000`。

### 3. 调大学习率和梯度裁剪阈值

原来的设置：

```text
max_learning_rate = 1e-5
min_learning_rate = 2e-6
max_grad_norm = 0.1
```

新的默认设置：

```text
SFT_MAX_LR = 1e-4
SFT_MIN_LR = 2e-5
SFT_WARMUP_ITERS = 500
SFT_MAX_GRAD_NORM = 1.0
```

原因：

single-example overfit 实验里，`lr=1e-4` 和 `max_grad_norm=1.0` 可以很快
把一条样本背下来，说明训练代码路径本身没问题。原全量训练的
`lr=1e-5` 和 `max_grad_norm=0.1` 对当前 adapter SFT 来说偏保守。

### 4. 默认不恢复 optimizer state

新增：

```python
SFT_RESUME_OPTIM = os.environ.get("SFT_RESUME_OPTIM", "0") == "1"
```

默认：

```text
SFT_RESUME_OPTIM=0
```

也就是默认只恢复模型权重，不恢复 AdamW 的动量状态。

原因：

这次继续训练改了学习率和梯度裁剪策略，旧 optimizer state 可能会拖住新的训练
目标，所以默认重新初始化 optimizer。

如果想连 optimizer state 一起恢复，可以运行：

```bash
SFT_RESUME_OPTIM=1 python sft_chat_templetes.py
```

## Loss 改动

新增 `run_sft_loss()`，用于给每条样本的第一枚 assistant token 加权。

默认：

```text
SFT_FIRST_TOKEN_WEIGHT = 30.0
```

旧训练 loss：

```python
loss = modules.run_cross_entropy_for_gem(logits, y, ignore_index=-666)
```

新训练 loss：

```python
loss = run_sft_loss(
    logits,
    y,
    ignore_index=-666,
    first_token_weight=first_token_weight,
)
```

这个 loss 仍然沿用 `run_cross_entropy_for_gem()` 的 causal shift 方式，只是在每条
样本的第一个被监督的 assistant token 上乘以 `first_token_weight`。

这样改的原因：

平均 SFT loss 下降不代表自由生成正常。之前模型 teacher-forcing loss 有下降，
但第一枚 assistant token 仍然很差。自由生成时第一步一旦选成 `The`，后面就很
容易回到预训练百科续写模式。

## Eval 指标改动

新增 `estimate_first_token_rank()`。

现在 eval 时除了 train/val loss，还会打印：

```text
[first-token] iter XXXXX | rank R | top1 T | n 50
```

含义：

```text
rank：真实第一枚 assistant token 在 logits 里的平均排名，越低越好
top1：真实第一枚 assistant token 排名第一的比例，越高越好
n：实际统计的验证样本数
```

这个指标比平均 val loss 更适合观察这次的问题，因为这次失败主要发生在自由生成
第一步。

继续训练后的当前观察：

```text
iter 10500 | first-token rank 3321.96 | top1 0.0800
iter 11000 | first-token rank 2397.86 | top1 0.1000
```

相比之前：

```text
mean rank around 10474
top1 around 0.05
```

说明方向是对的。

## Checkpoint 改动

checkpoint 间隔现在可以通过环境变量控制：

```python
checkpoint_interval = int(os.environ.get("SFT_CHECKPOINT_INTERVAL", "1000"))
```

默认每 1000 iter 存一次。

输出路径仍然是：

```text
checkpoints/RE_sft_EvolSft_HiRA_r_16_gpt2med_iter_{it}.pt
```

## 默认运行方式

直接运行：

```bash
python sft_chat_templetes.py
```

然后选择：

```text
2
```

等价于默认使用：

```text
SFT_RESUME_CKPT_PATH=./checkpoints/RE_sft_EvolSft_HiRA_r_16_gpt2med_iter_10000.pt
SFT_TARGET_ITER=30000
SFT_MAX_LR=1e-4
SFT_MIN_LR=2e-5
SFT_WARMUP_ITERS=500
SFT_MAX_GRAD_NORM=1.0
SFT_FIRST_TOKEN_WEIGHT=30
SFT_RESUME_OPTIM=0
```

也可以显式写成：

```bash
SFT_TARGET_ITER=30000 \
SFT_MAX_LR=1e-4 \
SFT_MIN_LR=2e-5 \
SFT_FIRST_TOKEN_WEIGHT=30 \
SFT_MAX_GRAD_NORM=1.0 \
python sft_chat_templetes.py
```

## 继续观察重点

应该重点看：

```text
first-token rank 是否继续下降
first-token top1 是否继续上升
val loss 是否没有明显爆炸
HiRA update_ratio 是否平稳上升
固定 prompt 的 greedy/free generation 是否从百科续写变成正常回答
```

当前 `iter 11000` 的趋势是正常的：

```text
first-token rank 从 3000+ 降到 2000+
top1 从 0.08 升到 0.10
update_ratio 从约 0.0048 升到约 0.0107
val loss 没有崩
```

如果到 `iter 15000` 左右 first-token rank 仍然在几千，可以考虑：

```text
SFT_FIRST_TOKEN_WEIGHT=50
```

如果训练不稳定，则优先降低：

```text
SFT_MAX_LR
```

如果想关闭 first-token 加权，使用：

```text
SFT_FIRST_TOKEN_WEIGHT=1
```

