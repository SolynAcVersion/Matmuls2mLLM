# Stage2 训练记录

## 目标

本阶段目标是验证当前多模态路线是否可行：

- 视觉编码器使用 `openai/clip-vit-base-patch16`
- 语言底座使用 `stage3j` 文本模型
- 将 `HiRA` checkpoint 先合并为普通 `Linear`
- 冻结视觉编码器和语言主干，只训练 `projector`
- 先在合成颜色/形状数据上完成稳定对齐，再进入更复杂阶段

## 起点问题

最开始的 `stage2` 不是从一个稳定的数据和评估设置起步，而是在不断试错中收敛出来的。主要问题有：

1. 图文训练样本一开始没有完全对齐到 `stage3j` 的 chat 格式。
2. 多模态训练早期直接混入了 `yes/no` 验证任务，导致输出分布互相竞争。
3. 生成器最开始逻辑偏复杂，还带过 GPU/批量渲染版本，不适合后续快速迭代。
4. 评估最开始使用整句 `exact match`，但输出模板已经多样化，这会严重低估真实效果。

## 关键修改过程

### 1. 语言主干从 HiRA checkpoint 改为先 fold 再训练

为了降低显存占用并避免在线 `HiRA` 带来的额外开销，增加了 `fold_stage3j_hira_to_dense.py`，将 `stage3j` 的 `HiRALinear` 参数合并成普通权重，再供 `vit_train.py` 加载。

这样做之后，`projector-only` 训练能够在现有显卡上稳定运行。

### 2. 批构造切到 stage3j chat 格式

`vit_modules.py` 中的图文 batch 构造被改为：

- 使用 `encode_chat_example(...)`
- 使用 `<|user|> ... <|assistant|> ... <|endoftext|>` 风格
- 仅监督 assistant 输出部分
- 图像仍然作为 visual prefix embedding 拼接到文本前面

这一步的意义是让多模态训练目标与原始文本 assistant 分布保持一致。

### 3. 移除 yes/no 任务

中期实验表明，`shape` / `color` / `shape_color` / `verify_shape_color(yes/no)` 混训会导致明显的“路由冲突”：

- `what color? -> triangle`
- `what shape? -> yes`
- `describe ... -> blue triangle`

这不是视觉完全失效，而是 decoder 首 token 分布被不同任务抢占。  
因此后续版本彻底移除了 `yes/no` 任务。

### 4. 生成器收缩为 CPU、单图、低复杂度版本

`data/vit_colors_shapes/color_shape_generator.py` 最终被收成：

- 纯 CPU
- `PIL + ImageDraw`
- 单物体
- `.bmp` 输出
- 简单 for-loop 生成
- 普通行日志输出进度

这样后续继续改 prompt、改输出模板、改任务比例时，生成器本身不会成为干扰项。

### 5. 从裸词输出改成自然语言输出

尝试过以下几种输出风格：

1. 裸词风格
   - `blue`
   - `triangle`
   - `blue triangle`

2. 固定模板风格
   - `The color is blue.`
   - `The shape is triangle.`
   - `The color is blue. The shape is triangle.`

3. 多模板自然语言风格
   - `The object is blue.`
   - `Its color is blue.`
   - `The object is a triangle.`
   - `It is a blue triangle.`
   - `The object is a blue triangle.`

中间有一次尝试把所有输出都固定成 `The ...` 开头，是为了缓解多任务竞争；后来又进一步做成“自然语言模板多样化”，避免模型只背一个句首模式。

### 6. 任务分布改成以 shape_color 为主

尝试后确认，更稳的分布不是人工均衡的 `1:1:1`，而是：

- `shape`: 15%
- `color`: 15%
- `shape_color`: 70%

理由：

- `shape_color` 更接近后面 caption 式输出的目标
- `shape` 和 `color` 只作为辅助任务，帮助属性拆分
- 这比把三类任务完全平权更接近自然分布

### 7. greedy eval 从 exact match 改为语义抽取

在输出模板多样化后，整句匹配会把大量“语义正确但措辞不同”的预测算错。  
例如：

- gold: `It shows a green ellipse.`
- pred: `The object is a green ellipse.`

原先会判错，实际上视觉语义完全正确。  

因此 `vit_train.py` 的 greedy eval 后来改成：

- 从预测句子抽取颜色词：`red/yellow/blue/green/white/black`
- 从预测句子抽取形状词：`rectangle/ellipse/triangle`
- `color` 任务只比颜色
- `shape` 任务只比形状
- `shape_color` 任务要求两者都对

这一步是本阶段最关键的评估修正之一。

## 中期现象

在 `yes/no` 尚未移除、输出模板和评估方式还没稳定时，出现过明显的任务干扰，例如：

- `shape? -> green`
- `what is its color? -> ellipse`
- `what shape? -> yes`

这说明模型不是完全不会看图，而是“任务输出分布”设计有问题。

后续通过：

- 去掉 `yes/no`
- 降低任务竞争
- 增加自然语言输出模板
- 改为语义抽取评估

这些问题逐步缓解。

## 最终训练结果

最终一次完整训练结束时的核心指标为：

```text
[eval] iter  12000 | val_loss 0.823154
[greedy/eval] color:1.000 overall:0.969 shape:1.000 shape_color:0.906
[final] val_loss 0.760858
[final greedy/eval] color:1.000 overall:0.969 shape:1.000 shape_color:0.906
```

这组结果说明：

1. 单属性识别已经学稳。
   - `color = 1.000`
   - `shape = 1.000`

2. 组合属性也已经相当稳定。
   - `shape_color = 0.906`

3. 当前系统已经证明：
   - visual prefix 路线有效
   - `merged stage3j backbone + projector-only` 可行
   - 当前 synthetic `stage2` 已经成功打通

## 本次 Stage2 最终交互样例

以下保留原始输出：

```text
本次stage2 - final text: color?
output: The object has a triangle shape.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: what is its color?
output: The object is red.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: color
output: The object has a triangle.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: shape?
output: The object is a red triangle.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp 
text: describe
output: The object is a triangle.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: its color
output: The object is red.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: what is its color?
output: The object is red.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: what is its shape
output: The object is a triangle.
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: describe the object
output: The object is a red triangle.
```

这组样例说明最终模型在自由输入时的表现并不是完全“按任务名机械返回”，而是：

- 能稳定识别单物体的颜色与形状
- 有时会在 `color?` 这种极短 prompt 下偏向回答形状
- 对自然描述类 prompt（如 `describe the object`）表现更稳定

这也进一步支持后续阶段应当更多朝“自然描述分布”推进，而不是继续堆短促命令式问法。

## 阶段结论

`stage2` 已经达成目标：

1. 证明当前视觉前缀路线有效。
2. 证明 `stage3j` 文本能力没有在 projector-only 训练中被破坏。
3. 证明模型已经具备基础颜色/形状视觉语义。
4. 证明后续可以进入更复杂的 synthetic 场景，再考虑 Flickr30k。

## 下一阶段建议

后续更合理的路线是：

1. 先做 `stage3 synthetic`
   - 多物体
   - 位置关系
   - 更自然 caption
   - 更复杂属性绑定

2. 再进入 Flickr30k

这样比在单物体 stage2 上继续反复抠 prompt 更有价值。
