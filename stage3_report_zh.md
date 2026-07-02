# Stage3 训练记录

## 目标

`stage3` 的目标不是继续做单物体属性识别，而是把任务推进到：

- 双物体
- 左右位置绑定
- 自然 caption 输出
- 为后续更复杂 synthetic 和 Flickr30k 做过渡

当前数据由 [color_shape_generator.py](/home/yldeveloper/learning/CS336/ai/data/vit_colors_shapes/color_shape_generator.py:1) 生成，采用：

- `scene`: 70%
- `left_object`: 15%
- `right_object`: 15%

其中：

- `scene` 负责整图描述
- `left_object` 负责左物体描述
- `right_object` 负责右物体描述

## 数据形式

每张图固定两个物体：

- 一个放在左边区域
- 一个放在右边区域

输出模板统一围绕以下语义单元展开：

- `a red triangle on the left`
- `a blue rectangle on the right`

例如：

- `The image shows a black triangle on the left and a red rectangle on the right.`
- `There is a white rectangle on the left and a red triangle on the right.`
- `The scene shows a blue ellipse on the right.`

## 本轮训练结果

本轮末尾日志：

```text
[eval] iter  12000 | val_loss 0.531538
[greedy/eval] left_object:0.562 overall:0.354 right_object:0.000 scene:0.500
[final] val_loss 0.490160
```

中间样例：

```text
[sample] prompt='what does the image show?' gold='There is a black triangle on the left and a red rectangle on the right.' pred='The image shows a black triangle on the left and a red rectangle on the right.'

[sample] prompt='what do you see?' gold='This image contains a yellow rectangle on the left and a black triangle on the right.' pred='The image shows a yellow triangle on the left and a black triangle on the right.'

[sample] prompt='describe both objects' gold='I see a white rectangle on the left and a red triangle on the right.' pred='The image shows a white rectangle on the left and a red triangle on the right.'
```

## 当前结论

这次 `stage3` 结果不能算失败，但还不能算“任务完成”。它说明了两件事。

### 1. 整图 caption 能力已经起来了

从样例看，模型已经明显学会：

- 输出自然 caption
- 描述左右两个物体
- 在很多情况下把颜色、形状、左右位置绑定正确

例如：

- `black triangle on the left + red rectangle on the right`
- `white rectangle on the left + red triangle on the right`

这些说明视觉前缀不只是识别单物体，而是已经开始承载双物体描述。

### 2. 问答路由明显还没建立

你给的自由测试非常清楚地说明了这一点：

```text
image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: w
output: The image shows a blue triangle on the left and a blue rectangle on the right.

image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: hello
output: The image shows a blue triangle on the left and a blue rectangle on the right.

image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: what is its color?
output: The image shows a blue rectangle on the left and a blue rectangle on the right.

image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: what is the shape
output: The scene shows a blue triangle on the left and a blue rectangle on the right.

image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: what is the right one's shape?
output: The scene shows a blue rectangle on the left.

image path (blank for text-only): ./data/vit_colors_shapes/val/000000.bmp
text: what is the right one's color?
output: The scene shows a blue rectangle on the left.

image path (blank for text-only): ./data/vit_colors_shapes/val/000001.bmp
text: descibe
output: The image shows a red ellipse on the left and a red ellipse on the right.
```

这段输出说明当前模型策略非常明确：

- 只要看到图，它就强烈倾向于输出“整图 caption”
- 它还没有真正学会根据不同问法切换成“只答左物体”或“只答右物体”
- 甚至对无意义 prompt（`w`、`hello`）也会直接回退到整图描述

这和当前训练分布是一致的，因为：

- `scene` 占 70%
- 输出模板高度集中在整图 caption
- `left_object/right_object` 占比偏低

所以模型学到的最强先验就是：

> 给图以后，最稳的做法是直接描述整张图。

## 为什么 greedy/eval 看起来偏低

`scene=0.500`、`left_object=0.562`、`right_object=0.000` 说明：

1. 双物体完整绑定还不稳定
2. 单侧对象问答尤其是 `right_object` 明显偏弱
3. 模型更容易退回到“整图 caption”而不是按问题聚焦

尤其 `right_object=0.000` 很值得注意，这通常不是“完全不会看右边”，而是：

- 模型输出了整图 caption
- 或者输出了左边内容
- 因而没有满足“只答右物体”的评分标准

从自由测试里看，这种现象是存在的。

## 对这次 stage3 的判断

如果把目标定为：

- 学会双物体自然描述

那么这次已经有明显进展。  

如果把目标定为：

- 同时学会 `scene`
- 学会 `left_object`
- 学会 `right_object`
- 并且对开放问法也能正确路由

那么这次还不够。

## 建议的下一步

当前最合理的收口不是继续盲目延长训练，而是调整数据分布和任务设计：

1. 降低 `scene` 比例
   - 例如从 `70%` 降到 `50%`
2. 提高 `left_object/right_object` 比例
   - 例如改成 `25% / 25%`
3. 增加更直接的左右问法
   - `what is on the right?`
   - `describe the right object`
   - `what color is the right object?`
   - `what shape is the right object?`
4. 若继续强调 caption-first，则单独接受一个事实：
   - 它会把大多数图像输入都当成 caption 任务来回答

也就是说，当前系统已经更像一个“图像描述器”，还不是一个稳定的多物体 VQA 模型。

## 阶段结论

`stage3` 当前状态可以概括为：

- caption 能力已出现
- 左右绑定部分成功
- 问题路由能力明显不足
- 需要调整任务分布，而不是单纯继续跑更久
