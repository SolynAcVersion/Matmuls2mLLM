# From MatMuls to mLLM

一个从零实现、逐步走到多模态规划阶段的实验性仓库。项目主线来自 CS336 的手写 Transformer 训练实践：先自己实现分词器、解码器式语言模型、优化器与训练循环，再做文本预训练和 SFT，最后把目标指向 mLLM。

> 截至 2026-06-26：仓库里已经完成的是 **text-only** 路线；`ViT` / 多模态桥接 **还没有正式开始实现**。`reports/` 里已经有 ViT 规划文档，但当前代码库中还没有视觉编码器、projector、图文拼接训练脚本或多模态推理入口。

## 项目目标

这个项目想做的不是“调一个现成框架”，而是尽量自己实现一条完整链路：

- BPE 训练与分词
- GPT-2 Medium 量级的 decoder-only Transformer
- 文本预训练
- 基于 HiRA / LoRA 风格适配器的 SFT
- 面向未来多模态扩展的代码与实验积累

更完整的实验叙事见：

- [reports/From MatMuls to mLLM.md](reports/From%20MatMuls%20to%20mLLM.md)
- [reports/reports_merged_timeline.md](reports/reports_merged_timeline.md)
- [reports/vit_multimodal_tutorial_zh.md](reports/vit_multimodal_tutorial_zh.md)
- [reports/vit_stage3j_migration_tutorial_zh.md](reports/vit_stage3j_migration_tutorial_zh.md)

## 当前做到哪里了

### 1. 文本底座已经跑通

当前仓库已经实现并实际跑过：

- 从零实现的 BPE / Tokenizer 主链路
- 基于 `pybind11` 的 C++ BPE 加速模块
- 高性能多进程 tokenize 流程
- GPT-2 Medium 量级 decoder-only Transformer
- OpenWebText / Wikipedia 风格数据上的预训练脚本
- 多轮文本 SFT 实验

当前主模型大致是：

- `vocab_size = 32000`，SFT 时额外补上 4 个 chat special tokens
- `context_length = 1024`
- `d_model = 1024`
- `num_layers = 24`
- `num_heads = 16`
- `d_ff = 2752`
- `RoPE + RMSNorm + SwiGLU`

这部分核心代码主要在 [modules.py](./modules.py)、[tokenization.py](./tokenization.py)、[train.py](./train.py)。

### 2. SFT 已经走到 text-only 路线边界

根据 `reports/reports_merged_timeline.md` 和当前脚本状态，项目在文本 SFT 上已经得到一个比较明确的工程结论：

- `stage3j` 是目前文本侧较好的 checkpoint 底座
- `stage4_task_aligned_v1/v2` 让模型更稳定地学会了“回答模式”
- 但模型依然没有可靠跨过“内容真正贴题、语义真正具体”这条线

也就是说，当前这条 `GPT2-Medium + HiRA + 普通 SFT` 路线，已经比较接近 text-only 条件下的现实边界：模型能学会“该怎么回答”，但很难稳定学会“回答得足够好”。

仓库里目前的文本 SFT 主入口是 [re_sft_ds.py](./re_sft_ds.py)。

### 3. ViT 还没有开始做

这一点需要明确写清楚：

- `reports/` 里已经有 ViT / multimodal 的设计、目标和训练规划
- 但当前仓库 **没有** ViT 实现代码
- 也 **没有** 图像数据管线、视觉特征投影层、图文拼接训练入口
- 因此这个仓库现在还不是一个可运行的 mLLM，只是已经走到“从文本走向多模态”的门口

如果你是因为标题里的 `mLLM` 点进来，需要先接受这一点：**当前开源的是完整的文本阶段积累，不是已经完成的多模态模型。**

## 仓库结构

- [modules.py](./modules.py)：核心模型实现，包含线性层、Embedding、RMSNorm、RoPE、Attention、TransformerLM、AdamW、采样、BPE、Tokenizer 等
- [tokenization.py](./tokenization.py)：训练 BPE、切分 parquet、构建 `.npy` 预训练分片
- [train.py](./train.py)：文本预训练入口
- [sft_chat_templetes.py](./sft_chat_templetes.py)：chat 模板、tokenizer 装载、HiRA/LoRA 适配器和较早期 SFT 逻辑
- [re_sft_ds.py](./re_sft_ds.py)：当前较新的文本 SFT 主线
- [gen_stage4_task_aligned_deepseek.py](./gen_stage4_task_aligned_deepseek.py)：用 DeepSeek API 生成 stage4 task-aligned 数据
- [build_bpe_fast.py](./build_bpe_fast.py)：编译 `bpe_fast.cpp`
- [build_bpe_merge_cpp.py](./build_bpe_merge_cpp.py)：编译 `bpe_merge.cpp`
- [reports/](./reports)：完整实验记录、总结和 ViT 规划

## 运行说明

### 依赖

仓库目前还是“研究脚本”形态，没有整理成正式 package。至少需要这些依赖：

```bash
pip install torch numpy pandas regex wandb einops jaxtyping pybind11 requests
```

### 编译 C++ BPE 扩展

```bash
python build_bpe_merge_cpp.py build_ext --inplace
python build_bpe_fast.py build_ext --inplace
```

如果编译成功，会得到类似：

- `bpe_merge.cpython-*.so`
- `bpe_fast.cpython-*.so`

### 准备密钥

部分脚本会读取根目录下的 `pswd.json`：

```json
{
  "wandb-api-key": "YOUR_WANDB_KEY",
  "deepseek-api-key": "YOUR_DEEPSEEK_KEY"
}
```

其中：

- `train.py` 会用到 `wandb-api-key`
- `gen_stage4_task_aligned_deepseek.py` 会用到 `deepseek-api-key`

### 数据与脚本入口

当前脚本依赖本地 `data/` 下的特定文件，路径大多是硬编码的，因此更适合作为实验复现参考，而不是开箱即用的一键训练仓库。

常见入口如下：

```bash
# 1) 训练 BPE / 处理 parquet / 生成预训练 npy
python tokenization.py

# 2) 文本预训练
python train.py

# 3) 生成 stage4 task-aligned SFT 数据
python gen_stage4_task_aligned_deepseek.py

# 4) 文本 SFT
python re_sft_ds.py
```

## 当前阶段结论

如果只看这份代码库当前真实状态，可以把结论压缩成三句话：

1. 文本预训练和文本 SFT 主链路已经走通。
2. `stage4_task_aligned_v2` 基本标出了当前 text-only SFT 的工程边界。
3. ViT / multimodal 仍处于“报告里已经规划、代码里尚未实现”的状态。

## 后续方向

从当前 `reports` 的结论看，继续在同类 text-only SFT 上堆更多普通数据，边际收益已经很低。下一阶段更合理的方向是：

- 引入视觉编码器
- 建立 visual projector
- 先做合成颜色/形状识别
- 再逐步扩展到真正的 multimodal bridge

但再次强调：**这些还没有在当前仓库中落地。**
