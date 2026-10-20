<div align="center">

# mini-DeepInfer

**跟着一次真实请求，读懂 DeepInfer 的执行过程。**

A small inference engine for learning DeepInfer, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

**简体中文** · [English](README.en.md)

[开始阅读](#沿着一个请求读代码) · [运行模型](#运行真实模型) · [验证记录](docs/VALIDATION.md) · [开发路线](docs/ROADMAP.md)

</div>

mini-DeepInfer 借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的设计，用一个小型推理引擎解释：输入怎样变成 token，prefill 和 decode 如何衔接，KV Cache 写到哪里，以及请求结束后如何回收。

当前已经能用项目自己的 Qwen2 forward 和张量 KV Cache，运行 **DeepSeek-R1-Distill-Qwen-1.5B**。可以先在 CPU 上跟读和实验；**Ascend 910C 是目标平台，完整模型的 NPU 验证仍待完成。**

## 已经跑通什么

| 检查 | 当前结果 |
| --- | --- |
| 真实权重 CPU FP32 | 5 token prefill + 4 步固定 token decode；与 Transformers 缓存路径的 logits 最大绝对差为 **0** |
| 缓存与重算 | 最大绝对差约 `7.56e-5`，参考模型自身也有相同差异；检查位置的 top-1 全部一致 |
| 文本生成 | 生成 96 token，按长度停止后 **512/512** 个 KV 槽位全部回收 |
| 自动检查 | 16 项测试通过；覆盖模型对照、因果遮罩、非连续槽位、复用和请求生命周期 |

以上是固定模型、输入和依赖版本下的结果，完整环境与原始 JSON 见[验证记录](docs/VALIDATION.md)。同步动态批处理也已实现，支持请求进入、退出和混合 prefill/decode，见[第三课](docs/BATCHING.md)；调度重叠尚未实现。

一次真实的 CPU 运行输入是 `What is 1 plus 1? Answer in one short sentence.`，输出节选如下：

```text
Okay, so I need to figure out what 1 plus 1 is. Hmm, I remember learning this in school,
but I'm not entirely sure. Let me think. ...

[length; 96 output tokens; free KV slots 512/512]
```

这是 R1 模型尚未结束的思考片段，展示生成与回收流程。完整输出见[运行记录](docs/results/2026-10-20/generation-cpu.txt)，不作为回答质量或性能结论。

## 运行真实模型

先克隆项目并安装模型依赖：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m pip install -e ".[model]"
```

按照[第二课的下载步骤](docs/REAL_GENERATION.md#准备环境与权重)准备本地权重后，可以先在 CPU 上生成：

```bash
python -m minideepinfer.generate \
  --model weights/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu --dtype float32 \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 96
```

Ascend 环境需要匹配的 PyTorch、`torch_npu` 和 CANN；NPU 运行与数值验证命令见[第二课](docs/REAL_GENERATION.md)。本实现适合短上下文教学，CPU 运行也需要加载完整模型权重。

## 沿着一个请求读代码

如果你熟悉基本的 Python / PyTorch，可以按下面的顺序读。模型计算由本项目实现，Transformers 提供 tokenizer 和数值对照。

| 你想弄清的问题 | 从这里读 |
| --- | --- |
| prefill 和 decode 每轮分别输入什么？ | [`Engine.step()`](minideepinfer/engine.py) |
| token、位置和 KV 槽位怎样对应？ | [`ModelRunner.prepare_input()`](minideepinfer/model_runner.py) |
| attention 怎样读写历史 K/V？ | [`qwen2.py`](minideepinfer/models/qwen2.py) · [`TensorKVCache`](minideepinfer/tensor_cache.py) |
| logits 怎样成为下一个 token？ | [`ModelRunner.execute_sample()`](minideepinfer/model_runner.py) |
| 请求结束或取消后，谁负责回收？ | [`Engine`](minideepinfer/engine.py) · [`SlotPool`](minideepinfer/kv_cache.py) |

还没接触过推理引擎，可以从[第一课：跟踪一个请求](docs/FIRST_REQUEST.md)开始。整数 Runner 让输入、位置和槽位都能直接打印出来，Python 3.10+ 即可运行，无需模型依赖或权重：

```bash
python -m minideepinfer
python -m unittest discover -s tests -v
```

未安装模型依赖时执行 8 项引擎测试，另 8 项明确跳过。[第二课：真实权重与 KV Cache](docs/REAL_GENERATION.md)再把同一条执行路径换成真实模型。

## 为什么参考 DeepInfer

DeepInfer 是技术主线。当前按照它的职责划分，展开输入准备、主模型执行、采样和结果提交；后续再学习跨轮状态与调度重叠。[源码映射](docs/SOURCE_MAP.md)列出上游固定版本、对应入口和教学简化。

**当前 Qwen2 模型用于打通入门执行路径，不代表已经复现 DeepInfer 的进阶机制。** 单请求与同步批次保留对照，接下来补齐 Ascend 910C 验证。MLA、MoE、投机解码与 bounded replay 留在后续独立章节，范围见[开发路线](docs/ROADMAP.md)。

[mini-sglang](https://github.com/sgl-project/mini-sglang) 提供精简实现与教学组织的参考；Ascend SGLang 提供 NPU 接口与兼容性参考。详细设计见[架构说明](docs/ARCHITECTURE.md)，来源与许可证见[参考与归属](docs/REFERENCES.md)。

## 参与这个项目

欢迎带着一个具体问题来交流：哪一步看不懂，哪条命令没有跑通，或哪项缓存行为需要补一个例子。可以[提交 Issue](https://github.com/AkiYangDev/mini-deepinfer/issues)，也欢迎按[贡献指南](CONTRIBUTING.md)改进代码和教程。

如果这条学习路线对你有用，欢迎 Star，后续继续一起读代码、做实验。

本项目为独立社区项目，与 DeepSeek、SGLang 及华为无官方隶属关系。原创代码与文档采用 [Apache-2.0](LICENSE)，模型及第三方代码遵循各自许可证。
