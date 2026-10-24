<div align="center">

# mini-DeepInfer

**跟着一次真实请求，读懂 DeepInfer 的执行过程。**

A small inference engine for learning DeepInfer, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

**简体中文** · [English](README.en.md)

[开始阅读](#沿着一个请求读代码) · [运行模型](#运行真实模型) · [验证记录](docs/VALIDATION.md) · [开发路线](docs/ROADMAP.md)

</div>

mini-DeepInfer 借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的设计，用一个小型推理引擎解释：输入怎样变成 token，prefill 和 decode 如何衔接，KV Cache 写到哪里，以及请求结束后如何回收。

现在可以用项目自己的前向计算，在 **Ascend 910C 上运行 Qwen3.5-2B 的纯文本推理**。Qwen3.5-0.8B 用于开发验证，已有 DeepSeek-R1-Distill-Qwen-1.5B 保留为回归基线。完整注意力保存 KV，DeltaNet 保存每个请求的卷积和递推状态。

当前主要示例使用 FP32。BF16 已通过固定短轨迹的逐 token / FP32 状态参考检查，原生分块与跨调度生成的差异见[对照说明](docs/BF16_REFERENCE.md)。支持同步动态批处理、分块 prefill、请求取消和缓存回收。

## 已经跑通什么

| 检查 | 当前结果 |
| --- | --- |
| Qwen3.5-0.8B / Ascend 910C FP32 | 分块 prefill + 4 步固定 token decode；最大绝对误差约 `2.00e-5` |
| Qwen3.5-2B / Ascend 910C FP32 | 单请求与双请求数值对照通过；最大绝对误差约 `2.29e-5`，检查位置 top-1 全部一致 |
| Qwen3.5 BF16 递推对照 | 0.8B、2B 与 2B 双请求的固定短轨迹，在同扫描方式与状态精度下最大差异 **0**，零容差通过 |
| 2B 双请求生成 | 中文解释与英文问答均遇 EOS 结束，512/512 逻辑槽位回收 |
| Ascend 910C 事件实验 | 微型张量的跨流读取、CPU 结果拷贝与延迟回收通过；[第六课](docs/NPU_EVENTS.md)保留实际输出 |
| 2B 结果延后提交实验 | 六轮真实权重 FP32 轨迹与同步 Runner 的 logits 差异为 0；覆盖分块、取消和异常清理，见[第七课](docs/DEFERRED_RUNNER.md) |

真实 2B 运行中的英文回答：

```text
1 plus 1 equals 2.

[eos; 9 output tokens; free KV slots 512/512]
```

[完整双语输出](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt)、输入、环境、逐步误差和 BF16 失败记录均见[验证记录](docs/VALIDATION.md)。这些是短上下文正确性检查，没有吞吐或延迟结论。

## 先跑一个小例子

想先看懂执行过程，可以先跑一个无需依赖、权重或 NPU 的小例子：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

它跟着 A、B、C 三个请求，展示 B 换了行怎样接上自己的结果，以及 A 取消后为什么还不能立刻归还缓存。讲解见[第五课：请求结束了，缓存为什么还不能还？](docs/STEP_STATE.md)。这是手动事件驱动的 CPU 实验，真实模型路径仍按下面的方法运行。

有 Ascend 环境后，可以接着读[第六课：NPU 算完了，CPU 就能读吗？](docs/NPU_EVENTS.md)，用真实事件和小张量检查同样的顺序，无需加载权重。

[第七课](docs/DEFERRED_RUNNER.md)把这套顺序接到真实模型：先提交计算，结果回来后再更新进度。它也提供无需权重的 PyTorch 小模型示例，当前只允许一批在途。

## 运行真实模型

在仓库目录安装模型依赖：

```bash
python -m pip install -e ".[qwen35]"
```

按照[第四课的固定版本下载命令](docs/QWEN35.md#准备权重)准备权重。在已有匹配的 PyTorch、torch_npu 和 CANN 环境中运行：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

没有 NPU 时，可改成 `--device cpu --dtype float32`，先用 0.8B 跟读和调试。本实现适合短上下文教学，CPU 也需要加载完整文本模型。

## 沿着一个请求读代码

如果你熟悉基本的 Python / PyTorch，可以按下面的顺序读。模型计算由本项目实现，Transformers 提供 tokenizer 和数值对照。

| 你想弄清的问题 | 从这里读 |
| --- | --- |
| prefill 和 decode 每轮分别输入什么？ | [`Engine.step()`](minideepinfer/engine.py) |
| token、位置和 KV 槽位怎样对应？ | [`ModelRunner.prepare_input()`](minideepinfer/model_runner.py) |
| attention 怎样读写历史 K/V？ | [`qwen2.py`](minideepinfer/models/qwen2.py) · [`TensorKVCache`](minideepinfer/tensor_cache.py) |
| 线性注意力怎样记住历史，又怎样隔开请求？ | [`qwen3_5.py`](minideepinfer/models/qwen3_5.py) · [`HybridCache`](minideepinfer/hybrid_cache.py) |
| logits 怎样成为下一个 token？ | [`ModelRunner.execute_sample()`](minideepinfer/model_runner.py) |
| 请求结束或取消后，谁负责回收？ | [`Engine`](minideepinfer/engine.py) · [`SlotPool`](minideepinfer/kv_cache.py) |

还没接触过推理引擎，可以从[第一课：跟踪一个请求](docs/FIRST_REQUEST.md)开始。整数 Runner 让输入、位置和槽位都能直接打印出来，Python 3.10+ 即可运行，无需模型依赖或权重：

```bash
python -m minideepinfer
python -m unittest discover -s tests -v
```

未安装模型依赖时执行 30 项引擎与状态测试，模型对照明确跳过。[第二课：真实权重与 KV Cache](docs/REAL_GENERATION.md)再把同一条执行路径换成真实模型。

## 为什么参考 DeepInfer

DeepInfer 是技术主线。当前按照它的职责划分，展开输入准备、主模型执行、采样和结果提交；第五至七课逐步从整数事件走到真实 NPU 与模型，把跨轮状态、批次映射和结果提交讲清楚。调度重叠仍需后续验证。[源码映射](docs/SOURCE_MAP.md)列出上游固定版本、对应入口和教学简化。

**Qwen 小模型承载教学实例，DeepInfer 继续作为引擎设计主线。** 单请求与同步批次保留对照，[第四课](docs/QWEN35.md)展开混合模型的状态管理。MLA、MoE、投机解码与 bounded replay 留在后续独立章节，范围见[开发路线](docs/ROADMAP.md)。

[mini-sglang](https://github.com/sgl-project/mini-sglang) 提供精简实现与教学组织的参考；Ascend SGLang 提供 NPU 接口与兼容性参考。详细设计见[架构说明](docs/ARCHITECTURE.md)，来源与许可证见[参考与归属](docs/REFERENCES.md)。

## 参与这个项目

欢迎带着一个具体问题来交流：哪一步看不懂，哪条命令没有跑通，或哪项缓存行为需要补一个例子。可以[提交 Issue](https://github.com/AkiYangDev/mini-deepinfer/issues)，也欢迎按[贡献指南](CONTRIBUTING.md)改进代码和教程。

如果这条学习路线对你有用，欢迎 Star，后续继续一起读代码、做实验。

本项目为独立社区项目，与 DeepSeek、SGLang 及华为无官方隶属关系。原创代码与文档采用 [Apache-2.0](LICENSE)，模型及第三方代码遵循各自许可证。
