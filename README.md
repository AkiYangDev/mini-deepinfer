<div align="center">

# mini-DeepInfer

**跟着一次真实请求，读懂 DeepInfer 的执行过程。**

A small inference engine for learning DeepInfer, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

**简体中文** · [English](README.en.md)

[开始运行](docs/START_HERE.md) · [A、B、C 演示](docs/WALKTHROUGH.md) · [十四课学习路线](docs/LEARNING_PATH.md) · [v0.2 说明](docs/RELEASE_v0.2.md)

</div>

mini-DeepInfer 是一个面向 Ascend 910C 的教学推理引擎，设计参考 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。你可以沿着一次请求，读清输入准备、模型计算、采样和结果提交的顺序，再看看请求结束后，缓存到底什么时候才能归还。

目前已用项目自己的前向计算，在 **Ascend 910C 上跑通 Qwen3.5-2B 文本生成**。教程从能直接打印状态的整数例子开始，逐步换成 PyTorch 小模型和真实权重，共十四课。开发时也使用 Qwen3.5-0.8B，原有 DeepSeek-R1-Distill-Qwen-1.5B 示例继续用于回归检查。

v0.2 把跨轮执行接到文本生成入口，并提供一个无需权重的完整调度演示。[版本说明](docs/RELEASE_v0.2.md)列出新增入口和验证范围。

## 先跑起来

Python 3.10+，先看一个无需模型依赖、权重或 NPU 的例子：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m pip install -e .
python -m minideepinfer.step_demo
```

例子里，A 取消后仍要等设备用完，才能归还缓存；B 换了批次行号，仍要接上自己的上一轮结果。程序会打印这些状态变化，用手动事件模拟设备完成，无需执行模型。

已有 PyTorch 时，接着跑一个无需下载权重的混合注意力小模型：

```bash
python -m minideepinfer.pipeline_demo
```

这个例子里，A 的长 prompt 分块计算，B 先生成结束，C 排队后复用 B 的槽位。实际输出包括：

```text
  B: finished; cached CPU/issued=3/3; slots=0
  C: running; cached CPU/issued=0/3; slots=6
  observed released: B
  reused B -> C: slots [14, 15, 16]
  free slots: 0/20
```

B 已释放，空闲槽位却变成了 0，因为 C 已经拿走它们。[跟着这次调度读代码](docs/WALKTHROUGH.md)，看清 CPU 进度、已发出计算和缓存归属怎样变化；依赖安装见[运行指南](docs/START_HERE.md)。

## 在 Ascend 910C 上运行真实模型

先按[运行指南](docs/START_HERE.md#ascend-910c真实权重)准备匹配的 PyTorch、torch_npu、CANN 和固定版本的 Qwen3.5-2B 权重。在仓库目录运行：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

真实 2B 运行中的英文回答：

```text
1 plus 1 equals 2.

[eos; 9 output tokens; free KV slots 512/512]
```

[完整双语输出](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt)保留在仓库。默认采用 FP32、eager 和 greedy，适合短上下文教学。没有 NPU 时，可以改用 `--device cpu` 和 0.8B 权重；仍需要加载完整文本模型。

## 沿着请求读代码

| 你想弄清的问题 | 从这里读 |
| --- | --- |
| prefill、decode 每一步输入什么？ | [`Engine.step()`](minideepinfer/engine.py) · [第一课](docs/FIRST_REQUEST.md) |
| 多请求怎样合批，位置和缓存怎样隔离？ | [`BatchEngine`](minideepinfer/batch_engine.py) · [第三课](docs/BATCHING.md) |
| Qwen3.5 为什么需要两种缓存？ | [`HybridCache`](minideepinfer/hybrid_cache.py) · [第四课](docs/QWEN35.md) |
| 计算已经发出，取消后怎样收尾？ | [`DeferredRunner`](minideepinfer/deferred_runner.py) · [第七、八课](docs/DEFERRED_ENGINE.md) |
| CPU 没读到 token，下一轮怎样接着算？ | [`decode_pipeline.py`](minideepinfer/decode_pipeline.py) · [第十一课](docs/DECODE_PIPELINE.md) |
| EOS、新请求、分块 prompt 怎样进入同一个循环？ | [`StoppingDecodePipeline`](minideepinfer/stopping_pipeline.py) · [第十二至十四课](docs/CHUNK_PIPELINE.md) |
| 已发出、已提交和设备已用完，有什么区别？ | [`IssuedProgress` / `plan_step()`](minideepinfer/planning.py) · [第十四课](docs/CHUNK_PIPELINE.md) |

默认 BatchEngine 支持同步动态批处理、输入预算、分块 prefill、取消和回收；`--deferred` 把发起与结果提交拆开，仍只允许一批在途。现在也可以用 `--pipeline` 选择跨轮执行，配合 `--stagger-prompts --trace-rounds` 观察文本到达和缓存回收，见[自己的文本怎样进入跨轮循环](docs/PIPELINE_TEXT.md)。各章节入口见[学习路线](docs/LEARNING_PATH.md)。

## 验证到了哪里

| 检查 | 已有证据 |
| --- | --- |
| 真实 2B / Ascend 910C FP32 | 单请求、双请求与独立参考对照通过，检查位置 top-1 一致 |
| 跨轮分块、到达与复用 | 九轮、十四组完整 logits 最大差约 `2.67e-5`，原容差通过；四个中间块不采样，24/24 槽位归还 |
| 新文本生成入口 | 同步、跨轮与逐条到达输出一致；聊天生成自然遇 EOS，含额外已发出一步的 43 组 logits 对照通过，最大差约 `4.94e-5`；见[运行说明](docs/PIPELINE_TEXT.md) |
| 取消与主机异常 | 等最后使用完成后清理，Runner 可复用；等待失败保留资源的分支另有 CPU 注入检查 |
| BF16 | 固定短轨迹通过同扫描方式、FP32 状态参考；原生分块与跨调度长生成仍有差异，见[说明](docs/BF16_REFERENCE.md) |

这些结果验证了短上下文下的数值和状态处理。性能实验目前只覆盖少量短请求，尚未得到稳定提速的结论。[验证记录](docs/VALIDATION.md)保留环境、原始输出和源码 SHA；v0.1 与 v0.2 开发版的报告分别保存，核对命令见[文本入口说明](docs/PIPELINE_TEXT.md)。

## 为什么参考 DeepInfer

读 DeepInfer 时，输入准备、设备计算和结果处理分散在不同组件里，还要考虑前后轮之间的依赖。本项目把这些步骤拆成小实验，便于单独运行和对照。[源码映射](docs/SOURCE_MAP.md)列出了参考的上游版本、对应入口，以及教学版省略或改写的部分。Qwen3.5 则用来观察真实模型计算和两种缓存的变化。

[mini-sglang](https://github.com/sgl-project/mini-sglang)提供精简实现和教学组织的参考，Ascend SGLang 提供 NPU 接口与兼容性参考。MLA、MoE、投机解码、bounded replay、图执行和分布式留在[后续路线](docs/ROADMAP.md)。来源与许可证见[参考记录](docs/REFERENCES.md)。

欢迎带着一个具体问题[提交 Issue](https://github.com/AkiYangDev/mini-deepinfer/issues)：哪一步看不懂，哪条命令没跑通，或哪项缓存行为需要补例子。也欢迎按[贡献指南](CONTRIBUTING.md)改进代码和教程。如果这条学习路线对你有用，欢迎 Star。

本项目为独立社区项目，与 DeepSeek、SGLang 及华为无官方隶属关系。原创代码与文档采用 [Apache-2.0](LICENSE)，模型及第三方代码遵循各自许可证。
