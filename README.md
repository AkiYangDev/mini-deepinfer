<div align="center">

# mini-DeepInfer

**在 Ascend 910C 上，读懂推理引擎的执行过程。**

A readable inference engine for learning DeepInfer, on CPU and Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

**简体中文** · [English](README.en.md)

[快速开始](#快速开始) · [Ascend 910C 实机](#在-ascend-910c-上生成文本) · [学习路线](docs/LEARNING_PATH.md) · [源码对应](docs/SOURCE_MAP.md)

</div>

mini-DeepInfer 参考 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的执行设计，用 Python 和 PyTorch 实现调度、模型前向与缓存读写。你可以从 CPU 上的状态演示开始，一路跑到 Ascend 910C 上的真实 Qwen3.5 模型，跟踪一次请求怎样计算、提交、停止和归还缓存。

## 项目亮点

- **Ascend 910C 实机验证。** Qwen3.5-2B 已跑通单请求、批处理与跨轮文本生成，仓库保留[真实输出](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt)、数值对照和对应源码版本。
- **没有 NPU，也能跑模型实验。** 整数例子只需 Python；装好 PyTorch 后，可以运行随机小模型，观察真实 forward、混合缓存、跨轮调度和候选拒绝，全程无需下载权重。
- **十四课，沿请求读代码。** 从 prefill、decode 走到分块、动态到达与取消，逐步分清 CPU 已提交、设备已发出和最后使用完成。每课有[运行入口与代码位置](docs/LEARNING_PATH.md)。
- **双模型生成与状态回退。** Qwen3.5-0.8B 提候选、2B 验证；main 支持自己的文本、流式输出和提交后取消。用[逐轮演示](docs/MODEL_DRAFT_WALKTHROUGH.md)看清候选猜错后，两边怎样接回已提交的前缀。

## 快速开始

Python 3.10+，在源码目录运行。这个例子不需要 PyTorch、权重或 NPU：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

A 取消了，设备却还在使用它的缓存；B 换了批次行号，仍要接上自己的上一轮结果。程序通过手动事件推进设备完成状态，把提交与回收的顺序打印出来。

已有 PyTorch 时，接着运行一个无需权重的 Qwen3.5 小模型：

```bash
python -m minideepinfer.pipeline_demo
```

这次 A 的长 prompt 分块计算，B 先结束，C 排队后拿到 B 的槽位。实际输出片段：

```text
  B: finished; cached CPU/issued=3/3; slots=0
  C: running; cached CPU/issued=0/3; slots=6
  observed released: B
  reused B -> C: slots [14, 15, 16]
  free slots: 0/20
```

B 已释放，空闲槽位仍是 0，因为 C 已经拿走它们。[跟着 A、B、C 读代码](docs/WALKTHROUGH.md)，看清进度与缓存归属怎样变化。安装依赖见[运行指南](docs/START_HERE.md)；随机模型用于观察执行状态，输出文本不代表模型能力。

<details>
<summary>在 CPU 上看一次候选拒绝与缓存恢复</summary>

```bash
python -m minideepinfer.model_draft_demo --cpu-example rejection --trace-rounds --width 3 --output outputs/draft-rejection.json
```

draft 从“已算 6 个 → 保留 4 个 → 补到 5 个输入”接回正确前缀。[逐轮讲解](docs/MODEL_DRAFT_WALKTHROUGH.md)对照候选、实际模型输入和最终提交；这个例子也不需要权重或 NPU。

</details>

## 在 Ascend 910C 上生成文本

先按[运行指南](docs/START_HERE.md#ascend-910c真实权重)准备匹配的 PyTorch、torch_npu、CANN，以及本地 Qwen3.5-2B 权重。在仓库目录执行，以下多行命令使用 Bash：

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

[完整双语输出](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt)已保存。添加 `--pipeline` 可以让自己的文本走跨轮执行，再用 `--stagger-prompts --trace-rounds` 观察请求到达与缓存回收，见[文本接入说明](docs/PIPELINE_TEXT.md)。

### 双模型、流式输出与取消

main 的 `generate --speculative` 复用同一个候选验证与回退循环，支持聊天模板、raw 输入、EOS、逐组输出和协作取消。[使用说明](docs/SPECULATIVE_TEXT.md)包含完整参数、独立 greedy 对照和原始结果。

本页命令以 **main（0.5.0.dev0）** 为准。固定版本学习可使用 [v0.4.0](docs/RELEASE_v0.4.md)；该版本包含双模型逐轮实验，自己的文本、流式输出与取消需要 main。

<details>
<summary>运行一次流式生成，在提交组边界取消</summary>

准备本地 Qwen3.5-0.8B 和 2B 权重后执行：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B \
  --speculative --device npu:0 --dtype float32 \
  --prompt "The capital of France is" --raw \
  --max-new-tokens 10 --cache-capacity 128 --draft-width 3 \
  --stream --cancel-after-tokens 2 --check-greedy \
  --output outputs/stream-cancel.json
```

Ascend 910C 上的记录在完整交付下一组后停在 4 个 token，两边各归还全部 14 个槽位。取消门槛不会截断已经交付的接受组；这里的 greedy 检查只对照已交付前缀，不表示剩余输出或 EOS 已通过验证。

</details>

## 沿着代码学习

| 想读懂的机制 | 代码与教程 |
| --- | --- |
| 一次请求的 prefill、decode 与输出提交 | [`Engine.step()`](minideepinfer/engine.py) · [第一课](docs/FIRST_REQUEST.md) |
| 动态批处理、token 预算与分块 prefill | [`BatchEngine`](minideepinfer/batch_engine.py) · [第三课](docs/BATCHING.md) |
| Qwen3.5 的 KV、卷积窗口与递推状态 | [`HybridCache`](minideepinfer/hybrid_cache.py) · [第四课](docs/QWEN35.md) |
| 计算已发出后，读取、取消与回收的顺序 | [`DeferredRunner`](minideepinfer/deferred_runner.py) · [第七、八课](docs/DEFERRED_ENGINE.md) |
| CPU 尚未读到 token，下一轮怎样接着算 | [`StoppingDecodePipeline`](minideepinfer/stopping_pipeline.py) · [第十一至十四课](docs/CHUNK_PIPELINE.md) |
| 候选验证、接受、改正与两份缓存的回退 | [`verify_greedy()`](minideepinfer/speculative.py) / [`ModelDraft`](minideepinfer/model_draft.py) · [双模型逐轮演示](docs/MODEL_DRAFT_WALKTHROUGH.md) |

[完整学习路线](docs/LEARNING_PATH.md)保留十四课的运行顺序；[源码映射](docs/SOURCE_MAP.md)列出固定版本 DeepInfer 的对应入口与教学简化。同步、跨轮与投机参考分别保留自己的执行路径。

## 验证与实验

| 场景 | 已有记录 |
| --- | --- |
| 真实权重与模型计算 | Qwen3.5-2B / Ascend 910C FP32 的单请求、双请求与独立参考对照；见[验证记录](docs/VALIDATION.md) |
| 跨轮文本生成 | 同步、跨轮与逐条到达输出一致；EOS 轨迹的 43 组完整 logits 对照通过，最大差约 `4.94e-5`；见[文本检查](docs/PIPELINE_TEXT.md) |
| 双模型状态接续 | 实机宽度 1/3 的输出与独立 greedy 一致；CPU 自然拒绝例子展示 draft 截短与恢复；见[实现与实验](docs/MODEL_DRAFT.md) |
| 流式取消 | 实机已交付的 4 个 token 与独立 greedy 前缀一致，target / draft 各归还 14 个槽位；见[取消记录](docs/SPECULATIVE_TEXT.md#ascend-910c-检查) |
| 概率采样（CPU） | 小词表分布实验、双模型接受/补采样与冷缓存对照；无需权重，见[随机采样](docs/SAMPLING.md) |

每份报告保存输入、配置、原始结果和源码 SHA；历史结果按固定 commit 核对，见[证据说明](docs/EVIDENCE.md)。性能实验也保留了短负载更慢的情况：[同步与跨轮](docs/PIPELINE_BENCHMARK.md)、[首轮提交](docs/FIRST_RESULT.md)、[双模型成本](docs/MODEL_DRAFT.md)。接受率和减少的 forward 次数不能直接写成吞吐收益。

## 范围与来源

当前主例采用 Qwen3.5、FP32、eager 和 greedy，面向短上下文教学。BF16 的参考检查和分块差异见[说明](docs/BF16_REFERENCE.md)；原有 DeepSeek-R1-Distill-Qwen-1.5B 示例继续用于回归。DeepSeek-V4.1 模型架构、MLA/MoE、MTP、bounded replay、图执行和分布式仍在[后续路线](docs/ROADMAP.md)中。

CPU 另有正温度的[投机采样参考](docs/SAMPLING.md)，复用已有缓存回退；这一新路径尚未在 Ascend 910C 验证。

执行设计主要参考 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。[mini-sglang](https://github.com/sgl-project/mini-sglang)提供精简实现与教学组织参考，Ascend SGLang 提供 NPU 接口与兼容性参考；具体来源和许可证见[参考记录](docs/REFERENCES.md)。

欢迎 Star 收藏，也欢迎带着一条没跑通的命令、一处没讲清的状态变化[提交 Issue](https://github.com/AkiYangDev/mini-deepinfer/issues/new/choose)。代码和教程贡献见[贡献指南](CONTRIBUTING.md)。

本项目为独立社区项目，与 DeepSeek、SGLang 及华为无官方隶属关系。原创代码与文档采用 [Apache-2.0](LICENSE)，模型及第三方代码遵循各自许可证。
