<div align="center">

# ![mini-DeepInfer](docs/assets/readme-hero-whale.png)

**读懂 DeepInfer 的执行设计，跑通 Ascend 910C 上的真实推理。**

从三条 CPU 命令，到 Qwen3.5 双模型投机解码。

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Ascend 910C](https://img.shields.io/badge/Ascend%20910C-FP32%20checked-0f766e)](docs/VALIDATION.md) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-2563eb)](pyproject.toml) [![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-475569)](LICENSE)

**简体中文** · [English](README.en.md)

[快速开始](#快速开始) · [Ascend 910C 实机](#在-ascend-910c-上生成文本) · [投机解码](#投机解码与随机采样) · [十四课](docs/LEARNING_PATH.md) · [DeepInfer 源码对应](docs/SOURCE_MAP.md)

</div>

mini-DeepInfer 是一个面向 **Ascend 910C** 的精简推理引擎。我们借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的执行设计，用 Python 和 PyTorch 实现请求调度、模型计算、输出提交与缓存管理，再用真实 Qwen3.5 权重把它们跑通。

你可以沿着一条请求读完整个过程，也可以修改调度、打断生成，看看引擎如何继续工作。

## 值得动手的四件事

<img align="right" src="docs/assets/deepseek-girl-full.png" width="126" alt="蓝白裙装、鲸尾的 DeepSeek 娘角色插画" />

- **在 Ascend 910C 上跑真实模型。** Qwen3.5-2B 已跑通文本生成、动态批处理与跨轮执行，命令和原始输出[直接可查](docs/VALIDATION.md)。
- **看清每一步调度。** 长 prompt 怎样分块，新请求怎样插入批次，CPU 还没读到上一轮 token 时下一轮怎样继续；跟着 [A、B、C 三条请求](docs/WALKTHROUGH.md)观察缓存被谁持有、何时复用。
- **把双模型投机解码走完整。** Qwen3.5-0.8B 提候选、2B 验证，包含拒绝回退、线性状态恢复、随机采样、逐组流式输出与取消。[从一次猜错开始读](docs/MODEL_DRAFT_WALKTHROUGH.md)。
- **从能运行的代码学起。** 三条命令启动 CPU 调度演示，无需依赖和权重。[十四课](docs/LEARNING_PATH.md)带你逐步加入批处理、分块 prefill 和跨轮执行，并对照 [DeepInfer 源码](docs/SOURCE_MAP.md)理解设计取舍。

<br clear="both" />

## 快速开始

先让引擎跑起来。这个调度演示只用 Python 3.10+ 标准库，不装依赖，不下权重：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

例子里 A 已取消，设备却还在使用它的缓存；B 换了批次行号，也必须接上自己的上一轮结果。手动事件把“发出计算、提交结果、归还缓存”分开打印出来。

固定学习版本为 **[v0.5.0](docs/RELEASE_v0.5.md)**，包含自己的投机文本、流式取消、温度与 top-k / top-p 采样。[运行指南](docs/START_HERE.md)使用同一个 tag；旧版本与报告继续保留。

**想跑真实 forward？** 已有 PyTorch 时，继续运行无需权重的随机 Qwen3.5 小模型：

```bash
python -m minideepinfer.pipeline_demo
```

![CPU 实际输出：B 结束后，C 已复用它的缓存槽位](docs/assets/cache-reuse.svg)

B 已释放，空闲槽位为什么仍是 0？因为 C 已经接过了它们。A 的长 prompt 还在分块执行，三个请求的进度各不相同。[跟着 A、B、C 读代码](docs/WALKTHROUGH.md)，或查看[完整终端输出](docs/assets/pipeline-demo.txt)。随机模型用于观察执行过程，输出文本不代表回答能力。

## 选一条路线，动手学

| 你想先做什么 | 从这里开始 |
| --- | --- |
| 没有 NPU，先看调度和缓存 | [CPU 入门](docs/START_HERE.md) → [A/B/C 演示](docs/WALKTHROUGH.md) |
| 有 Ascend 910C，运行自己的问题 | [准备环境与权重](docs/START_HERE.md#ascend-910c真实权重) → [文本生成](#在-ascend-910c-上生成文本) |
| 看小模型猜错后怎样恢复 | [双模型逐轮演示](docs/MODEL_DRAFT_WALKTHROUGH.md) → [概率采样](docs/SAMPLING.md) |

每条路线都有可运行的例子。[课程目录](docs/LEARNING_PATH.md)列出了每一课的命令与代码位置。

<details>
<summary>先在 CPU 上看一次候选拒绝与缓存恢复</summary>

```bash
python -m minideepinfer.model_draft_demo --cpu-example rejection --trace-rounds --width 3 --output outputs/draft-rejection.json
```

draft 从“已算 6 个 → 保留 4 个 → 补到 5 个输入”接回正确前缀。[逐轮讲解](docs/MODEL_DRAFT_WALKTHROUGH.md)对照候选、实际模型输入和最终提交；这个例子不需要权重或 NPU。

</details>

## 在 Ascend 910C 上生成文本

Qwen3.5-2B 已通过真实权重的单请求、批处理与跨轮生成检查。先按[运行指南](docs/START_HERE.md#ascend-910c真实权重)准备匹配的 PyTorch、torch_npu、CANN 和本地权重，再从仓库目录运行。多行命令使用 Bash：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

真实运行中的英文回答：

```text
1 plus 1 equals 2.

[eos; 9 output tokens; free KV slots 512/512]
```

[完整双语输出](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt)已保存。添加 `--pipeline` 让文本走跨轮执行，再用 `--stagger-prompts --trace-rounds` 观察请求到达与缓存回收，见[文本接入说明](docs/PIPELINE_TEXT.md)。

## 投机解码与随机采样

让 Qwen3.5-0.8B 先猜，再由 2B 验证。接受时一次交付多个 token；拒绝时丢弃临时结果，恢复两份缓存，再从正确前缀继续。KV、卷积窗口与递推状态都要一起处理，才能让下一轮模型计算接得上。

直接输入自己的问题，观察逐组流式输出，或在提交后取消生成，[文本入口](docs/SPECULATIVE_TEXT.md)给出完整命令。正温度采样复用同一循环，包含概率接受和拒绝后的补采样；[采样教程](docs/SAMPLING.md)从小词表实验讲到 Ascend 910C 的数值与状态恢复验证。模型用 NPU FP32 计算，概率保留在 CPU FP64。

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

实机记录在完整交付下一组后停在 4 个 token，两边各归还 14 个槽位。取消门槛不会截断已经提交的接受组；greedy 检查只对照已交付前缀，不表示剩余输出或 EOS 已验证。随机采样不能同时使用 `--check-greedy`。

<p align="center"><img src="docs/assets/deepseek-girl-chibi.jpg" width="96" alt="眼睛发光的 Q 版大肥鱼小彩蛋" /></p>

</details>

## 从问题找到代码

| 想读懂的机制 | 代码与教程 |
| --- | --- |
| 一次请求的 prefill、decode 与输出提交 | [`Engine.step()`](minideepinfer/engine.py) · [第一课](docs/FIRST_REQUEST.md) |
| 合批、token 预算与分块 prefill | [`BatchEngine`](minideepinfer/batch_engine.py) · [第三课](docs/BATCHING.md) |
| KV、卷积窗口与递推状态 | [`HybridCache`](minideepinfer/hybrid_cache.py) · [Qwen3.5 模型](docs/QWEN35.md) |
| CPU 尚未读到 token，下一轮怎样接着算 | [`StoppingDecodePipeline`](minideepinfer/stopping_pipeline.py) · [跨轮与分块](docs/CHUNK_PIPELINE.md) |
| 候选拒绝后，两份缓存怎样恢复 | [`ModelDraft`](minideepinfer/model_draft.py) · [双模型演示](docs/MODEL_DRAFT_WALKTHROUGH.md) |
| 概率接受与拒绝后的补采样 | [`verify_sampled()`](minideepinfer/sampling.py) · [采样实验](docs/SAMPLING.md) |

[十四课](docs/LEARNING_PATH.md)给出完整顺序；[DeepInfer 源码映射](docs/SOURCE_MAP.md)标明保留的机制、对应入口和教学简化。

## Ascend 910C 上，已经跑通什么

| 已有记录 | 验证了什么 |
| --- | --- |
| [Qwen3.5-2B 实机](docs/VALIDATION.md) | Ascend 910C FP32 的模型计算、单请求与批处理 |
| [跨轮文本](docs/PIPELINE_TEXT.md) | 同步、跨轮和逐条到达输出一致；43 组完整 logits 对照通过 |
| [投机采样](docs/SAMPLING.md#ascend-910c验证概率和回退) | 28 组概率、6 组冷缓存对照通过，真实覆盖 draft 拒绝恢复 |
| [流式取消](docs/SPECULATIVE_TEXT.md#ascend-910c-检查) | 已交付的 4 个 token 匹配独立 greedy 前缀，两份缓存均归还 |

每项结果都附输入、配置、原始记录与源码 SHA，可以[按固定 commit 复核](docs/EVIDENCE.md)。想继续研究性能，可以从[同步与跨轮](docs/PIPELINE_BENCHMARK.md)、[首轮提交](docs/FIRST_RESULT.md)和[双模型成本](docs/MODEL_DRAFT.md)入手：这些记录包含短负载变慢的情况，方便理解优化何时有效。接受率和 forward 次数需要结合实际耗时判断。

## 范围与来源

项目保持单设备、FP32、eager、短上下文的教学范围。greedy 和正温度投机采样已有实现；BF16 的参考检查与分块差异见[说明](docs/BF16_REFERENCE.md)，旧 DeepSeek-R1-Distill-Qwen-1.5B 示例继续回归。DeepSeek-V4.1 架构、MLA/MoE、MTP、bounded replay、图执行和分布式仍在[后续路线](docs/ROADMAP.md)中。

执行设计主要参考 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)，[mini-sglang](https://github.com/sgl-project/mini-sglang)提供精简实现与教学组织参考，Ascend SGLang 提供 NPU 接口参考。[来源与许可证](docs/REFERENCES.md)逐项记录。

**在 Ascend 910C 上学推理引擎，从这三条命令开始。** 欢迎 Star，留着对照源码；也欢迎把跑不通的命令或想弄明白的机制[写进 Issue](https://github.com/AkiYangDev/mini-deepinfer/issues/new/choose)。想动手补一个机制、改一课教程，从[贡献指南](CONTRIBUTING.md)加入。

独立社区项目，与 DeepSeek、SGLang、华为无官方隶属关系。原创代码和文档采用 [Apache-2.0](LICENSE)，模型与第三方代码遵循各自许可证。角色图片另见[素材说明](docs/assets/ARTWORK.md)。
