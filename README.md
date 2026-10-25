<div align="center">

# ![mini-DeepInfer](docs/assets/readme-hero-whale.png)

**读懂 DeepInfer，从一条请求开始。**

一个面向 Ascend 910C 的精简教学实现，使用真实 Qwen3.5 权重。

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Ascend 910C](https://img.shields.io/badge/Ascend%20910C-FP32%20checked-0f766e)](docs/VALIDATION.md) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-2563eb)](pyproject.toml) [![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-475569)](LICENSE)

**简体中文** · [English](README.en.md)

[快速开始](#快速开始) · [Ascend 910C 实机](#在-ascend-910c-上生成文本) · [投机解码](#投机解码与随机采样) · [十四课](docs/LEARNING_PATH.md) · [DeepInfer 源码对应](docs/SOURCE_MAP.md)

</div>

<img align="right" src="docs/assets/deepseek-girl-full.png" width="126" alt="蓝白裙装、鲸尾的 DeepSeek 娘角色插画" />

mini-DeepInfer 是一个以 **[DeepInfer](https://github.com/deepseek-ai/DeepInfer) 执行设计**为主线的教学推理引擎。它用精简的 Python 与 PyTorch 实现，把请求调度、模型执行、采样和结果提交串起来，让你能顺着一条请求读懂这些环节怎样配合。

如果你已经会部署模型，想进一步读 DeepInfer 的源码，可以从这里开始。[十四课](docs/LEARNING_PATH.md)从最小的请求循环写起，逐步加入批处理、分块 prefill 和跨轮执行。重点看计算发出后的状态：设备还在用哪些数据，CPU 什么时候提交结果，请求取消后缓存何时才能归还。

[DeepInfer 源码映射](docs/SOURCE_MAP.md)列出原版入口、这里的对应实现和教学简化。读完调度与在途状态，再看[双模型逐轮演示](docs/MODEL_DRAFT_WALKTHROUGH.md)，跟着一次候选拒绝，检查输出提交和缓存恢复怎样接上下一轮。

Ascend 910C 是这个项目的实机平台，Qwen3.5 用来验证这些机制。没有 NPU，也能先跑 CPU 调度和随机小模型；准备好设备与权重后，再运行真实文本生成，让 0.8B 提候选、2B 做验证。

<br clear="both" />

## 快速开始

有 Python 3.10+ 就能运行下面的调度例子，不需要安装依赖或下载权重：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

例子里 A 已取消，设备却还在使用它的缓存；B 换了批次行号，也必须接上自己的上一轮结果。手动事件把“发出计算、提交结果、归还缓存”分开打印出来。

按教程学习，可以使用固定版本 [v0.5.0](docs/RELEASE_v0.5.md)，[运行指南](docs/START_HERE.md)也基于这个 tag。它包含双模型文本生成、流式输出、取消和温度、top-k / top-p 采样。

main 还提供[单模型入口 `generate --single`](docs/SINGLE_MODEL.md)：只加载一份 Qwen3.5，也能采样、流式输出和取消。这个入口属于 v0.6 开发版，运行前请确认使用的是 main。

安装 PyTorch 后，再跑一个随机初始化的小型 Qwen3.5，看看真实模型计算怎样接入调度：

```bash
python -m minideepinfer.pipeline_demo
```

![CPU 实际输出：B 结束后，C 已复用它的缓存槽位](docs/assets/cache-reuse.svg)

这段输出里，B 刚结束，C 就接过了它的缓存槽位，所以空闲槽位仍是 0；A 的长 prompt 还在分块计算。跟着 [A、B、C 读代码](docs/WALKTHROUGH.md)，可以把每次调度与缓存变化对上，也可以直接查看[完整终端输出](docs/assets/pipeline-demo.txt)。这个模型没有训练，重点看执行过程。

## 接下来读哪里

| 你想先做什么 | 从这里开始 |
| --- | --- |
| 没有 NPU，先看调度和缓存 | [CPU 入门](docs/START_HERE.md) → [A/B/C 演示](docs/WALKTHROUGH.md) |
| 有 Ascend 910C，运行自己的问题 | [准备环境与权重](docs/START_HERE.md#ascend-910c真实权重) → [文本生成](#在-ascend-910c-上生成文本) |
| 看小模型猜错后怎样恢复 | [双模型逐轮演示](docs/MODEL_DRAFT_WALKTHROUGH.md) → [概率采样](docs/SAMPLING.md) |

[课程目录](docs/LEARNING_PATH.md)列出了完整顺序、运行命令和代码位置，按自己的进度往下读即可。

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

换成自己的问题后，可以一边生成一边看输出，也可以中途请求取消，[文本入口](docs/SPECULATIVE_TEXT.md)有完整命令。想理解随机采样时为什么还能接受候选，就从[小词表实验](docs/SAMPLING.md)往下读，再看 Ascend 910C 上的概率与状态恢复记录。这条参考路径使用 NPU FP32 计算模型、CPU FP64 计算概率。

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

## 对照 DeepInfer 读代码

先从自己关心的问题进入，再回到 DeepInfer 的对应职责：引擎组织一轮执行，Runner 准备输入和采样，在途状态记录尚未收取的计算，结果处理决定哪些输出与进度可以提交。[源码映射](docs/SOURCE_MAP.md)把这些关系逐项列出，下面是教学版的阅读入口。

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

这些记录都保留了输入、配置、原始输出和源码 SHA，可以[按对应 commit 复核](docs/EVIDENCE.md)。性能实验另见[同步与跨轮](docs/PIPELINE_BENCHMARK.md)、[首轮提交](docs/FIRST_RESULT.md)和[双模型成本](docs/MODEL_DRAFT.md)。其中也有短负载变慢的结果：多接受几个 token、少做几次 forward，并不一定让整个请求更快，具体要看恢复、复制和等待花了多少时间。

## 范围与来源

当前主线是单设备、FP32、eager 和短上下文，适合学习与实验。greedy、温度与 top-k / top-p 采样、流式输出和取消均已实现；BF16 的参考检查与分块差异见[说明](docs/BF16_REFERENCE.md)，旧 DeepSeek-R1-Distill-Qwen-1.5B 示例继续回归。DeepSeek-V4.1 架构、MLA/MoE、MTP、bounded replay、图执行和分布式尚未实现，见[后续路线](docs/ROADMAP.md)。

执行设计主要参考 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)，[mini-sglang](https://github.com/sgl-project/mini-sglang)提供精简实现与教学组织参考，Ascend SGLang 提供 NPU 接口参考。[来源与许可证](docs/REFERENCES.md)逐项记录。

如果你也在学推理引擎，欢迎 Star，留着边跑边读。哪条命令跑不通、哪段代码还没讲清楚，都可以[开一个 Issue](https://github.com/AkiYangDev/mini-deepinfer/issues/new/choose)。想补例子或改教程，可以先看[贡献指南](CONTRIBUTING.md)。

独立社区项目，与 DeepSeek、SGLang、华为无官方隶属关系。原创代码和文档采用 [Apache-2.0](LICENSE)，模型与第三方代码遵循各自许可证。角色图片另见[素材说明](docs/assets/ARTWORK.md)。
