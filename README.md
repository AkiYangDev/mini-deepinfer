<div align="center">

# mini-DeepInfer

**跟着一次真实请求，读懂 DeepInfer 的执行过程。**

A small inference engine for learning DeepInfer, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

**简体中文** · [English](README.en.md)

[开始运行](docs/START_HERE.md) · [十四课学习路线](docs/LEARNING_PATH.md) · [v0.1 说明](docs/RELEASE_v0.1.md) · [验证记录](docs/VALIDATION.md)

</div>

mini-DeepInfer 借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的设计，用一个小型推理引擎解释：prompt 怎样分块计算，decode 怎样接上前轮 token，KV Cache 和线性注意力状态由谁持有，以及请求结束后为什么还不能马上归还缓存。

项目已经用自己的前向计算，在 **Ascend 910C 上运行 Qwen3.5-2B 纯文本推理**。从无依赖的整数例子，到 PyTorch 小模型，再到真实权重，十四课沿着同一条请求执行路径展开。Qwen3.5-0.8B 用于开发，DeepSeek-R1-Distill-Qwen-1.5B 保留为回归基线。

## 先跑起来

Python 3.10+，先看一个无需模型依赖、权重或 NPU 的例子：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m pip install -e .
python -m minideepinfer.step_demo
```

A 取消了，为什么缓存还不能还？B 换了批次行号，怎样找到自己的上一轮结果？这个例子会打印状态变化，用手动事件解释依赖；它不执行模型。

已有 PyTorch 时，接着跑一个无需下载权重的混合注意力小模型：

```bash
python -m minideepinfer.verify_chunk_pipeline --device cpu
```

它把 decode 和分块 prompt 放进同一批，加入晚到的新请求，并与整段 prompt 的同步计算对照。输出包含每轮计划、误差和缓存回收结果。完整安装步骤与依赖分层见[运行指南](docs/START_HERE.md)。

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

## v0.1 能学到什么

| 你想弄清的问题 | 从这里读 |
| --- | --- |
| prefill、decode 每一步输入什么？ | [`Engine.step()`](minideepinfer/engine.py) · [第一课](docs/FIRST_REQUEST.md) |
| 多请求怎样合批，位置和缓存怎样隔离？ | [`BatchEngine`](minideepinfer/batch_engine.py) · [第三课](docs/BATCHING.md) |
| Qwen3.5 为什么需要两种缓存？ | [`HybridCache`](minideepinfer/hybrid_cache.py) · [第四课](docs/QWEN35.md) |
| 计算已经发出，取消后怎样收尾？ | [`DeferredRunner`](minideepinfer/deferred_runner.py) · [第七、八课](docs/DEFERRED_ENGINE.md) |
| CPU 没读到 token，下一轮怎样接着算？ | [`decode_pipeline.py`](minideepinfer/decode_pipeline.py) · [第十一课](docs/DECODE_PIPELINE.md) |
| EOS、新请求、分块 prompt 怎样进入同一个循环？ | [`StoppingDecodePipeline`](minideepinfer/stopping_pipeline.py) · [第十二至十四课](docs/CHUNK_PIPELINE.md) |
| 已发出、已提交和设备已用完，有什么区别？ | [`IssuedProgress` / `plan_step()`](minideepinfer/planning.py) · [第十四课](docs/CHUNK_PIPELINE.md) |

默认 BatchEngine 支持同步动态批处理、输入预算、分块 prefill、取消和回收；`--deferred` 把发起与结果提交拆开，仍只允许一批在途。跨轮设备 token 接续、回调到达和延迟回收属于独立教学实验，通过 `verify_chunk_pipeline` 运行，尚未接入 `generate`。各章节入口见[学习路线](docs/LEARNING_PATH.md)。

## 验证到了哪里

| 检查 | 已有证据 |
| --- | --- |
| 真实 2B / Ascend 910C FP32 | 单请求、双请求与独立参考对照通过，检查位置 top-1 一致 |
| 跨轮分块、到达与复用 | 九轮、十四组完整 logits 最大差约 `2.67e-5`，原容差通过；四个中间块不采样，24/24 槽位归还 |
| 取消与主机异常 | 等最后使用完成后清理，Runner 可复用；等待失败保留资源的分支另有 CPU 注入检查 |
| BF16 | 固定短轨迹通过同扫描方式、FP32 状态参考；原生分块与跨调度长生成仍有差异，见[说明](docs/BF16_REFERENCE.md) |

模型对照用于短上下文正确性检查。已有有限负载的时间线和计时没有证明通用提速；项目不承诺生产吞吐。实机原始结果、环境和源码 SHA 见[验证记录](docs/VALIDATION.md)，v0.1 的对应源码可用 `python scripts/check_evidence.py` 核对，无需再次启动 NPU。

## 为什么参考 DeepInfer

DeepInfer 是引擎设计主线：输入准备、主模型执行、采样、结果提交以及跨轮状态有效期，逐步落成可运行的代码。[源码映射](docs/SOURCE_MAP.md)列出固定上游版本、入口和简化；Qwen 承载模型实例，不把 DeltaNet 称为 DeepInfer 独有机制。

[mini-sglang](https://github.com/sgl-project/mini-sglang)提供精简实现和教学组织的参考，Ascend SGLang 提供 NPU 接口与兼容性参考。MLA、MoE、投机解码、bounded replay、图执行和分布式留在[后续路线](docs/ROADMAP.md)。来源与许可证见[参考记录](docs/REFERENCES.md)。

欢迎带着一个具体问题[提交 Issue](https://github.com/AkiYangDev/mini-deepinfer/issues)：哪一步看不懂，哪条命令没跑通，或哪项缓存行为需要补例子。也欢迎按[贡献指南](CONTRIBUTING.md)改进代码和教程。如果这条学习路线对你有用，欢迎 Star。

本项目为独立社区项目，与 DeepSeek、SGLang 及华为无官方隶属关系。原创代码与文档采用 [Apache-2.0](LICENSE)，模型及第三方代码遵循各自许可证。
