# 参考、来源与归属

[返回首页](../README.md)

更新时间：2026-10-24。本文记录设计与代码来源。引擎为原创教学实现，未直接引入上游 DeepInfer 或 SGLang 引擎源码；Qwen2 数学实现的参考与改写另列如下。

## DeepInfer

- 仓库：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- 本轮源码参考提交：[8b16149741a4935e1dec8cea27853f273b374c5b](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b)
- 上游许可证：[Apache-2.0](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/LICENSE)

| 原版文件 | 已核对的设计点 | 教学版计划 |
| --- | --- | --- |
| [llm_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py) | 引擎协调请求、调度、执行和结果处理；调度器来自 C++ 扩展 | 用可读的 Python 状态转换建立同步执行循环 |
| [model_runner.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py) | 输入元数据、模型执行与图执行选择 | 显式 StepPlan 和 Runner，逐步接入优化 |
| [cache_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/cache_engine.py) | 设备缓存布局、分配及模型相关状态 | 区分设备缓存管理与完整服务级缓存系统 |
| [zos_status.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/zos_status.py) | 步骤状态、事件和批次重映射元数据 | 同步路径验证后，教学化展示在途步骤生命周期 |
| [worker.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py) | 本轮 forward、上一轮 post-process、本轮 sample；设备上的前轮 token 经 MetadataZOS 传递 | [固定批次实验](DECODE_PIPELINE.md)先学习设备输入依赖和处理顺序，再扩展动态调度 |
| 同一 Worker / WorkerStepState 与 llm_engine.py 的 DeferredRelease | 结果与资源释放分开，步骤对象有效期覆盖后继使用者 | [成员退出实验](STOPPING_PIPELINE.md)使用每请求最后使用事件，学习 EOS/取消后的丢弃与回收；不是原版完整取消实现 |
| MetadataZOS 的 remap_cpu/remap 与 prev_sampled_token_ids | CPU 区分新请求和前轮行，保留设备采样来源 | [到达实验](ARRIVALS.md)用独立的 FIFO 准入和混合输入准备解释来源区别，非原版完整调度策略 |
| [platform.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/platform.py) | 昇腾平台、HCCL、主机内存与 CUDA 风格接口适配 | 优先明确的 NPU 接口，按 Ascend 910C 实测选择后端 |
| [YoCo 等价性实验](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/yoco/README.md) | 元数据与真实模型输出分层验证 | 保留模型前提，先构造独立等价性实验 |

该版本 README 声明的昇腾硬件是 Ascend 950。mini-DeepInfer 的 Ascend 910C 支持将通过本项目实测建立；上游支持范围不能直接证明本项目的兼容性。

## mini-sglang 与 SGLang

- [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)：参考其精简模块、教学入口和 README 的信息组织。上游采用 MIT 许可证。
- [sgl-project/sglang](https://github.com/sgl-project/sglang)：作为服务机制、实现和后续基准的参考。
- [Ascend/sglang，release/2026930](https://github.com/Ascend/sglang/commits/release/2026930)：用户提供的昇腾实现参考分支。本轮锁定 [545ed1c6649321aa61b57a287032a7791690247e](https://github.com/Ascend/sglang/tree/545ed1c6649321aa61b57a287032a7791690247e)，重点核对 attention、MLA、MoE 和图执行接口；实际平台与算子兼容性仍需本项目验证。

首页参考 mini-sglang 对定位、学习价值、核心机制和使用入口的组织方式，使用本项目自己的文案和实际进度。上游的性能、代码量与平台支持不作为本项目的宣传数据。

## 官方部署报告

[DeepSeek V4.1 Flash 大规模推理部署实践](https://zhuanlan.zhihu.com/p/2095560295930508953)，DeepSeek 已认证机构号，2026-10-20。本轮通过浏览器读取正文。

报告补充了缓存存储、负载均衡和部署取舍的背景。其规模与性能数据不直接作为本项目的实验指标；尤其应区分 bounded replay 的近似恢复与 mask/compact 两种实现之间的等价性检查。来源到阶段交付的具体映射见[从设计参考到第一段实现](SOURCE_MAP.md)。

## NPU 事件接口

第六课的设备接口对照 [TorchNPU 26.0 / PyTorch 2.10.0 支持表](https://www.hiascend.com/document/detail/en/Pytorch/2600/apiref/PyTorchNativeapi/docs/en/native_apis/pytorch_2-10-0/torch-cuda.md)、[Ascend/pytorch v2.10.0 的 streams.py](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/streams.py)与 [NPUEvent::query()](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUEvent.cpp#L77)。实际安装版本为 torch_npu 2.10.0.post6，按[执行记录](results/2026-10-24/npu-step-events.json)绑定源码 SHA；`RecordedEvent` / `PendingTokens` 为本项目的教学实现，未复制上游事件实现代码。微型张量实验不代表模型执行重叠已验证，细节见[第六课](NPU_EVENTS.md)。

## 模型与协议

[`models/qwen3_5.py`](../minideepinfer/models/qwen3_5.py) 对照 [Transformers v5.12.1 的 Qwen3.5 实现](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py) 改写纯文本计算：保留零中心 RMSNorm、部分 RoPE、带 gate 的完整注意力、Gated DeltaNet 和 checkpoint 参数映射；将 DeltaNet 展开为逐请求、逐 token 的参考递推。保留 Qwen / Hugging Face 版权归属，适用 Apache-2.0。省去视觉、训练、MTP 和优化内核，缓存由引擎显式管理。

Qwen3.5 权重固定到 [0.8B / 2fc06364](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) 与 [2B / 15852e8c](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)。0.8B 用于开发验证，2B 为主要示例；运行范围见[第四课](QWEN35.md)与[验证记录](VALIDATION.md)。

验证专用的 [`reference_qwen3_5.py`](../minideepinfer/reference_qwen3_5.py) 调用 Transformers v5.12.1 的 `torch_recurrent_gated_delta_rule`，独立控制扫描方式和 FP32 状态存储。缓存差异对照 [v5.12.1 cache_utils.py](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/cache_utils.py) 中的 `LinearAttentionLayer.lazy_initialization()` / `update_recurrent_state()`。适配器不参与引擎执行，原生参考模式仍保留，详见 [BF16 对照说明](BF16_REFERENCE.md)。

[`minideepinfer/models/qwen2.py`](../minideepinfer/models/qwen2.py) 对照 [Transformers v4.46.3 的 Qwen2 实现](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py) 实现数学计算和 checkpoint 参数映射，其中 RMSNorm 与旋转位置编码公式按上游简化改写。保留 Qwen / Alibaba / Hugging Face 的版权归属，适用 [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.46.3/LICENSE)。本项目采用显式 attention 和外部槽位缓存，批次通过隔离 mask 支持多个请求；省去训练、分布式、动态 RoPE、滑窗与多模型兼容分支。

入门权重固定到 [ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562)。运行时的模型计算使用本项目实现；Transformers 用于 tokenizer 和测试参考。

- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)：Qwen 架构入门模型。
- [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) / [V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)：MLA 与 MoE 教学主线。
- [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)：未来接入相应 DeepSeek 模型协议时的候选组件；其模型范围与引擎执行能力分别核对。

模型权重和 tokenizer 使用对应模型仓库的许可证与说明。模型架构、机制实验、完整服务支持分别标明验证范围。

## 引入代码时的记录规则

正温度的 [CPU 投机采样](SAMPLING.md)使用 [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) Algorithm 1 的接受与补采样规则。实现为原创教学代码，未复制论文实现或第三方采样代码；执行与资源归属继续参考上文固定 DeepInfer 版本。

直接复用或改写上游代码时，记录项目、commit、原文件路径和修改目的，保留适用的版权/许可证及必要 NOTICE。第三方代码引入后，在对应目录附带来源与许可证信息。

仅借鉴概念时，也在教学章节说明来源、与原版的对应关系和简化条件。当前项目的 Apache-2.0 许可不改变模型及第三方组件的许可。

第十四课的[跨轮分块与进度划分](CHUNK_PIPELINE.md)继续参考同一固定 DeepInfer 版本的 Worker 顺序、在途状态与 prompt 采样边界。共用 Step 规划函数、IssuedProgress、输入预算和每请求最后使用事件为本项目的教学实现，未引入新的上游代码或许可证。

v0.3 的[可控 greedy 投机实验](SPECULATIVE.md)重新核对同一 commit 的 `deepinfer/model_executor/layers/sampler.py` 中 `_verification()`、`deepinfer/engine/llm_engine.py` 的 `_process_model_outputs()` 和 Worker 的 `_correct_seqlens_qk()`。保留验证、接受前缀与最终输出、后续有效长度三个学习问题；教学代码为原创实现，不复制上游 sampler、MTP、C++ 调度或概率接受算法。纯注意力有效 KV 回退不代表上游采用同一种存储操作。
