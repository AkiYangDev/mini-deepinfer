# 从设计参考到第一段实现

[返回首页](../README.md) · [开发路线](ROADMAP.md) · [来源记录](REFERENCES.md)

更新时间：2026-10-24。本文把部署报告、DeepInfer 和 Ascend SGLang 的不同参考作用对应到项目交付；源码阅读不等于 Ascend 910C 兼容性验证。

## 三类参考，各自解决一个问题

| 参考 | 主要作用 | 使用方式 |
| --- | --- | --- |
| [DeepSeek V4.1 Flash 大规模推理部署实践](https://zhuanlan.zhihu.com/p/2095560295930508953) | 理解部署目标、缓存生命周期、负载均衡与算法取舍 | 为机制实验提供假设和背景；区分部署经验与普遍成立的结论 |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b) | 理解引擎、执行器、输入元数据和在途状态的边界 | 建立原版与教学版对应关系，保留简化说明 |
| [Ascend SGLang：release/2026930](https://github.com/Ascend/sglang/tree/545ed1c6649321aa61b57a287032a7791690247e) | 理解 NPU 运算接口、缓存布局、stream/event 和图执行 | 提取适用的最小接口，在目标软件/硬件环境独立验证 |

mini-sglang 继续作为紧凑实现和教学组织的参考。mini-DeepInfer 自己实现请求循环、调度与缓存；后端接口可借鉴成熟项目，验证条件随代码一同记录。

## DeepInfer 主线与当前骨架

Qwen3.5 继续使用下面相同的执行职责。新增 [`HybridCache`](../minideepinfer/hybrid_cache.py) 将完整注意力 KV 与线性层状态分开，状态由请求拥有并在同步执行完成后回收。[第四课](QWEN35.md) 对照 Qwen / Transformers 的模型计算，不把 DeltaNet 当成 DeepInfer 独有机制。

技术设计以固定版本 DeepInfer 为主要参考。下面是概念与职责映射，当前骨架为原创实现，不是对原版的逐行缩写，也尚未复现其特色优化。

| DeepInfer 源码入口 | 当前教学版落点 | 保留与简化 |
| --- | --- | --- |
| [`LLMEngine.step()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L784) | [`Engine.step()`](../minideepinfer/engine.py) | 保留调度后执行的主循环；原版调用 C++ Scheduler 与 worker，当前只有 Python FIFO 和一个运行请求 |
| 同一入口中的 `scheduler.schedule()`、有序的 sequence metadata 与 worker 调用 | [`BatchEngine._schedule()` / `step()`](../minideepinfer/batch_engine.py)、[`ModelRunner.prepare_batch()` / `run_batch()`](../minideepinfer/model_runner.py) | 第三课保留逐请求元数据与输出行对应，一轮合并成一次 forward；Python FIFO、全量预留、显式隔离 mask 和混合 prefill/decode 是教学选择，不复现原版完整调度策略 |
| `LLMEngine.step()` 中的 `is_last_prompt_chunk`，以及 [`ModelRunner.prepare_sampler_inputs()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py#L213) | `Step.is_last_prompt_chunk`、Runner 中间块不采样、BatchEngine 分开提交缓存进度与输出 token | 保留 prompt 完成与采样的边界；decode 优先和剩余预算切块为本项目的简单调度策略，不复现上游完整 C++ 实现 |
| [`ModelRunner._prepare_input_tensors()` / `execute_main_model()` / `execute_sample()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py#L110) | [`ModelRunner.prepare_input()` / `execute_main_model()` / `execute_sample()`](../minideepinfer/model_runner.py) | 显式传递输入、位置和缓存映射，分开准备、forward 与 greedy；当前 `run()` 等待设备完成后才返回，尚无图执行或调度重叠 |
| [`LLMEngine._process_model_outputs()` / `post_process()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L624) | Runner 返回后的输出与请求状态提交 | 原版通过 scheduler 处理结果；当前直接提交一个 token，后续扩展批次和投机接受/回退时保留明确提交边界 |
| [`BlockManager` 与 `CacheEngine`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/cache_engine.py) | [`SlotPool`](../minideepinfer/kv_cache.py) | 当前只演示逻辑归属和回收，不等同于原版的块管理、设备存储或模型状态布局 |
| [`WorkerStepState` / `MetadataZOS`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/zos_status.py) 与 [`DeferredRelease`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L64) | [`step_state.py`](../minideepinfer/step_state.py)，[第五课](STEP_STATE.md)独立 CPU 实验 | 保留按请求映射旧输出、跨轮状态和等待使用者完成后释放的概念；手动事件、整数结果和整批逻辑槽位保护，不复现设备映射内核或上游各类池，尚未接入 NPU/主引擎 |

FIFO、prefill/decode 和槽位分配是推理框架的通用基础，单独实现它们还不足以展示 DeepInfer 的特点。后续每项教学实现都需要给出原版入口、保留的机制、简化理由和相应验证，逐步进入执行/采样分工、在途状态管理、投机提交/回退和 bounded replay。

入门阶段的 Qwen 小模型用于验证真实 forward 与缓存循环，是本项目的教学选择；它不代表已经复现 DeepInfer 所支持模型的计算。MLA/MoE 与 V4.1 特定机制仍按各自模型前提讲解。

## Ascend SGLang 阅读入口

本轮固定提交：`545ed1c6649321aa61b57a287032a7791690247e`。分支名会继续移动，后续实现不能仅依赖分支名复现。

以下路径均相对于 `python/sglang/srt/hardware_backend/npu/`：

| 路径 | 本轮已观察到的内容 | 对应阶段 |
| --- | --- | --- |
| `attention/ascend_torch_native_backend.py` | 参考 attention 计算及 extend/decode 的输入、缓存读取路径 | M0—M1：先构造清楚的参考计算 |
| `attention/ascend_backend.py` | `npu_fused_infer_attention_score` 与 v2 调用，以及序列长度、块表等元数据 | M0 探测、M2/M4 优化：逐项核对 dtype、shape、layout、mask 和缓存格式 |
| `modules/deepseek_v2_attention_mla_npu.py` | RoPE、KV 归一化/写入、矩阵变换和多流依赖 | M3：对照 MLA 参考实现；融合和多流随后加入 |
| `moe/topk.py` | 不同路由算子入口 | M3：核对路由语义与返回格式 |
| `moe/init_routing.py` | token 分组、专家 token 计数及不同版本入口 | M3：从逐专家参考计算过渡到分组计算 |
| `moe/matmul.py` | grouped matmul 等专家计算入口 | M3/M4：验证 BF16 专家计算与工作区要求 |
| `graph_runner/npu_graph_runner.py` | `NPUGraph`、capture 和动态输入更新相关逻辑 | M4：在 eager 路径通过后加入图执行 |

算子在分支中出现，不代表当前 CANN/`torch_npu` 或 Ascend 910C 上的所有形状都可用。兼容性记录应绑定实际调用参数和环境；不要一次复制生产后端的全部分支。

## 报告对教学路线的具体启发

- **先解释 prefill 与 decode 的负载差异。** 教学示例从单设备开始，跨机部署留到有相应实验条件的阶段。
- **将 prefill 的 token 预算与时间预算分开。** 基础调度稳定后，可用受控代价模型演示“相同 token 数不等于相同计算代价”；真实模型系数需要测量。
- **缓存按生命周期解释。** 先实现当前模型需要的设备缓存，再讨论前缀复用、状态缓存和持久化存储；这些对象的布局和复用条件取决于模型。
- **把异步调度画成依赖时间线。** 下一轮调度、上一轮输出处理和当前设备计算之间的依赖必须显式表达。
- **投机验证长度属于调度决策。** 先讲清提交与回退，再考虑置信度和吞吐曲线驱动的预算分配；模型头与权重兼容性另行验证。

报告中的规模、带宽、并行配比和吞吐改善来自其部署条件，不直接成为 Ascend 910C 教学项目的配置或性能目标。

## bounded replay：两种对照不能混用

报告明确将 SWA bounded replay 描述为通过有限回放近似恢复 SWA 状态。教学中需要区分：

1. **同一种 bounded replay 语义的不同实现。** 例如上游 YoCo 实验中的 mask 与 compact，检查元数据和输出是否一致。
2. **bounded replay 与完整重算。** 检查恢复状态、后续 logits 和必要的任务质量差异，同时测量节省的计算；不预设严格等价。

mask/compact 对照通过，不能证明近似回放与完整重算严格等价。V4.1 的模型前提也不能直接套用到 V2-Lite。

## 当前进度与下一项交付

同步执行与真实模型路径已落地：第一课保留 [`Engine.step()`](../minideepinfer/engine.py)，第三课增加 [`BatchEngine`](../minideepinfer/batch_engine.py) 与真正合并计算的多请求 Runner。阅读入口见[第一课](FIRST_REQUEST.md)、[第二课](REAL_GENERATION.md)和[第三课](BATCHING.md)，验证范围见[实验记录](VALIDATION.md)。当前没有调度重叠。

Qwen3.5 的 CPU 与 Ascend 910C FP32 路径、BF16 的固定短轨迹参考检查已经记录；原生 BF16 分块与长生成差异继续保留。[第五课](STEP_STATE.md)新增独立的跨轮状态实验，解释按请求映射旧输出，以及 CPU 和设备全部用完后才回收。

下一项交付是实际 NPU 事件与张量生命周期验证，然后才把在途状态接到真实 Runner。硬件执行前检查资源占用，复用已有权重和数值基线；同步和新路径的结果、取消与异常回收需要分别验证。设备重叠与性能收益以实际时间线和同负载计时为准。
