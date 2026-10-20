# 从设计参考到第一段实现

[返回首页](../README.md) · [开发路线](ROADMAP.md) · [来源记录](REFERENCES.md)

更新时间：2026-10-20。本文把部署报告、DeepInfer 和 Ascend SGLang 的不同参考作用对应到项目交付；源码阅读不等于 Ascend 910C 兼容性验证。

## 三类参考，各自解决一个问题

| 参考 | 主要作用 | 使用方式 |
| --- | --- | --- |
| [DeepSeek V4.1 Flash 大规模推理部署实践](https://zhuanlan.zhihu.com/p/2095560295930508953) | 理解部署目标、缓存生命周期、负载均衡与算法取舍 | 为机制实验提供假设和背景；区分部署经验与普遍成立的结论 |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b) | 理解引擎、执行器、输入元数据和在途状态的边界 | 建立原版与教学版对应关系，保留简化说明 |
| [Ascend SGLang：release/2026930](https://github.com/Ascend/sglang/tree/545ed1c6649321aa61b57a287032a7791690247e) | 理解 NPU 运算接口、缓存布局、stream/event 和图执行 | 提取适用的最小接口，在目标软件/硬件环境独立验证 |

mini-sglang 继续作为紧凑实现和教学组织的参考。mini-DeepInfer 自己实现请求循环、调度与缓存；后端接口可借鉴成熟项目，验证条件随代码一同记录。

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

最小同步骨架已落地：[`Engine.step()`](../minideepinfer/engine.py) 显式组织调度、输入元数据、Runner 和结果提交；[`SlotPool`](../minideepinfer/kv_cache.py) 演示逻辑槽位归属。它们是概念借鉴后的原创实现，尚无真实模型、设备计算或异步执行。阅读入口见[第一课](FIRST_REQUEST.md)。

下一项交付为真实模型单请求生成，包含：

1. **环境清单。** 记录设备枚举与物理卡映射、可用 HBM、驱动、CANN、PyTorch 和 `torch_npu`；公开版本去掉私人机器信息。
2. **基础运算探测。** 用固定小输入检查 BF16 matmul、归一化、RoPE 基础运算、缓存索引写入和 attention 参考计算。
3. **融合 attention 探测。** 仅对实际需要的最小形状验证调用，保留参考结果，明确未通过与未验证项。
4. **M1 输入/缓存约定。** 固定入门模型的 token、位置、KV 写入和采样接口，再实现真实权重生成。

硬件执行前检查资源占用，并使用独立环境。M0 的结束标志是一套可复现的实际结果；文档完善或 import 成功不替代硬件验证。
