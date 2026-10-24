# mini-DeepInfer v0.1.0

[返回首页](../README.md) · [运行指南](START_HERE.md) · [学习路线](LEARNING_PATH.md)

v0.1 可以在 Ascend 910C 上运行 Qwen3.5-2B 文本生成。模型的前向计算、批次调度和缓存管理都在项目代码里，便于跟读和修改。十四课从单请求开始，逐步解释 DeepInfer 的执行思路：下一轮怎样使用设备上的 token，请求结束后，还有哪些工作必须等设备完成。

## 这个版本包含什么

| 入口 | 能力 | 使用方式 |
| --- | --- | --- |
| Engine | 同步单请求、显式位置与槽位、greedy、EOS/长度停止 | 第一、二课的参考路径 |
| BatchEngine | 动态批处理、decode 优先的输入预算、分块 prefill、取消和整请求回收 | `generate` 的多请求路径 |
| DeferredRunner / BatchEngine 分步接口 | 分开发起与结果提交，途中可接收或取消请求，排空后清理 | `--deferred`；仍只有一批在途 |
| 跨轮实验 | 设备 token 接续、回调到达、FIFO 准入、分块 prompt、EOS/取消后最后使用事件回收 | `verify_chunk_pipeline`；独立循环，未接入 `generate` |
| 模型与缓存 | Qwen3.5 纯文本 forward、完整注意力 KV、按请求独占的 DeltaNet 状态 | 2B 主例，0.8B 开发；Qwen2 保留回归 |
| 教学材料 | 十四课、整数模拟、随机小模型、真实权重数值报告和有限计时记录 | [学习路线](LEARNING_PATH.md) |

同步与跨轮执行使用同一个 `plan_step()` 构造 Step，固定与动态实验也共用 `_launch()` / `_sample()`。调度循环仍各自保留：同步引擎按已提交的位置规划，跨轮循环还要记住发出但未提交的工作，以及每条请求最后一次使用缓存的事件。

## 运行和测试

发布前增加了 32 组跨轮轨迹，改变 prompt 长度、预算、并发、到达和取消时机，并加入受控 EOS。另有 16 组同步轨迹和 5 种单请求预算，覆盖刚好占满缓存、只输出一个 token 和连续复用 Runner。所有结果都与原始 Engine 的整段 prompt 执行比较；参考路径不调用共用规划函数。

检查同时核对输出、连续位置、输入预算、槽位归属和结束后的清理。完整测试共 140 项：Transformers 5.12.1 环境全部通过；4.46.3 环境通过 128 项、跳过 12 项独立参考检查；未装模型依赖时通过 57 项、跳过 83 项。

安装包也单独检查过：将 wheel 装进新虚拟环境后，离开仓库目录运行基础例子；已有 PyTorch 的环境则再跑随机模型对照，确认程序确实从安装包加载。基础包没有运行时依赖，模型依赖按需安装，步骤见[运行指南](START_HERE.md)。目前通过源码或发布附件安装，尚未上传 PyPI。

## Ascend 910C 实测

实测使用 Qwen3.5-2B、FP32 和一个逻辑设备，关闭 HF32。软件为 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0。

最新跨轮轨迹包含九轮、十四组完整 logits 对照：最大绝对差 `2.6702880859375e-5`，在原 `atol=1e-4, rtol=2e-4` 内；四个中间 prompt 块不采样，输出与整段参考一致。新请求复用已结束请求的槽位，旧前缀和线性状态清除；分块期间取消、主机异常后的排空清理与再次复用通过，最终归还 24/24 槽位。

[原始报告与源码索引](results/v0.1/evidence.json)记录了这次运行及十五份相关源码的 SHA。运行 `python scripts/check_evidence.py` 可以核对当前文件。脚本只比较文件，不执行模型；各模块实际覆盖哪些路径，以报告中的输入和轨迹为准。更早章节对应的版本和条件见[验证记录](VALIDATION.md)。

发布整理保留了这份已验证的执行源码，因此直接沿用上述结果。原程序一次加载权重，共执行 18 次 forward，包括参考、主轨迹、异常清理和复用检查，没有做性能计时。

## 使用边界

主例使用单设备、eager、greedy、FP32 和短上下文。完整注意力采用显式 mask，DeltaNet 按 token 展开递推，方便观察计算和状态变化。现有性能实验只覆盖少量短请求，尚未证明稳定提速。

BF16 的固定短轨迹已通过相同扫描方式、FP32 状态参考的对照，原生分块与不同调度下的长生成仍有差异，见[失败记录与原因分析](BF16_REFERENCE.md)。等待失败后的资源保护用 CPU 异常注入检查过，真实设备故障恢复尚未验证。

视觉编码器、MLA/MoE、MTP/投机解码、bounded replay、分页与共享前缀、抢占、图执行、并行部署和 HTTP 服务不属于本版本。它们在有独立模型、场景和验证需求后再加入，见[后续路线](ROADMAP.md)。

## English release summary

v0.1.0 runs text-only Qwen3.5-2B on Ascend 910C, with model computation, scheduling, and cache management in the project code. Fourteen lessons follow a request from synchronous execution to cross-round device-token handoff, arrivals, stops, and cache reuse.

The release adds scheduling checks across varied inputs and arrival times, tests the installed wheel outside the repository, and links the saved NPU results to the source used for each run. The bilingual setup guide starts with examples that need no weights. All 140 tests pass with Transformers 5.12.1; model checks that cannot run in the older-reference or dependency-free environments are reported as skipped.

The FP32 device run passes fourteen full-logit comparisons at the existing tolerances, with a maximum absolute difference of about `2.67e-5` and all 24 cache slots reclaimed. Release preparation leaves that execution code unchanged, so the saved results still match the source. Cross-round scheduling is a separate experiment from `generate`. BF16 differences across scan methods and schedules are documented, and the small timing workloads have not shown a consistent speedup.
