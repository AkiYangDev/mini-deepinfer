# mini-DeepInfer v0.1.0

[返回首页](../README.md) · [运行指南](START_HERE.md) · [学习路线](LEARNING_PATH.md)

这是第一个整理完整学习入口的教学版本。它用 Qwen3.5-2B 在 Ascend 910C 上运行真实文本推理，再通过可对照的小实验，解释 DeepInfer 的执行职责、设备 token 接续和跨轮状态有效期。

## 这个版本包含什么

| 入口 | 能力 | 使用方式 |
| --- | --- | --- |
| Engine | 同步单请求、显式位置与槽位、greedy、EOS/长度停止 | 第一、二课的参考路径 |
| BatchEngine | 动态批处理、decode 优先的输入预算、分块 prefill、取消和整请求回收 | `generate` 的多请求路径 |
| DeferredRunner / BatchEngine 分步接口 | 分开发起与结果提交，途中可接收或取消请求，排空后清理 | `--deferred`；仍只有一批在途 |
| 跨轮实验 | 设备 token 接续、回调到达、FIFO 准入、分块 prompt、EOS/取消后最后使用事件回收 | `verify_chunk_pipeline`；独立循环，未接入 `generate` |
| 模型与缓存 | Qwen3.5 纯文本 forward、完整注意力 KV、按请求独占的 DeltaNet 状态 | 2B 主例，0.8B 开发；Qwen2 保留回归 |
| 教学材料 | 十四课、整数模拟、随机小模型、真实权重数值报告和有限计时记录 | [学习路线](LEARNING_PATH.md) |

`plan_step()` 共用同步与跨轮 Step 构造，`_launch()` / `_sample()` 共用固定与动态实验的设备输入与采样。同步提交和设备已发出进度各有明确的状态，v0.1 保留不同的调度生命周期，没有为减少表面行数合并它们。

## 发布前补了哪些检查

新检查使用固定 seed，在 32 组跨轮轨迹中改变 prompt 长度、预算、并发、到达和取消时机，并加入受控 EOS；另有 16 组同步轨迹和 5 种单请求预算。对照路径是原始 Engine 的整段 prompt 执行，不经过共用规划函数。检查输出前缀、连续位置、总输入预算、槽位独占、停止进度和最终清理；刚好占满缓存、只输出一个 token、连续复用 Runner 也覆盖。

这些检查没有发现需要修改设备执行核心的缺陷，因此本次发布整理保留现有执行源码。当前完整测试为 140 项：Transformers 5.12.1 环境全部通过；4.46.3 环境 128 通过、12 项独立参考跳过；无模型依赖时 57 通过、83 跳过。跳过的模型检查不计为通过。

安装验证从构建的 wheel 开始，在新虚拟环境、仓库目录之外运行无依赖示例；带 PyTorch 的安装后入口另跑随机模型对照。源码目录能运行和安装包能运行分别检查。基础包无运行时依赖，模型依赖按需安装；未发布到 PyPI。

## Ascend 910C 证据

v0.1 复用已完成的 Qwen3.5-2B FP32 实机检查。环境为 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0，关闭 HF32，使用一个逻辑设备。

最新跨轮轨迹包含九轮、十四组完整 logits 对照：最大绝对差 `2.6702880859375e-5`，在原 `atol=1e-4, rtol=2e-4` 内；四个中间 prompt 块不采样，输出与整段参考一致。新请求复用已结束请求的槽位，旧前缀和线性状态清除；分块期间取消、主机异常后的排空清理与再次复用通过，最终归还 24/24 槽位。

[证据索引](results/v0.1/evidence.json)固定原始报告的 SHA，报告保留十五份相关源码 SHA。运行 `python scripts/check_evidence.py` 可核对当前检出文件。索引检查通过表示保存的实测与对应源码一致，不等于重新执行了 NPU 测试，也不意味着每个相关模块的所有分支都做过硬件验证。更早章节保留各自的历史源码与结果，详细条件见[验证记录](VALIDATION.md)。

本次发布整理没有再次下载权重、加载 2B 或采集 trace。原跨轮程序一次加载权重、共 18 次 forward，包括独立参考、主轨迹、异常和复用，没有性能计时。

## 使用边界

主例使用单设备、eager、greedy、FP32 和短上下文。完整注意力采用便于阅读的显式 mask，DeltaNet 用逐 token 递推。它适合跟读与实验，不提供生产吞吐承诺。

BF16 的固定短轨迹已通过匹配扫描方式与 FP32 状态的参考；原生分块与不同调度下的长生成仍有差异，[失败记录与原因分析](BF16_REFERENCE.md)继续保留。CPU 注入检查覆盖等待失败时的资源保护，不宣称真实设备故障恢复。

视觉编码器、MLA/MoE、MTP/投机解码、bounded replay、分页与共享前缀、抢占、图执行、并行部署和 HTTP 服务不属于本版本。它们在有独立模型、场景和验证需求后再加入，见[后续路线](ROADMAP.md)。

## English release summary

v0.1.0 provides a small, runnable teaching engine inspired by DeepInfer, with text-only Qwen3.5-2B on Ascend 910C. It includes synchronous batching and prompt chunks, split launch/commit execution, and a separate cross-round device-token experiment covering arrivals, stops, and safe reuse.

This release adds generated scheduling checks, installed-wheel smoke checks, a source-backed NPU evidence index, a shorter bilingual README, and an end-to-end setup guide. All 140 tests pass with Transformers 5.12.1; the older-reference and dependency-free environments explicitly skip unsupported model checks.

The preserved FP32 device run passes fourteen full-logit comparisons at existing tolerances, with maximum absolute difference about `2.67e-5` and all 24 cache slots reclaimed. Runtime source is unchanged by this release preparation; evidence hashes are checked against the release tree. No new performance claim, device run, weight download, or trace collection is added. Cross-round scheduling remains separate from `generate`, and BF16 scan-dependent differences remain documented.
