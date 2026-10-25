# mini-DeepInfer v0.4.0

[返回首页](../README.md) · [逐轮演示](MODEL_DRAFT_WALKTHROUGH.md) · [实现与实验](MODEL_DRAFT.md)

v0.4 把“小模型提出候选，target 验证，再从正确状态继续”做成一条可运行的教学路径。两个模型各持有自己的 KV 和线性状态，复用已有接受、提交与回退代码；这一版仍是单请求、同步、eager、greedy，和跨轮文本生成入口分开。

## 先看一次拒绝

按[运行指南](START_HERE.md)安装兼容 CPU PyTorch，在 v0.4.0 源码目录运行：

```bash
python -m minideepinfer.model_draft_demo --cpu-example rejection --trace-rounds --width 3 --output outputs/draft-rejection.json
```

无需权重或 NPU。随机小模型的候选来自自身预测，不读取 target 答案，也不手动改错。前两轮会看到：

```text
提出 [39, 18, 70]；draft 实际额外输入 [39, 18]。
target 输入 [43, 39, 18, 70]，预测 [1, 96, 51, 58]。
接受 0 个候选，提交 [1]（target 改正）。
...
draft 已算 6 → 保留 4 → 补到 5；恢复线性状态=True，重算 0，补算 1 个输入。
```

读者可以对着这几行，追踪候选为什么不能提交、target 和 draft 分别保留了哪些输入，以及线性状态为什么不能仅靠截短 KV 恢复。[完整讲解](MODEL_DRAFT_WALKTHROUGH.md)另有全部接受的对照、状态图和真实 NPU 输出。观察只在正确性运行中启用，不增加模型调用，不与计时混用。

## 这一版还包含什么

- 加载前严格核对两套 tokenizer 文件和特殊 token ID，保证候选能直接作为 target 输入。
- draft 对齐实际提交的前缀；需要时恢复线性状态并重算保留输入。只提一个候选时没有临时输入，不保存 draft 快照。
- 计算异常后禁止继续复用 draft；同步成功后才能归还资源，关闭等待失败可保留对象再收尾。
- 失败或 Ctrl+C 时保存阶段、异常链和 draft 清理状态；未观察到的 target 清理明确写 `NOT_OBSERVED`。

已有 v0.3 可控候选与 lookup 教程继续保留。它们与双模型参考都不是概率接受、MTP/DSpark 或 SWA Bounded Replay，也没有接入跨轮投机、HTTP 服务或分布式。

## 发布验收与性能边界

214 项测试在 Transformers 5.12.1 环境全部通过；4.46.3 环境通过 202 项、跳过 12 项；基础环境通过 75 项、跳过 139 项。wheel 在仓库外检查默认入口、两个逐轮 CPU 例子、快照对照及失败报告，源码包包含教程、测试与原始结果。

2026-10-25 的发布候选在单逻辑 Ascend 910C 上，以现有 Qwen3.5-0.8B/2B 权重、FP32、HF32 关闭、五输入十输出完成验收。宽度 1/3 接受 4/5、6/7，AL 为 1.8、3.0；输出同独立 greedy，十六次冷缓存检查通过预设容差，两边各归还 14 个槽位。两套权重各加载一次，加载与分词约 4.072 秒；完整程序五十次 forward，无下载、计时或 trace 采集。这里的逐轮观察不是性能 profiling。

早期双模型实验比普通 greedy 更慢，随后省去宽度 1 draft 快照的单项对照只测到约 0.7% 的耗时下降。它们各绑定当时的固定源码，数据完整保留在[实现与实验](MODEL_DRAFT.md)中。本次验收不把旧时间拼成新版本的性能结果，也不宣称投机加速。

## 安装与证据

[v0.4.0 发布页](https://github.com/AkiYangDev/mini-deepinfer/releases/tag/v0.4.0)提供 wheel、源码包和 `SHA256SUMS`，未发布到 PyPI。下载 wheel 后，在已有兼容 PyTorch 的环境中：

```bash
python -m pip install --no-deps ./mini_deepinfer-0.4.0-py3-none-any.whl
python -m minideepinfer.model_draft_demo --cpu-example rejection --trace-rounds --width 3 --output outputs/draft-rejection.json
```

三份新报告分别保存重复 CPU、拒绝 CPU 与真实 NPU 轨迹，各记录十二份源码的 SHA。在 Git 源码检出中核对：

```bash
python scripts/check_evidence.py --index docs/results/v0.4/evidence.json
```

检查只核对文件，不重新运行模型。旧报告继续按其固定 commit 核对，原始 JSON 和 SHA 不改，旧 tag 与发布附件保留。

## English release notes

v0.4.0 adds a separate cached greedy draft model and a round-by-round walkthrough. Run the CPU command above to see naturally rejected candidates from untrained random Qwen3.5 models with untied output heads. No answer lookup, candidate injection, checkpoint or NPU is required. The walkthrough follows proposal, verification, correction, target rollback and next-round draft alignment; the default repeated-token example provides an all-accepted comparison.

The release includes tokenizer compatibility checks, conditional draft snapshots, failed-draft reuse protection, and interruption/cleanup reports. The synchronous single-request path remains separate from cross-round text generation. It is not MTP/DSpark, probability-preserving sampling or SWA Bounded Replay.

All 214 tests pass with Transformers 5.12.1; 202 pass and twelve skip with 4.46.3; the base environment passes 75 and skips 139. Installed-package examples are checked outside the repository. The Ascend 910C FP32 release candidate accepts 4/5 and 6/7 proposals at widths 1/3, matches independent greedy output, passes sixteen cold-cache checks and returns all fourteen slots per model. It loads each existing checkpoint once and makes fifty forward calls, without performance timing. Earlier negative and small-effect timing results retain their original source revisions; this release makes no speedup claim.

The release provides a wheel, source distribution and SHA256SUMS, not a PyPI upload. Three new source-backed reports and full terminal observations are included; older tags, raw reports and release assets remain available.
