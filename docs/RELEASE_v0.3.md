# mini-DeepInfer v0.3.0

[返回首页](../README.md) · [运行指南](START_HERE.md) · [学习路线](LEARNING_PATH.md)

v0.3 围绕一个问题展开：target 已经算过一段候选，但只接受前半段，下一轮该从哪里继续？

这一版把验证、输出提交、有效 KV 回退和 Qwen3.5 线性状态恢复串起来，再接一个只读已提交上下文的 lookup draft。CPU 例子无需下载权重；真实 Qwen3.5-2B 在 Ascend 910C 上检查同一条状态路径。它是独立的同步教学入口，已有文本生成和跨轮循环继续保留。

## 按这三步运行

基础包要求 Python 3.10+；CPU 模型检查使用过 Python 3.12。先按[安装指南](START_HERE.md)安装到同一个环境，已有兼容 PyTorch 时不需要另装模型依赖。

先用可控候选看验证决定：

```bash
python -m minideepinfer.spec_demo --output outputs/spec-controlled.json
```

这个随机纯注意力小模型先计算 greedy 答案，再故意改错候选。首候选拒绝、第二候选拒绝和全部接受，target 分别计算 10、6、4 次；三种模式输出相同，分别归还全部 12 个槽位。候选来自答案，所以只能用来检查分支，不能用来证明加速。[跟着一轮验证读代码](SPECULATIVE.md#先运行)。

然后换成 Qwen3.5 的两种缓存：

```bash
python -m minideepinfer.spec_demo --hybrid --output outputs/spec-hybrid.json
```

完整注意力 KV 可以截短有效前缀，卷积窗口和递推矩阵却已经受错误候选影响。参考实现先保存线性状态，拒绝后恢复，再重算本轮保留的输入。target 次数因此变成 18、10、4，重算成本包含在内；每次提交后都与空缓存的整段计算对照。[看状态怎样恢复](SPECULATIVE.md#qwen35-猜错后还要恢复线性状态)。

最后去掉预先算好的答案：

```bash
python -m minideepinfer.spec_demo --draft lookup --hybrid \
  --output outputs/spec-lookup.json
```

lookup 只在已提交历史里找相同后缀，把已有续文作为候选。找不到就不提候选；普通 greedy 在生成结束后才运行，用于独立对照。随机 Qwen3.5 的重复输出恰好让四个候选全部接受，target 为六次 forward；去掉 `--hybrid` 的纯注意力例子则始终找不到候选。两者都保存，而不是只展示容易接受的结果。[候选来源和查找成本](LOOKUP_DRAFT.md)。

三个命令应写出 `passed: true` 的 JSON。报告中还有额外的答案准备、事后 baseline 或冷缓存检查成本，不能只拿 target 次数计算速度。浮点误差无需跨环境逐位相同，应按各检查的容差判断。

## Ascend 910C 上确认了什么

已有匹配的 PyTorch、torch_npu、CANN 和本地 Qwen3.5-2B 权重时，可运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m minideepinfer.spec_demo \
  --draft lookup --model /path/to/Qwen3.5-2B --device npu:0 \
  --output outputs/spec-lookup-npu.json
```

环境与权重准备见[运行指南](START_HERE.md)。这一版使用单逻辑设备、FP32、HF32 关闭、eager、greedy 和短输入，不接入跨轮 pipeline。保存的真实权重结果如下：

| 实验 | 已确认结果 | 实际工作量 |
| --- | --- | --- |
| 受控候选与混合状态恢复 | 三种模式输出均同独立 greedy；17 次提交后的冷缓存检查通过 | target 含重算 18/10/4 次 forward；每份线性状态快照约 19.27 MiB |
| 独立 lookup draft | 输出同事后 greedy；9 次冷缓存检查通过；14 个槽位归还 | 接受 0/5；target 为 12 次 forward、21 个输入 token，baseline 为 10 次、14 个 |

冷缓存检查覆盖有效 K/V、卷积状态、递推矩阵和前缀末尾完整 logits，均通过预设 `atol=1e-4, rtol=2e-4`。这是本项目的独立整段计算对照，没有新增 Transformers 或 BF16 投机结论。

lookup 的这个实机例子增加了工作量。它保留为教学路径，不作为默认加速功能；没有计时、吞吐或延迟收益，也没有为了找接受率更高的输入追加运行。快照与重算参考不等于 DeepInfer 的 SWA Bounded Replay，lookup 也不是 MTP/DSpark。

## 安装包与证据

[v0.3.0 发布页](https://github.com/AkiYangDev/mini-deepinfer/releases/tag/v0.3.0)提供 wheel、源码包和 `SHA256SUMS`。wheel 包含运行模块；源码包另含教程、测试、报告和检查脚本。目前从仓库或发布页安装，尚未发布到 PyPI。

下载 wheel 后，可以在仓库以外安装和查看入口；这不会安装模型依赖：

```bash
python -m pip install --no-deps ./mini_deepinfer-0.3.0-py3-none-any.whl
python -m minideepinfer.spec_demo --help
```

189 项测试在 Transformers 5.12.1 环境全部通过；4.46.3 环境通过 177 项、跳过 12 项；基础环境通过 70 项、跳过 119 项模型检查。安装包在仓库外执行三个 CPU 模式，并核对报告、输出和槽位回收。

本次发布整理没有改动模型、缓存、验证或 draft 的执行源码，复用已保存的 CPU 和 Ascend 910C 结果，没有重新加载实机权重。六份投机报告归入[v0.3 索引](results/v0.3/evidence.json)，每份保留自己的原始 SHA 和执行 commit：

```bash
python scripts/check_evidence.py --index docs/results/v0.3/evidence.json
```

这条历史检查在含完整历史的 Git 检出中运行，检查成功只证明报告对应的源码，没有重新执行模型。直接下载的源码包不含 Git 历史，不能据此完成历史检查；详细区别见[证据说明](EVIDENCE.md)。旧版本的 tag、报告和发布制品继续保留。

## English release notes

v0.3.0 adds a three-step greedy speculative-decoding lesson: controlled candidates, Qwen3.5 state snapshots and replay, then an independent lookup draft that reads committed history only. Run `spec_demo`, add `--hybrid`, then add `--draft lookup`. These CPU examples need PyTorch but no checkpoint. The existing text and cross-round frontends remain separate.

Preserved Qwen3.5-2B FP32 results on Ascend 910C match independent greedy output. The controlled run passed 17 cold-cache checks; lookup passed nine, covering valid K/V, convolution and recurrent states, and endpoint full logits at `atol=1e-4, rtol=2e-4`. Lookup accepted none of five proposals and increased target work from 10 forwards / 14 input tokens to 12 forwards / 21 tokens. This is a working teaching path, with no timing or speedup claim. It does not implement probability-preserving sampling, a learned draft, MTP/DSpark, SWA Bounded Replay or cross-round speculation.

All 189 tests pass with Transformers 5.12.1; 177 pass and 12 skip with 4.46.3; the base environment passes 70 and skips 119 model checks. The wheel is checked outside the repository. Release preparation changes no execution source and reuses preserved hardware results without loading the model again. The release provides a wheel, source distribution and SHA256SUMS; it is not on PyPI. The six-report evidence index retains each recorded source revision and requires Git history for verification.
