# 让小模型提出候选

[返回首页](../README.md) · [验证与回退](SPECULATIVE.md) · [lookup 候选](LOOKUP_DRAFT.md)

前一个 lookup 例子从已提交历史里找续文。它能独立提出候选，真实短输入却一个也没接受。这次换成一个真正的小模型：Qwen3.5-0.8B 提候选，Qwen3.5-2B 验证。先把两个模型的状态接续讲清楚，再看节省的 target 计算够不够支付 draft 成本。

这是 main 的 `0.4.0.dev0` 教学入口，已发布的 v0.3.0 安装包不包含它。从 main 安装时使用当前检出；v0.3 的固定 tag 和安装包继续保留。

## 先跑 CPU 小模型

安装方法沿用[CPU 指南](START_HERE.md)，已有兼容 PyTorch 即可，不需要权重或 Transformers。在 main 仓库目录：

```bash
git switch main
python -m pip install -e . --no-deps
python -m minideepinfer.model_draft_demo --output outputs/model-draft-cpu.json
```

程序随机初始化两个尺寸不同的 Qwen3.5 小模型，先计算独立 target greedy，再检查候选宽度 1 和 3。两边的有效缓存都与空缓存的整段计算比较，最后应看到 `greedy outputs and both caches match: True`。默认不计时。

这组随机模型恰好重复输出 token `4`，候选全部接受。宽度 1 的 target/draft 分别计算 6/4 次，宽度 3 是 4/6 次；完整程序还包含独立 greedy 和十四次冷缓存检查，共 44 次 forward。这个易命中的例子只用于看状态，不能代表真实接受率。

## 先过词表这一关

小模型输出的整数 ID 必须能直接作为 target 的输入。两个模型都叫 Qwen3.5，或者词表大小相同，都不足以证明这一点。

[`check_tokenizer_pair()`](../minideepinfer/model_draft.py)在加载权重之前，要求两套 `tokenizer.json`、tokenizer 配置逐字节相同；聊天模板和额外词表若存在，也必须两边都有且相同。模型配置中的词表大小、BOS、EOS 和 PAD ID 另行检查。这个严格规则会拒绝某些语义等价但文件不同的组合，暂不实现 token ID 转换。

现有 0.8B 和 2B 的四份文件逐一相同，词表大小都是 248320，EOS ID 都是 248044。本次使用的配置 SHA 和四份共享文件 SHA 都保存在原始报告中，并在实机加载前再次核对。

## 两个模型各持有自己的状态

target 继续使用 [`generate()`](../minideepinfer/speculative.py) 的验证与回退。新增 [`ModelDraft`](../minideepinfer/model_draft.py)实现同一个候选回调，内部复用 `CachedTarget`；两个模型的槽位编号可以相同，实际张量和线性状态互不共享。

设已提交前缀是 `P A`，其中 `A` 是 target 上轮刚输出、还没有消费的 token：

| 时刻 | target 已计算的前缀 | draft 已计算的前缀 |
| --- | --- | --- |
| 准备提出候选 | `P` | 首次计算 `P A`，或从上一轮补到这里 |
| draft 提出 `B C D` | 仍为 `P` | `P A B C`；计算 `C` 后预测出 `D`，不用先输入 `D` |
| target 接受 `B`，改正为 `X` | 回退并保留 `P A B` | 暂时还停在 `P A B C` |
| 下一次请求候选 | `X` 仍待 target 消费 | 恢复到共同前缀 `P A B`，再计算实际提交的 `X` |

draft 的快照保存在计算完上一份已提交前缀之后。若临时候选输入进入了错误分支，下一轮先恢复快照、重算保留的候选，再补算 target 实际提交的 token。完整注意力的错误 KV 退出有效范围，线性状态按同一套规则恢复。全部接受时，已有状态直接保留，再补算尚未作为 draft 输入的候选和 bonus。

最终只剩一个 target 输出时，候选上限为零，draft 不再计算。结束后两边分别同步和清理；draft 的等待失败会继续持有自己的槽位。测试检查拒绝后的恢复、EOS 在候选中被接受、初始提交回调异常，以及关闭失败时不能提前归还资源。

这套回调复用已有接受、提交和长度修正的分工，是对原[DeepInfer 来源映射](SPECULATIVE.md#这和-deepinfer-怎样对应)的延伸。双模型草稿器是本项目的简化实现，不是 DeepInfer 的 MTP，也没有实现概率接受、残差采样或 SWA Bounded Replay。

## Ascend 910C 的一次固定实验

复用已有环境和本地权重，不需要重新下载。权重固定 revision 见[Qwen3.5 说明](QWEN35.md)。在 main 仓库目录：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m minideepinfer.model_draft_demo \
  --draft-model weights/Qwen3.5-0.8B --target-model weights/Qwen3.5-2B \
  --device npu:0 --width 1 --repeats 3 \
  --output outputs/qwen35-model-draft.json
```

单逻辑 Ascend 910C，Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6，FP32、HF32 关闭、eager、线程数一。输入原始文本 `The capital of France is`，五个 token，不加聊天模板；固定输出十个 token，不用 EOS 截短。每次请求创建独立缓存，无共享前缀命中。

0.8B 和 2B 各加载一次，加载与分词共约 3.797 秒。先运行独立 greedy，再检查宽度 1 和 3；计时前两种路径各预热一次，然后只对宽度 1 做三组成对计时，第二组交换顺序。预热两次合计约 0.951 秒，正式六次请求合计约 2.858 秒；这些是分项，未当作完整任务耗时。

### 正确性与工作量

两种宽度的输出都与独立 greedy 相同，两边各归还全部 14 个槽位，draft 的线性状态和快照清空。十六次冷缓存检查在原定 `atol=1e-4, rtol=2e-4` 下通过，覆盖 target 回退后和 draft 对齐后的 K/V、卷积窗口、递推矩阵与前缀末尾完整 logits。最大 logits 绝对差约 `3.20e-5`；逐元素容差检查另包含相对误差，不能只看最大值。

| 宽度 | 实测接受 / 提出候选 | 严格接受率 | AL，含最终输出 | target forward / 输入 token | draft forward / 输入 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 / 5 | 80% | 9 / 5 = 1.8 | 7 / 16 | 5 / 13 |
| 3 | 6 / 7 | 85.71% | 9 / 3 = 3.0 | 5 / 18 | 7 / 13 |

这里的 AL 是实际含候选验证轮中提交的输出数 / 这些轮数，包含 correction 或 bonus，排除初始 prefill 和没有候选的直接输出。严格接受率直接使用接受与提出计数，没有从 AL 推算。三次正式宽度 1 运行均为接受 4/5、AL 1.8。

普通 greedy 是 target 10 次 forward、14 个输入 token。双模型宽度 1 的 target 七次包含一次恢复后重算；draft 的五次计算用来对齐已提交前缀并提出候选。宽度 3 的 target 五次也包含一次重算，draft 七次含四次逐候选计算。实际这段文本没有触发 draft 临时状态的截短恢复；相关拒绝分支由 CPU 的不同权重及强制改正测试覆盖，不包装成实机已走过的分支。

两模型的线性状态快照尺寸这次恰好相同：每份 20,201,472 字节，约 19.27 MiB。宽度 1 每边各复制五份，累计各 101,007,360 字节；宽度 3 每边各三份。这是克隆状态张量的字节数，不是进程峰值显存或带宽测量。

### 接受率提高，整次运行仍更慢

正式六次请求全部成功，三组中位数如下。TTFT、TPOT、E2E 和 ITL 都是离线 CPU 观察到提交的口径，没有 HTTP 服务。接受的一组 token 使用同一个提交时间，因此 ITL 中包含零值，不能把它读成逐 token kernel 延迟。E2E 取各路径最后一次提交，清理次序不同；主比较使用包括缓存初始化与最终清理的 whole。加载、预热和冷参考检查不在正式计时中。

| 路径 | AL | 实测严格接受率 | TTFT ms | TPOT ms | E2E ms | whole ms | 输出 token/s | 失败数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 greedy | 不适用 | 不适用 | 55.596 | 39.648 | 412.467 | 412.485 | 24.243 | 0 |
| 双模型，宽度 1 | 1.8 | 4/5 = 80% | 54.891 | 53.986 | 540.907 | 541.096 | 18.481 | 0 |

全部配对 whole 样本保留：

| 组 | 执行顺序 | 普通 greedy ms | 双模型 ms |
| --- | --- | ---: | ---: |
| 1 | greedy → 双模型 | 410.977 | 543.129 |
| 2 | 双模型 → greedy | 412.485 | 538.052 |
| 3 | greedy → 双模型 | 412.755 | 541.096 |

这个短负载下，双模型 whole 中位数增加约 31.2%，三组都更长。target 从十次降到七次，但额外的 draft 计算、快照和同步都参与整条路径。没有 trace 或分阶段计时，不能把多出的时间逐项归因；TTFT 的小差异也不作为收益。宽度 3 只做了正确性检查，没有对应计时，不能把它的 AL 与宽度 1 的耗时拼在一起。

本次完整程序共 138 次模型 forward：包含数值检查、十六次冷参考、预热和正式样本。没有重新下载、重启服务、采集 trace 或追加输入找更高接受率。这是一个能正确接续状态的双模型教学路径，尚无成为默认加速路径的依据。

## 原始报告与下一步

[CPU 报告](results/2026-10-24/model-draft-cpu.json)和[Ascend 910C 报告](results/2026-10-24/qwen35-08b-2b-model-draft.json)绑定十一份相关执行源码，报告 SHA 固定在[索引](results/model-draft/evidence.json)：

```bash
python scripts/check_evidence.py --index docs/results/model-draft/evidence.json
```

前一版 lookup 报告改按原 commit 校验，原始 JSON 和 SHA 不改。本轮不是新的 Transformers 或 BF16 对照；没有加入跨轮 pipeline、图执行或分布式。后续若优化，应先从这次可见的计算和复制成本选一个问题，而不是继续同时增加候选宽度、采样算法和更多模型。

## English summary

The `model_draft_demo` entry point on main (`0.4.0.dev0`) uses a separate greedy draft model and cache. It is absent from v0.3.0 assets. The draft aligns its previous tentative branch to the newly committed target prefix, reuses valid KV, restores linear state when needed, and computes correction or bonus inputs before proposing again. Both caches use the existing `CachedTarget`; this is not DeepInfer MTP or probability-preserving speculative sampling.

Real local Qwen3.5-0.8B and 2B checkpoints passed a strict tokenizer-file and special-ID contract before loading. On one logical Ascend 910C with FP32 and HF32 disabled, widths 1/3 accepted 4/5 and 6/7 proposals, with measured committed output per draft round of 1.8/3.0 including correction or bonus. Output matched independent target greedy, all sixteen cold-cache checks passed at `atol=1e-4, rtol=2e-4`, and both pools returned fourteen slots. Draft truncation recovery is covered by CPU tests; it was not triggered by this real trajectory.

Only width 1 was timed: one warmup per path, three alternating-order pairs, the same raw five-token input and ten-token output. Median whole latency increased from 412.485 ms to 541.096 ms; output throughput fell from 24.243 to 18.481 tokens/s. Every pair was slower. Median offline TTFT/TPOT/E2E were 55.596/39.648/412.467 ms for greedy and 54.891/53.986/540.907 ms for the dual-model path. Accepted tokens share a CPU commit timestamp; zero ITLs are not kernel latency. All samples are preserved, with no claim of a production speedup.

Target calls fell from ten to seven, but the draft made five additional calls; each model also copied five 20,201,472-byte linear-state snapshots. The entire check, warmup and measurement program made 138 model calls, loading each existing checkpoint once. There were no downloads or traces. Run the tiny CPU mode without checkpoints using `python -m minideepinfer.model_draft_demo`; real paths and paired timing require the explicit arguments above.
