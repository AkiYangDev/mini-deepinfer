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

需要输入临时候选时，draft 在计算完已提交前缀后保存快照。若候选输入进入错误分支，下一轮先恢复快照、重算保留的候选，再补算 target 实际提交的 token。完整注意力的错误 KV 退出有效范围，线性状态按同一套规则恢复。全部接受时，已有状态直接保留，再补算尚未作为 draft 输入的候选和 bonus。只提一个候选时没有临时输入，当前实现跳过这份快照；下面保留首轮实验，并单独记录这项修改的对照。

最终只剩一个 target 输出时，候选上限为零，draft 不再计算。结束后两边分别同步和清理；draft 的等待失败会继续持有自己的槽位。测试检查拒绝后的恢复、EOS 在候选中被接受、初始提交回调异常，以及关闭失败时不能提前归还资源。

### 计算抛异常后，不从半份状态继续

模型计算会逐层写入缓存。假设 draft 要从 `P` 补算到 `P A`，第一层已经更新线性状态，第二层却抛了异常；此时记录的 token 仍可能只有 `P`，实际张量已经不再是完整的 `P`。再输入一次 `A`，会让部分层多计算一遍。单看 token 数没前进，不能认为缓存也没变。

当前 `ModelDraft` 在计算阶段出错后会标记 `failed=True`，保留原异常，拒绝后续候选调用，包括候选上限为零的调用。回退重算、快照、采样或观察回调抛异常，以及 `KeyboardInterrupt`，也按这个规则结束该请求。它不尝试修补一半写入的缓存；同步、清理后，用新的 draft 开始请求。

调用者仍负责关闭自己创建的 draft，和演示入口一致：

```python
draft = ModelDraft(small, capacity)
try:
    result = generate(target, prompt, count, draft, width=1)
finally:
    draft.close()
```

`close()` 先确认设备用完，再清理状态和归还槽位。如果等待失败，`failed=True`、`closed=False`，槽位仍归这个对象持有；保留对象，待设备能够完成同步后重试 `close()`，不能继续提出候选。成功关闭后 `failed` 保留失败记录。计算前的参数校验错误不改变缓存，修正参数后可以继续使用同一个对象。

这条异常路径用 CPU 在第一层真实写入缓存后注入故障，覆盖首次计算、补齐前缀、临时候选和混合状态恢复重算；另检查 target 与 draft 出错时，两边都先完成同步再归还槽位。这些是 CPU 的状态与清理测试，没有把故障注入扩展为新的 Ascend 910C 实机结果。

失败时也要保留能读懂的报告。`model_draft_demo` 现在会在退出前更新 JSON：普通异常记为 `failed`，单独的 Ctrl+C 记为 `interrupted`；中断后清理又失败时，记录最终失败和原中断。它们都不会写成检查通过，也不会自动补跑后面的预热或计时。先看以下字段：

| 字段 | 用来回答什么 |
| --- | --- |
| `failure_phase` | 出错时在正确性检查、预热还是哪一组正式运行 |
| `failed_operation` | 被捕获的 forward 或缓存对照属于 target 还是 draft；未捕获到这类操作时没有该字段 |
| `failed_run.generation_error` | 生成阶段为什么退出；生成正常返回时为 `null` |
| `failed_run.draft_cleanup` | draft 关闭是否成功，以及关闭后的失败标记、槽位和剩余状态 |
| `failed_run.target_cleanup` | target 是否已经返回可核对的槽位归还结果 |
| `error_chain` | 按最终异常到原始原因的顺序，保留生成与清理中先后发生的错误 |

`failed_run` 只在双模型生成或 draft 关闭出错时记录。已完成运行的后续检查失败，则在 `checked_runs`、`warmups` 或 `trials` 中查看对应结果。

例如，生成已经结束、draft 等待设备却失败，报告会同时写出 target 已归还的槽位和 draft 仍持有的资源。如果生成没有返回，当前接口无法直接观察 target 的最终清理结果，就写 `NOT_OBSERVED`；它既不代表成功，也不代表失败。不会因为 draft 清理完成，就顺便把 target 也算作完成。

出错后的 `model_work` 和 `total_forward_calls` 按进入 forward 的次数计数，包含抛异常的那一次；输入 token 数也是尝试提交的数量。它们不证明设备已经算完。`draft_work` 的 forward 计数则仍只在一次调用成功返回后增加，所以两者在失败报告中可能不同。生成错误和清理错误会分别保存；两条错误不等于两次请求失败。

这套回调复用已有接受、提交和长度修正的分工，是对原[DeepInfer 来源映射](SPECULATIVE.md#这和-deepinfer-怎样对应)的延伸。双模型草稿器是本项目的简化实现，不是 DeepInfer 的 MTP，也没有实现概率接受、残差采样或 SWA Bounded Replay。

## 首轮 Ascend 910C 固定实验

本节对应提交 `7b27e8e`，当时 draft 每次提出候选都建立快照。原始报告继续绑定该提交，以下数字保留当时的执行行为。

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

## 一个候选为什么不用保存 draft 快照

上面的状态表里，提出 `B C D` 需要先输入 `B C`，因此 draft 可能走进错误分支。但如果只提 `B`，计算完 `P A` 的 logits 就足够选出它，根本不用把 `B` 输入 draft。

若 target 拒绝 `B`、改为 `X`，draft 仍停在 `P A`，下一轮直接计算 `X`。若接受 `B` 并输出 bonus `Y`，draft 下一轮计算 `B Y`。两种情况都只是沿实际提交的前缀往后算，没有需要撤销的 draft 状态。

因此修改只有一个条件：候选上限大于一才建立 draft 快照。target 仍会把候选送入验证，拒绝后仍需恢复自己的线性状态，它的快照不动。测试另外检查 3→1→3 的宽度切换：先解决上一轮的临时分支，再决定当前轮是否需要快照。最终候选上限为零时不计算，关闭时仍清理之前的状态。

### 单独比较这一次复制

在同一环境重新加载现有的 0.8B 和 2B，各一次；模型、FP32/HF32 设置、五 token 原始输入和十 token 输出都与首轮一致。两种方式共用同一执行逻辑、模型实例和缓存观察回调，只让参考方式在宽度 1 的 draft 对齐后、argmax 前恢复旧快照。每个请求仍各建缓存，不共享前缀。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m minideepinfer.model_draft_demo \
  --draft-model weights/Qwen3.5-0.8B --target-model weights/Qwen3.5-2B \
  --device npu:0 --width 1 --repeats 3 --compare-draft-snapshots \
  --output outputs/qwen35-draft-snapshot.json
```

不提供模型路径和 `--device`，同一命令入口可检查随机 CPU 小模型。对照开关要求指定宽度 1。附带的宽度 3 检查在两次运行中都使用当前快照条件；它的最后一轮实际候选上限降为一，因此两次各保存两份快照，不能当作旧版宽度 3 的对照。

实机两种方式的输出逐 token 相同，接受率都是 4/5、AL 都是 9/5 = 1.8。target 仍为七次 forward / 十六个输入 token，draft 仍为五次 / 十三个输入 token。两边各归还十四个槽位，线性状态和快照清空；三十二次冷缓存检查通过原 `atol=1e-4, rtol=2e-4`，最大 logits 绝对差仍约 `3.20e-5`。实际轨迹仍未触发 draft 截短恢复，该分支由 CPU 测试覆盖。

宽度 1 每次请求的 draft 快照从五份降为零，少克隆 101,007,360 字节，约 96.3 MiB；target 的五份快照保留。这是张量复制计数，未测进程峰值显存。预热各一次后做三组成对测量，第二组交换顺序，正式六次请求全部成功。

| draft 快照方式 | AL | 实测严格接受率 | TTFT ms | TPOT ms | E2E ms | whole ms | 输出 token/s | 失败数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 宽度 1 仍保存 | 1.8 | 4/5 = 80% | 51.107 | 50.120 | 502.276 | 502.466 | 19.902 | 0 |
| 仅输入临时候选时保存 | 1.8 | 4/5 = 80% | 51.178 | 49.745 | 498.879 | 499.026 | 20.039 | 0 |

表中为三次中位数。每种方式的正式样本合计接受 12/15，AL 为 27/15；接受组共享 CPU 提交时间，零 ITL 继续按原定义保留。缓存初始化及最终同步清理计入 whole，模型加载、冷参考与预热不计入。

| 组 | 执行顺序 | 仍保存 ms | 跳过 ms |
| --- | --- | ---: | ---: |
| 1 | 仍保存 → 跳过 | 502.466 | 498.156 |
| 2 | 跳过 → 仍保存 | 502.000 | 499.026 |
| 3 | 仍保存 → 跳过 | 502.555 | 499.434 |

三组都略短，whole 中位数减少约 0.7%。复制确实省掉了，耗时差异很小，只能支持这个固定短负载上的局部改进。这次没有重新计时普通 greedy，不能把 499.026 ms 与首轮的 412.485 ms 拼成同场比较，也不能把两次实验间的总体差异归给这份改动。

完整程序共 186 次 forward：独立 greedy 十次、四次正确性运行共四十八次、冷参考三十二次、预热二十四次、正式样本七十二次。加载与分词约 4.107 秒，预热合计约 1.006 秒，正式六次合计约 3.004 秒；没有下载、trace 或追加输入。这项修改保留为默认行为，因为宽度 1 没有需要回退的 draft 写入，代码也更贴合实际状态变化。

## 原始报告与后续

[CPU 报告](results/2026-10-24/model-draft-cpu.json)和[Ascend 910C 报告](results/2026-10-24/qwen35-08b-2b-model-draft.json)绑定十一份相关执行源码，报告 SHA 固定在[索引](results/model-draft/evidence.json)：

```bash
python scripts/check_evidence.py --index docs/results/model-draft/evidence.json
```

快照修改另有 [CPU 报告](results/2026-10-24/draft-snapshot-cpu.json)、[Ascend 910C 报告](results/2026-10-24/qwen35-draft-snapshot.json)和[源码索引](results/draft-snapshot/evidence.json)：

```bash
python scripts/check_evidence.py --index docs/results/draft-snapshot/evidence.json
```

lookup 与首轮双模型报告按原 commit 校验，原始 JSON 和 SHA 不改。这两轮都没有新增 Transformers 或 BF16 对照，也没有跨轮 pipeline、图执行或分布式。后续应优先收集这个教学入口的复现反馈；若继续分析性能，再从已有计算、同步路径选择一个明确问题。

## English summary

The `model_draft_demo` entry point on main (`0.4.0.dev0`) uses a separate greedy draft model and cache. It is absent from v0.3.0 assets. The draft aligns its previous tentative branch to the newly committed target prefix, reuses valid KV, restores linear state when needed, and computes correction or bonus inputs before proposing again. Both caches use the existing `CachedTarget`; this is not DeepInfer MTP or probability-preserving speculative sampling.

Real local Qwen3.5-0.8B and 2B checkpoints passed a strict tokenizer-file and special-ID contract before loading. On one logical Ascend 910C with FP32 and HF32 disabled, widths 1/3 accepted 4/5 and 6/7 proposals, with measured committed output per draft round of 1.8/3.0 including correction or bonus. Output matched independent target greedy, all sixteen cold-cache checks passed at `atol=1e-4, rtol=2e-4`, and both pools returned fourteen slots. Draft truncation recovery is covered by CPU tests; it was not triggered by this real trajectory.

Only width 1 was timed: one warmup per path, three alternating-order pairs, the same raw five-token input and ten-token output. Median whole latency increased from 412.485 ms to 541.096 ms; output throughput fell from 24.243 to 18.481 tokens/s. Every pair was slower. Median offline TTFT/TPOT/E2E were 55.596/39.648/412.467 ms for greedy and 54.891/53.986/540.907 ms for the dual-model path. Accepted tokens share a CPU commit timestamp; zero ITLs are not kernel latency. All samples are preserved, with no claim of a production speedup.

Target calls fell from ten to seven, but the draft made five additional calls; each model also copied five 20,201,472-byte linear-state snapshots. The entire check, warmup and measurement program made 138 model calls, loading each existing checkpoint once. There were no downloads or traces. Run the tiny CPU mode without checkpoints using `python -m minideepinfer.model_draft_demo`; real paths and paired timing require the explicit arguments above.

Those initial reports are pinned to `7b27e8e`. Current main skips draft snapshots when only one candidate is requested: argmax reads the committed endpoint without feeding that candidate back into the draft. Target snapshots remain necessary. A separate same-load comparison (`--compare-draft-snapshots --width 1 --repeats 3`) removed five draft snapshots, or 101,007,360 cloned bytes per request, while preserving outputs, acceptance, model work and cleanup. All 32 cold-cache checks passed. Width-3 checks use the current policy in both runs; they are not an old-policy comparison.

The snapshot reports are pinned to `ed767bc`. Current main additionally marks a draft as failed after an exception during cache initialization, alignment, proposal computation or observation. A forward may have updated some layers before raising, even when the recorded token count did not advance. Failed drafts reject further proposal calls; `close()` must still drain, clear state and return slots. A failed drain retains ownership and permits retrying `close()`. Input validation before computation remains retryable. CPU fault injection covers real partial writes, prefill, prefix alignment, proposal and replay, interruption, and cleanup of both models. This exception handling change has no new NPU fault-injection or performance measurement.

The demo also saves exceptions and Ctrl+C as `failed` or `interrupted` reports, then re-raises them. Failure fields preserve the phase, an observed forward/cache-check role, the generation error, draft cleanup statistics and the exception chain. Target cleanup is marked completed only when `generate` returned its free-slot result; otherwise it is `NOT_OBSERVED`. If generation and draft cleanup both fail, both errors remain visible. Failed-run model-work counters count attempted forward entries and input tokens, including the call that raised; they do not establish device completion. These report paths are checked with CPU fault injection, without new hardware or timing claims.

Three alternating-order pairs had median whole latency of 502.466 ms with the old width-1 snapshots versus 499.026 ms without them, about 0.7% lower. Median TTFT/TPOT/E2E were 51.107/50.120/502.276 ms versus 51.178/49.745/498.879 ms; throughput was 19.902 versus 20.039 tokens/s. All six requests succeeded, each accepting 4/5 proposals with AL 1.8. This small fixed-workload change is not evidence of general speculative speedup. The new program made 186 forward calls and loaded each existing checkpoint once; ordinary greedy was checked for correctness but not retimed. Do not compare timings across the two experiments as a paired result.
