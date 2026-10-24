# 跨轮执行，这次快了吗？

[返回首页](../README.md) · [A/B/C 演示](WALKTHROUGH.md) · [源码映射](SOURCE_MAP.md)

这次没有。2026-10-24，用 Qwen3.5-2B 在 Ascend 910C 的一个逻辑设备上比较同步与跨轮路径，三类短负载的跨轮耗时都更长。输出一致，缓存也都归还了；这说明实现通过了本次检查，并不说明它更快。

本文保留首次基准的源码与结果。后续只改变首轮收取时机的实验见[首轮结果](FIRST_RESULT.md)，它使用同一轮运行里的基线，没有重写本文数据。

这个结果回答了一个容易混淆的问题：把下一轮计算提前发出去，只改变了执行顺序。它能否节省时间，还取决于准备输入、提交结果、接纳新请求和等待设备各自的成本。

## 怎样比较

入口是 [`benchmark_pipeline.py`](../minideepinfer/benchmark_pipeline.py)。复用一份只读权重，每次新建 Runner 和缓存，分别运行同步 `BatchEngine` 与现有 `StoppingDecodePipeline`。没有修改两条执行路径的实现。

| 条件 | 本次设置 |
| --- | --- |
| 模型 | Qwen/Qwen3.5-2B，沿用 revision `15852e8c16360a2fea060d615a32b45270f8a8fc` 的本地权重 |
| 设备 | Ascend 910C，一个逻辑设备，启动前可用显存 62,381 MiB；无其他设备进程 |
| 软件 | Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0，aarch64 |
| 计算 | FP32，关闭 HF32，eager、greedy，CPU 线程数 1 |
| 输入与输出 | 原始文本，不加聊天模板；计时前统一分词；关闭 EOS，每请求固定输出 8 token |
| 调度 | 同时最多容纳 2 个请求，每轮输入预算 8 token，开启分块 prefill，缓存容量 128 |
| 到达 | 一批请求同时进入队列，共用同一个计时起点；没有按轮次模拟不同到达时间 |
| 缓存 | 每次重建，无跨请求共享前缀，无跨次 KV 或线性状态复用 |
| 重复 | 每种负载两条路径各预热一次，再测三对；顺序依次为同步/跨轮、跨轮/同步、同步/跨轮 |

三类输入分别为：单请求长度 5；四条相同请求，各长 5；四条混合请求，长度 5、8、13、1。后两类有排队，不代表四条一直同时运行。原始文本、token ID、逐请求输出与逐轮计划全部保存在[原始报告](results/2026-10-24/qwen35-2b-pipeline-benchmark.json)。

这里的延迟是**离线 CPU 提交口径**：从统一提交开始，到该请求第一个或最后一个 token 已提交，并完成该轮清理后记时，分别作为 TTFT 和 E2E。TPOT 为每请求 `(E2E - TTFT) / 7`。它们包含排队，不包含分词，也不是 HTTP 客户端延迟。

整批耗时另以最终排空、设备同步完成为终点，输出吞吐为本批输出 token 总数除以这个时长。两条路径都记录轻量主机元数据和提交时间；计时中不打印、不采集 profiler、不读取完整 logits。缓存初始化、权重加载与预热在正式窗口之外，单独记录成本。

## 测到了什么

下面的 TTFT、TPOT、E2E 先在每次运行内按请求取均值，再取三次运行的中位数。耗时与吞吐各自取三次中位数；不能把这些独立聚合值再代回公式要求严格相等。

| 负载 | 路径 | TTFT ms | TPOT ms/token | E2E ms | 整批耗时 ms | 输出 token/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 单请求 | 同步 | 48.47 | 35.98 | 300.20 | 300.23 | 26.65 |
| 单请求 | 跨轮 | 90.33 | 31.72 | 312.37 | 312.49 | 25.60 |
| 同长四请求 | 同步 | 291.52 | 48.03 | 627.72 | 841.92 | 38.01 |
| 同长四请求 | 跨轮 | 364.60 | 44.27 | 674.48 | 877.62 | 36.46 |
| 混合四请求 | 同步 | 321.57 | 49.05 | 664.92 | 880.71 | 36.33 |
| 混合四请求 | 跨轮 | 404.70 | 44.62 | 725.45 | 944.12 | 33.89 |

正式测量共 54 次请求执行，失败 0 次；含预热共 72 次请求执行，输出全部对齐。每次结束后 128 个槽位全部归还，主机前缀与线性状态均为空。本次比较 greedy 输出，不重新做完整 logits 数值验收；后者见[已有文本入口验证](PIPELINE_TEXT.md)。

三对原始整批耗时也保留在这里，避免中位数遮住波动。每格按“同步 / 跨轮”排列，单位 ms：

| 负载 | 第 1 对 | 第 2 对 | 第 3 对 |
| --- | ---: | ---: | ---: |
| 单请求 | 298.72 / 312.49 | 300.23 / 313.01 | 300.51 / 311.74 |
| 同长四请求 | 836.33 / 875.48 | 843.14 / 877.62 | 841.92 / 897.14 |
| 混合四请求 | 876.31 / 944.12 | 885.63 / 971.80 | 880.71 / 905.98 |

按整批耗时中位数，跨轮分别增加约 4.1%、4.2%、7.2%。混合请求的跨轮耗时范围为 905.98–971.80 ms，波动比同步更大。三对短运行足以描述这次观察，不足以推断长上下文、大并发或生产服务的性能，也不适合估计 P99。

## 为什么 TPOT 更小，整批却更慢

先看单请求。两条路径都是 8 次 forward，实际处理 12 个输入 token，批次计划相同。但跨轮路径先调用下一轮 `_launch()`，然后才 `_finish()` 上一轮。首 token 在 CPU 上更晚提交：本次从约 48 ms 变成约 90 ms。

TPOT 只计算首 token 到末 token 之间的间隔。首 token 提交被推迟，这段区间会缩短，因此本表中较小的 TPOT 不能单独证明 decode 计算更快。应同时看 TTFT、E2E 和整批吞吐。

再看四请求负载。输入工作量相同，forward 次数却不同：

| 负载 | 每次同步 / 跨轮 forward | 两条路径各自处理的输入 token | 额外输入 token |
| --- | ---: | ---: | ---: |
| 单请求 | 8 / 8 | 12 | 0 |
| 同长四请求 | 17 / 18 | 48 | 0 |
| 混合四请求 | 17 / 18 | 55 | 0 |

以同长请求为例，第 8 轮（从 0 编号），同步路径已提交第一条请求的结束，可以把第三条请求的 prefill 与第二条的最后一次 decode 合批。跨轮路径在提交上一轮结果前就安排了这一轮，此时第一条仍占据运行位置；这一轮只有第二条的 decode，第三条要到第 9 轮才进入。

这正是原始计划里能看到的差异：**没有多处理 token，但少了一次合批机会，多调用了一次 forward。** 混合负载也因此改变了 prefill 的分块边界。它们比较的是完整执行路径，不能把耗时差全部归因为流、事件或 CPU/NPU 重叠。

本次没有采集 trace，不能量化上述各环节分别占多少设备时间。当前证据支持保留同步默认入口，并把跨轮路径用于观察执行顺序与资源生命周期。直接提前归还缓存会破坏最后使用约束，不能为了少一轮等待这样修改。

## 成本与下一步

实机只加载一次现有权重，无下载、无重复启动。模型构建、加载和分词合计约 5.15 s；六次预热窗口合计 4.44 s，十八次正式窗口合计 12.47 s，二十四次缓存与执行器初始化合计约 0.05 s。合计 344 次 forward。这些分项不包括进程启动、报告落盘和结果收集，不能当作整个任务的墙钟时间。

这一轮保留两条执行路径，补上可重复比较和解释，不据此引入融合算子或另一套调度器。若继续优化，先隔离“首轮结果何时提交”这一个问题：在相同批次计划下比较 TTFT 与整批耗时，再决定是否值得改变现有循环。等待准入涉及不同的资源生命周期，作为另一个问题处理。修改前还需让旧报告按其版本源码校验，保留历史证据。

## 自己运行

这个入口在 v0.2.0 发布后加入 main；先更新源码，依赖和固定权重准备见[运行指南](START_HERE.md)。先跑无需权重的小模型检查：

```bash
python -m minideepinfer.benchmark_pipeline \
  --device cpu --repeats 1 --output outputs/pipeline-cpu.json
```

已有匹配 NPU 环境和本地权重时：

```bash
python -m minideepinfer.benchmark_pipeline \
  --model weights/Qwen3.5-2B --device npu:0 --repeats 3 \
  --output outputs/pipeline-npu.json
```

`--repeats` 限定为 1–3；每个测量项结束后先落盘，再检查输出、工作量和资源状态。失败就停止后续测量，已完成的记录继续保留。CPU 用例检查统计与执行逻辑，不能代替 Ascend 910C 的计时结果。

原始报告绑定十五份源码，报告本身的 SHA256 与对应固定 commit 保存在[证据索引](results/pipeline-benchmark/evidence.json)。在具备该历史对象的 Git 检出中核对原版源码，无需启动设备：

```bash
python scripts/check_evidence.py --index docs/results/pipeline-benchmark/evidence.json
```

## English summary

On one logical Ascend 910C device, Qwen3.5-2B FP32 took longer with the current cross-round path in all three short workloads. Median whole-workload time increased by 4.1%, 4.2%, and 7.2%. All 54 measured request executions matched their synchronous outputs, with no failures or leaked cache state.

Both paths use the same pre-tokenized inputs, eight output tokens per request, a common arrival time, fresh caches, and matching budgets. Latencies end at CPU commit after round cleanup; throughput includes final draining. Each workload has one warmup per path and three alternating paired measurements. Every sample is retained in the raw report.

A lower TPOT does not imply faster decoding here: the pipeline delays the first CPU-visible result. Four-request workloads also use 18 forwards instead of 17 because admission happens before the preceding round commits its completion. Token work is unchanged, but batch plans differ. These are whole-path comparisons, not measurements of isolated overlap gains or production serving performance.

The benchmark is available on main after v0.2.0. The commands above run the tiny CPU check or the bounded NPU comparison. No execution-core change or performance claim is added on the strength of this small experiment.
