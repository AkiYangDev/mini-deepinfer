# 首轮结果，为什么要等下一轮？

[返回首页](../README.md) · [前一次对照](PIPELINE_BENCHMARK.md) · [执行顺序出处](SOURCE_MAP.md)

前一次测量里，跨轮路径的首 token 比同步路径更晚交到 CPU。这次只改一件事：第一轮采样完成后先收取结果，再发起第二轮；后面的轮次继续跨轮执行。在 Ascend 910C 的短请求对照中，首 token 确实更早交付，整批耗时则没有稳定改善。

## 改动在哪里

[`StoppingDecodePipeline.run()`](../minideepinfer/stopping_pipeline.py) 新增实验参数 `early_first_result=True`，默认关闭。原来的顺序是：

```text
发起第 0 轮 → 采样第 0 轮 → 发起第 1 轮 → 收取第 0 轮 → 采样第 1 轮
```

打开参数后：

```text
发起第 0 轮 → 采样第 0 轮 → 收取第 0 轮 → 发起第 1 轮 → 采样第 1 轮
```

“收取”仍要等本轮设备事件完成，然后提交 token、处理停止和回收。只是让这一步先发生，没有提前释放还在使用的缓存。第 0 轮的张量状态继续保留，供下一轮读取设备 token；循环记录它已经收取，避免重复提交或重复调用回调。

这里改的是**第 0 轮**，不是自动找到第一个可采样的 prompt 块。若第 0 轮只处理了部分 prompt，它仍没有生成 token。回调也会提前发生，可能改变 EOS、取消和新请求到达后的计划；这些行为有 CPU 检查，但不纳入本次计时对照。

这个顺序变体沿着项目已有的 DeepInfer Worker 执行顺序实验展开。原版入口和教学简化见[源码映射](SOURCE_MAP.md)；它不是完整 ZOS 的复现。实验保留在同一个循环里，没有复制执行器。当前 `generate --pipeline` 继续使用原来的收取方式；新参数可通过 Python API 和下面的基准入口使用。

## 这次怎样排除其他变化

只比较同一份新版源码中的参数关闭与打开，不拿上次运行的数字当基线。输入都是原始文本 `The capital of France is`，分词后 5 token，不加聊天模板；两类负载分别为单请求、两条相同请求。每轮预算提高到 16，让所有 prompt 在第 0 轮处理完，最多 2 条请求，整个过程没有等待准入的新成员。

Qwen3.5-2B 沿用本地固定 revision `15852e8c16360a2fea060d615a32b45270f8a8fc` 的权重，在一个逻辑 Ascend 910C 设备上运行。Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0，aarch64；FP32、关闭 HF32、eager、greedy、CPU 线程 1。容量 128，独立新缓存，无共享前缀或跨次状态复用；固定输出 8 token，关闭 EOS。

每种负载各方式预热一次，再做三组成对计时，先后顺序交替。两种方式每次都必须满足：输出相同、逐轮计划相同、实际输入工作量相同、缓存全部回收。任一条件不符就保存报告并停止。本次每次都是 8 个 forward；单请求处理 12 个输入 token，双请求处理 24 个，没有额外输入计算。

计时沿用前一次的离线口径：所有请求同一到达起点，TTFT 到首 token 在 CPU 提交并完成本轮清理，E2E 到最后一个 token 同样提交；TPOT 为每请求 `(E2E-TTFT)/7`。整批耗时包括最终排空与同步。分词、权重加载和缓存初始化在窗口外。计时期间只记轻量主机元数据和时间，不打印或读取完整 logits。

## 实机结果

逐请求延迟先在每次运行内取均值，再取三次中位数；整批耗时和吞吐分别取三次中位数。

| 请求数 | 首轮收取方式 | TTFT ms | TPOT ms/token | E2E ms | 整批耗时 ms | 输出 token/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 延后 | 84.83 | 29.91 | 294.23 | 294.33 | 27.18 |
| 1 | 提前 | 48.80 | 34.45 | 289.86 | 289.96 | 27.59 |
| 2 | 延后 | 120.97 | 40.09 | 401.68 | 401.78 | 39.82 |
| 2 | 提前 | 73.52 | 46.86 | 401.49 | 401.61 | 39.84 |

TTFT 的中位数分别减少约 36.03 ms、47.45 ms。TPOT 反而增大，因为首 token 提早后，首末 token 之间的统计区间变长。这与上一轮“TPOT 较小但整批更慢”的观察一起说明，应该同时看首 token、末 token 和整体吞吐。

三组成对整批耗时全部保留，每格为“延后 / 提前”，单位 ms：

| 请求数 | 第 1 对 | 第 2 对 | 第 3 对 |
| --- | ---: | ---: | ---: |
| 1 | 290.14 / 289.17 | 294.83 / 289.96 | 294.33 / 295.26 |
| 2 | 447.74 / 403.14 | 401.24 / 401.61 | 401.78 / 400.79 |

单请求第三对、双请求第二对，提前收取的整批耗时略长。双请求还有一次 447.74 ms 的慢样本，照样保留，没有采集 trace 来解释它。因此本次支持“更早交付首轮结果”，不支持稳定吞吐加速，更不能与上一次不同预算、不同批次的结果拼接出倍数。

正式计时共 12 次运行、18 次请求执行，失败 0。两种方式都与本次延后路径的输出和完整计划一致；每次最终归还 128 个槽位，主机前缀与线性状态清空。另有两次单输出 token 的实机检查，覆盖只运行一轮就结束的排空。这里检查 greedy 输出与状态，不新增完整 logits 数值对照。

## 怎么运行、核对

该入口在 v0.2.0 发布后加入 main。已有兼容 PyTorch 时，先运行无需权重的小模型：

```bash
python -m minideepinfer.benchmark_first_result \
  --device cpu --repeats 1 --output outputs/first-result-cpu.json
```

按[运行指南](START_HERE.md)准备好本地权重和 NPU 环境后：

```bash
python -m minideepinfer.benchmark_first_result \
  --model weights/Qwen3.5-2B --device npu:0 --repeats 3 \
  --output outputs/first-result-npu.json
```

`--repeats` 限定为 1–3。每次先落盘，再验输出、计划、工作量和回收。完整数据见[原始报告](results/2026-10-24/qwen35-2b-first-result.json)，报告 SHA 和十六份源码绑定见[索引](results/first-result/evidence.json)。无需设备即可核对当前相关源码：

```bash
python scripts/check_evidence.py --index docs/results/first-result/evidence.json
```

旧报告现在按各自固定 commit 校验，使用方法见[历史证据说明](EVIDENCE.md)。它们仍描述旧版运行，新报告描述这次实验，两者分别保留。

本次只加载一次现有权重，共 130 次 forward。构建模型、加载与分词合计约 5.27 s；两次单 token 检查窗口合计 0.34 s，四次预热 1.39 s，十二次正式测量 4.21 s，十八次缓存初始化合计约 0.04 s。这些分项不包含进程启动、落盘和收集，不是整个任务的墙钟时间。无权重下载、无 profiler 采集。

CPU 检查还覆盖首轮 EOS、取消后剩余请求的 token 行映射、部分 prompt、回调加入新请求、回调报错后的清理和再次运行。通过这些边界后保留实验参数，生成入口的默认行为继续按原教程解释。下一项若研究排队请求准入，应另设共同负载和资源生命周期约束。

## English summary

Collecting round zero before launching round one reduced median CPU-visible TTFT from 84.83 to 48.80 ms for one request, and from 120.97 to 73.52 ms for two fixed requests on one logical Ascend 910C device. Qwen3.5-2B FP32 outputs, complete batch plans, input work and final cache state matched across both modes.

Whole-workload time did not improve consistently across the three alternating pairs. All samples are retained. Earlier delivery also increases measured TPOT by extending the interval from the first committed token to the last; it does not establish slower device decoding.

`StoppingDecodePipeline.run(early_first_result=True)` is an optional experiment. It collects only round zero early, including a partial prompt round if present, and changes the first callback's timing. Later rounds retain the existing order. The generation CLI keeps its existing behavior. The benchmark above runs a tiny CPU check or the bounded real-checkpoint comparison; historical reports are now checked against immutable recorded commits.
