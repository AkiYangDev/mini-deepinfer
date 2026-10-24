# 第九课：等待结果时，CPU 能先做什么？

[返回首页](../README.md) · [第八课：计算途中取消](DEFERRED_ENGINE.md) · [源码对应](SOURCE_MAP.md)

上一课把发起计算和收取结果拆开了。现在两个调用之间有了一段位置，但它能放什么工作？放进去以后，会不会更快？

这次选择新请求的分词。它只读取新文本和 tokenizer，不需要当前请求刚生成的 token，也不修改模型或设备缓存。相比之下，当前请求的下一次 decode 必须等结果，不能提前准备成一次完整的执行。

我们实现了一个小循环，并在 Ascend 910C 上做了对照：新顺序的批次、输出与回收都一致，**这组短请求没有显示提速**。三步时间线也只有一步观察到分词与设备 kernel 活动重叠。下面沿着代码看原因。

## 每轮只准备一条新请求

入口在 [`text_loop.py`](../minideepinfer/text_loop.py)。它接收一个文本迭代器，先准备第一条请求，后续每轮再取一条。不会在模型开始前把整个迭代器读完，也没有增加线程或另一套调度器。

核心顺序是：

```python
# 讲解片段：省略迭代器结束判断和错误处理。
engine.launch_step()
tokens = tokenizer.encode(next_text)
engine.submit(tokens, max_new_tokens=4)
engine.finish_step()
```

新请求只是进入等待队列。等当前结果处理完，下一轮仍由原来的 `_schedule()` 决定它能否进入批次。decode 优先、token 预算和最大槽位预留都沿用旧规则。

这很适合检查第八课的接口是否真的有用：CPU 可以准备 B，同时保留 A 的在途状态；B 无须知道 A 的回答。但调用顺序允许并行，不代表这次执行一定发生了并行，还要看设备时间线。

## 先在 CPU 上检查两种顺序

在仓库目录、有 PyTorch 的环境中运行：

```bash
python -m minideepinfer.benchmark_prepare \
  --device cpu --repeats 1 --output outputs/prepare-cpu.json
```

不传 `--model` 时，使用随机初始化的小 Qwen3.5 和示例 tokenizer，不下载权重。它用于检查流程，CPU 的计时不能说明 NPU 重叠。

对照只改变准备新文本的位置：

| 顺序 | 每轮做什么 |
| --- | --- |
| `after_finish` | 发起当前批次 → 收取结果 → 为一条新请求分词并入队 |
| `before_finish` | 发起当前批次 → 为一条新请求分词并入队 → 收取结果 |

两种顺序都使用 DeferredRunner，最多一批在途；新请求都在下一轮才参与调度。这样比较的是 CPU 工作放置的位置，不会混入同步 Runner 与新 Runner 的实现差别。

自动检查还覆盖了一种容易忽略的情况：B 的文本非法或超出容量，A 已经在算了。这里用 `finally` 先收取 A 的本轮结果，再把 B 的错误交还调用方。A 保留有效进度，调用方可以继续完成它。如果收取本身也失败，则按第八课的规则保留资源，不能因为前端报错就释放设备仍在使用的缓存。

## Ascend 910C 上测到了什么？

本次使用固定版本的 Qwen3.5-2B 文本权重、单逻辑设备、FP32。软件为 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0，CPU 计算线程数为 1，关闭 tokenizer 并行。

四条 raw prompt 的长度分别是 5、3、8、8 token。每条固定输出 4 token，关闭 EOS 提前停止；并发上限为 2，单轮输入预算为 4 token，缓存容量为 128。没有共享前缀复用或投机解码，也没有 HTTP 请求。命令为：

```bash
python -m minideepinfer.benchmark_prepare \
  --model weights/Qwen3.5-2B --device npu:0 --repeats 3 \
  --output outputs/qwen35-2b-prepare.json
```

每种顺序先完整预热一次，再按先后、后先、先后的顺序交替测量，共六次正式运行。每次都从独立的空请求缓存开始，使用相同权重、输入和调度计划，结束后归还 128/128 槽位，前缀与线性状态均为 0。

| 指标 | 收取后再准备 | 收取前准备 |
| --- | ---: | ---: |
| 四请求总耗时，ms | 636.574 | 639.394 |
| 输出吞吐，token/s | 25.135 | 25.024 |
| 四请求分词合计，ms | 0.412 | 0.421 |
| 请求平均 TTFT，ms | 193.585 | 195.160 |
| 请求平均 TPOT，ms/token | 48.982 | 49.272 |
| 请求平均 E2E，ms | 340.529 | 342.977 |
| 每次输出 / 失败请求 | 16 token / 0 | 16 token / 0 |

表中每项取该顺序三次运行的中位数；延迟项先在一次运行内对四个请求求均值，再取三次中位数。TTFT/E2E 从该文本开始分词计起，到 CPU 提交首个/最后一个输出 token；TPOT 为 `(E2E-TTFT)/3`。逐请求 ITL、全部原始测量值和输出保存在[结构化结果](results/2026-10-24/qwen35-2b-prepare-npu-fp32.json)，重复的批次计划只存一份。

计时包含分词、排队、生成和请求状态回收，不含权重加载、模型预热与缓存分配。两种顺序各有一次预热，计时程序与后面的 profiling 分进程运行，因此也不包含 trace 采集和解析成本。实机程序完成后均退出，没有保留计算进程。

这里没有可报告的加速。三次样本不足以把约 0.44% 的耗时差异归因为某项机制，但能看出这四条短文本的分词只占约 0.4 ms，远小于约 640 ms 的整轮工作。即使它们都被设备计算覆盖，这一项本身也没有多少可节省的时间。长文本或其他前端工作需要各自测量，不能套用本例。

## 调用顺序变了，设备真的同时在算吗？

另外采了预热后的前三步，CPU/NPU 同时记录，Level0、关闭 shape 和堆栈。采集使用独立的 [`examples/profile_prepare.py`](../examples/profile_prepare.py)，没有改写原计时程序：

```bash
python -m examples.profile_prepare \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/prepare-profile
```

在导出的 `ASCEND_PROFILER_OUTPUT` 目录中找到 `trace_view.json` 和 `kernel_details.csv`，将该目录记为 `TRACE_DIR`。在仓库目录的 Bash 中执行：

```bash
python -m minideepinfer.analyze_prepare_trace \
  --trace "$TRACE_DIR/trace_view.json" \
  --kernels "$TRACE_DIR/kernel_details.csv" \
  --output outputs/prepare-timeline.json
```

分析器将 `mini.tokenize` 的 CPU 区间与设备 kernel 区间求交。多个 kernel 并行或重复出现时先做区间合并，不把时长重复相加。本次 8,609 个 CSV kernel 的起始时间均能在同一 trace 时钟找到；两个导出的 duration 最大相差 0.001 µs，分析器按这一导出精度核对，计算仍使用 CSV 原值。

| 采集步骤 | 分词区间，µs | 与设备 kernel 区间相交，µs |
| --- | ---: | ---: |
| 0 | 191.59 | 191.59 |
| 1 | 190.57 | 0 |
| 2 | 186.19 | 0 |

第一步确实有重叠，后两步没有。本次导出的设备事件缺少 `cat="kernel"`，不能只按这个字段过滤，否则会把设备活动误算成零；分析器使用 kernel 表，并交叉核对 trace 的 `Computing` 事件。精简后的[时间线证据](results/2026-10-24/qwen35-2b-prepare-timeline.json)保留步骤区间、分词附近的设备区间、原始文件 SHA 和采集配置。

在这份 trace 中，`launch_step()` 的 CPU 调用本身约为 63–72 ms，而新文本分词只有约 0.19 ms。模型计算是调用期间逐项提交的，等函数返回时，设备可能已经完成了其中的大部分工作。正常 launch 路径没有显式等待整批完成，但其中的具体运算仍可能同步，也无法保证下一个 CPU 操作总有足够的设备工作可以重叠。

这些区间受 profiler 影响，只用于判断本次捕获中是否相交，不能拿来替代上表的不采集计时。它们也没有独立证明 `launch` 较长的具体根因；下一步应检查其算子提交和同步点，而不是直接增加并行层数。

## 在生成入口使用

`--prepare-during-step` 会自动启用分步 BatchEngine。默认生成方式保持不变：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "Explain what a KV cache does." \
  --prompt "What is 1 plus 1?" --max-new-tokens 32 \
  --cache-capacity 256 --max-num-seqs 2 --max-batched-tokens 4 \
  --chunked-prefill --prepare-during-step
```

这个模式从第一条请求开始执行，后续 prompt 逐条加入，因此和默认的“先把所有文本准备好再开始”可能形成不同批次。不要拿这两条命令的耗时直接当成本文的位置对照；上面的 benchmark 专门对齐了到达轮次和批次计划。

## 回到 DeepInfer 的设计

固定版本 DeepInfer 的 [`ModelRunner`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py)区分输入准备、主模型执行与采样，[`WorkerStepState`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/zos_status.py)保留跨轮仍要使用的信息。拆分职责后，才能讨论哪些工作依赖新 token、哪些可以提前做。

本课的文本入口是教学选择，没有复现上游 ZOS。它验证了一个具体的独立 CPU 工作可以放在哪里，也记录了收益很小、重叠不连续的实测结果。[第十课](SAMPLER_INPUTS.md)接着检查 `launch` 尾部的采样索引复制，做一个只移动准备位置的对照。本课报告继续对应当时的旧实现。
