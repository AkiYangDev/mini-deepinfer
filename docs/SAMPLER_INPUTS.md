# 第十课：两个行号，为什么会等模型？

[返回首页](../README.md) · [第九课：等待期间准备文本](PREPARE_DURING_STEP.md) · [源码对应](SOURCE_MAP.md)

上一课有两个步骤出现了同样的现象：模型发出以后，CPU 没有马上去准备下一条文本，而是在 `launch` 尾部的一次复制调用上停留了约 1.6 ms。代码里，这时正在把采样行号变成设备张量。

这次给索引创建单独加了标记，再用旧、新实现对照，确认了这处调用的位置。将索引提前到 forward 前准备后，采样步骤的主机复制调用从约 1.4–1.6 ms 降到约 42 µs，下一条文本的分词也能接上设备尾部工作。结果收取时仍然需要等待；整轮计时没有足够证据支持稳定提速。

## 行号什么时候就知道了？

一批可能包含 A 的中间 prompt 块和 B 的 decode。A 还不能输出，只有 B 需要采样，因此对应的行号是 `[1]`。这来自本轮 Step 的 `is_last_prompt_chunk` 标记，不需要模型先算出 logits。

旧顺序把这件事留到了模型执行之后：

```python
# 旧顺序的讲解片段。
packed = runner.execute_main_model(inputs)
selected = ...  # 取需要采样的行，计算 argmax。
indices = torch.tensor(rows, dtype=torch.long, device=runner.device)
tokens.index_copy_(0, indices, selected)
```

`torch.tensor(..., device=...)` 从主机数据构建设备张量。在这次环境与调用路径中，它的主机复制调用还包含了等待前面设备工作的时间。trace 中的 `aten::copy_` 耗时不能直接当作传输几个整数的 DMA 时间，更不能据此计算链路带宽。

现在先把不依赖模型输出的部分准备好：

```python
# 新顺序的讲解片段，完整实现见 deferred_runner.py。
inputs = runner.prepare_batch(steps)
rows = [i for i, step in enumerate(steps) if step.is_last_prompt_chunk]
if rows:
    indices = torch.tensor(rows, dtype=torch.long, device=runner.device)
    pending._buffers.append(indices)

packed = runner.execute_main_model(inputs)
# 后续仍按原来的行顺序取 logits、argmax、写回 token 缓冲区。
```

主引擎只改了 [`DeferredRunner.launch()`](../minideepinfer/deferred_runner.py) 的这处顺序，模型数学、缓存布局、调度器和采样规则都沿用原实现。索引仍由在途结果持有，防止设备使用期间丢掉引用。整批都是中间 prompt 块时，不创建采样索引，也不执行采样。

另一个边界也更清楚了：索引准备失败时，模型还没有开始写本轮缓存。测试直接使用 Runner 验证，这时已有前缀和线性状态保持原值；进入模型执行后再失败，仍按原规则等待并清理。上层 Engine 是否将请求标为失败，是另一层生命周期处理。

## 怎样只比较这一项改动？

对照脚本在 [`examples/compare_sampler_order.py`](../examples/compare_sampler_order.py)。旧实现来自固定提交 `84a5f55`，脚本先检查源码 SHA，再只在实验期间切换 Runner 类。两种实现共享一次加载的只读模型权重，每次使用独立请求缓存；输入循环、分词位置和批次预算完全相同。

在仓库目录、有 PyTorch 的 Bash 环境中，先导出旧源码，再运行无需权重的小模型检查：

```bash
mkdir -p outputs
git show 84a5f55e964c4df5b8749ba3ece75bd6d142f653:minideepinfer/deferred_runner.py \
  > outputs/late_runner.py
python -m examples.compare_sampler_order \
  --baseline-source outputs/late_runner.py --device cpu --repeats 1 \
  --output outputs/sampler-order-cpu.json
```

CPU 可以检查输出和状态，不能证明 NPU 复制会等待。真实 Ascend 910C 使用已有 Qwen3.5-2B 权重：

```bash
python -m examples.compare_sampler_order \
  --baseline-source outputs/late_runner.py \
  --model weights/Qwen3.5-2B --device npu:0 --repeats 3 \
  --trace-output outputs/sampler-profile \
  --output outputs/qwen35-2b-sampler-order.json
```

本次为单逻辑设备、FP32，Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0。四条 raw prompt 长度为 5/3/8/8 token，各固定输出 4 token，关闭 EOS；并发上限 2、单轮预算 4 token、容量 128，CPU 线程 1、tokenizer 并行关闭。

程序先各执行一次完整 logits 数值对照，再各预热一次，随后按旧新、新旧、旧新的顺序做三组计时。数值检查有额外的 CPU 拷贝，耗时不进入性能样本。计时完成就先保存报告，之后才为旧、新实现各采三步 Level0 CPU/NPU 时间线；shape、内存和堆栈采集关闭。一次程序只加载一次权重。

13 组完整 logits 的最大绝对差为 **0**，沿用 `atol=1e-4, rtol=2e-4`。两种实现的批次计划、输入输出和停止原因一致；数值、预热、计时及采集轨迹结束后，均归还 128/128 槽位，前缀和线性状态清理完成。

## 等待去了哪里？

在两个 trace 中，采样行号分别位于 forward 之后和之前。每份 trace 的 8,609 个设备 kernel 起始时间都与 CSV 核对，重叠计算沿用第九课的 Decimal 时钟和区间合并方法。

| 采样步骤 | 旧版索引复制调用，µs | 新版索引复制调用，µs | 旧版收取调用，µs | 新版收取调用，µs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1391.66 | 42.42 | 189.13 | 1190.48 |
| 2 | 1554.53 | 42.74 | 182.79 | 1298.04 |

表中复制与收取都是 profiler 记录的主机调用时长，不能相加当成独立设备算子时间。新版复制调用不再等本轮 forward 的尾部，但随后 `finish_step()` 要承担更多结果等待。设备仍然要完成同样的模型计算和采样。

这给 CPU 留出了一段可用间隙。旧版三个分词区间与设备活动相交的时长是 `175.31 / 0 / 0 µs`；新版是 `178.98 / 192.51 / 195.38 µs`，等于这次捕获中的三个分词区间长度。步骤 0 是中间 prompt 块，两版都没有索引上传，作为对照保留。

这些结果将原先的可疑调用对应到了具体源码位置，并显示改变准备顺序后，等待边界与重叠区间随之变化。它们不说明后续每个批次都有足够长的设备尾部，也不代表整个引擎已经实现 DeepInfer 的 ZOS。

精简后的[时间线证据](results/2026-10-24/qwen35-2b-sampler-timeline.json)保存区间、复制位置和原始 trace/CSV 的 SHA。对这次新建的采集目录，可在仓库目录用 Bash 复核：

```bash
python -m examples.analyze_sampler_order \
  --late-trace outputs/sampler-profile/late/*/ASCEND_PROFILER_OUTPUT/trace_view.json \
  --late-kernels outputs/sampler-profile/late/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --early-trace outputs/sampler-profile/early/*/ASCEND_PROFILER_OUTPUT/trace_view.json \
  --early-kernels outputs/sampler-profile/early/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --output outputs/sampler-timeline.json
```

## 整轮有没有更快？

本轮未开启 profiler 的三个配对样本如下，全部保留：

| 配对 | 旧版，ms | 新版，ms |
| --- | ---: | ---: |
| 1 | 678.092 | 675.583 |
| 2 | 677.196 | 675.562 |
| 3 | 677.085 | 724.568 |

新版第三次运行明显更慢，当前没有该次运行的逐阶段数据，原因尚未确定。中位数只相差约 0.24%，三组样本不能支持稳定吞吐改善的结论。

| 指标 | 旧版 | 新版 |
| --- | ---: | ---: |
| 四请求耗时中位数，ms | 677.196 | 675.583 |
| 输出吞吐中位数，token/s | 23.627 | 23.683 |
| 请求平均 TTFT，ms | 206.656 | 206.626 |
| 请求平均 TPOT，ms/token | 52.007 | 51.938 |
| 请求平均 E2E，ms | 362.677 | 362.484 |

延迟项先求一次运行中四个请求的均值，再取三次中位数。从每条文本开始分词计时，到 CPU 提交首个/最后一个 token；TPOT 为 `(E2E-TTFT)/3`。计时包含文本准备、排队、生成和回收，不含权重加载、缓存分配、数值对照、预热及后续采集。逐请求 ITL、0 失败记录和全部样本见[原始报告](results/2026-10-24/qwen35-2b-sampler-order-npu-fp32.json)。

保留这个改动的依据是：索引确实不依赖 logits，提前准备后去掉了 forward 后这一处不必要的主机等待，输出与状态也通过了对照。性能判断则继续限定在实际测到的范围。

## 与 DeepInfer 的对应

固定版本 DeepInfer 的 [`prepare_sampler_inputs()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py#L213)根据 sequence 和 input metadata 准备采样位置，主模型执行与实际采样另有入口。采样要消费 logits，采样所需的位置元数据则是另一类依赖。

这一课把这种职责区分落实到现有小引擎。下一步继续沿同一条执行链检查剩余的主机准备与提交开销；每次先明确依赖、再移动工作，确认结果与生命周期后才讨论计时。
