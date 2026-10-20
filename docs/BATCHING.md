# 第三课：让请求进入和离开批次

[返回首页](../README.md) · [第二课](REAL_GENERATION.md) · [源码对应](SOURCE_MAP.md)

单请求时，Engine 每轮只推进一个请求。现在 A、B 同时执行，A 先结束，C 随后进入：下一轮的执行成员就从 `[A, B]` 变成了 `[B, C]`。我们需要保持 B 的进度，把 A 的槽位交给 C，并将每一行模型输出交还给正确的请求。

## 先看执行轨迹

Python 3.10+，无需模型或依赖：

```bash
python -m minideepinfer.batch_demo
```

其中三轮分别是：

```text
batch: req-0/prefill, req-1/prefill
batch: req-1/decode, req-2/prefill
batch: req-1/decode, req-2/decode
...
Free slots: 10/10
```

第一轮 req-0 只生成一个 token 就结束。第二轮 req-2 用它释放的槽位进行 prefill，req-1 同时继续 decode。演示中的 Runner 仍是整数模拟；真实模型的计算方式如下。

## 调度这一轮的输入

[`BatchEngine`](../minideepinfer/batch_engine.py) 与第一课的 Engine 共用请求结构和提交校验，保留独立执行循环，方便对照阅读。

每轮先为已有运行请求各保留一个 decode token 的计算预算，再按 FIFO 接纳新请求。`max_num_seqs` 限制运行请求数，`max_num_batched_tokens` 限制本轮输入 token 总数，后者必须至少等于前者。

当前每个新请求一次性计算完整 prompt，并预留 `prompt长度 + max_new_tokens - 1` 个 KV 槽位。队头暂时放不下时，后面的新请求也等待，已有 decode 继续推进。单个 prompt 超过总 token 预算时直接拒绝，避免永远排队。这里没有 chunked prefill、抢占或共享前缀。

`_schedule()` 返回有序的 `tuple[Step, ...]`。每个 Step 的绝对位置从自己的请求计算，不能用拼接后 token 在大张量中的下标代替。

## 合并计算，隔开 attention

[`ModelRunner.prepare_batch()`](../minideepinfer/model_runner.py) 将各 Step 的输入拼接成一个 token 维度。线性层、归一化和 MLP 在整个张量上执行；一轮只调用一次模型。每个请求在 K/V 张量中仍有独立的槽位。

attention 同时遵守两条规则：query 只能看同一请求的 key，并且 key 的绝对位置不能晚于 query。例如输入是 `[A0, A1, B0]`，允许读取的位置是：

| query / key | A0 | A1 | B0 |
| --- | --- | --- | --- |
| A0 | 是 | 否 | 否 |
| A1 | 是 | 是 | 否 |
| B0 | 否 | 否 | 是 |

代码里的布尔 mask 用 `True` 表示屏蔽，与表中“允许”相反。decode 的 query 只有一个新 token，key 则包含该请求此前的全部上下文。

这是短上下文的参考路径：分数矩阵大小为 `所有query数 × 所有key数`，包括随后被 mask 掉的跨请求项。它展示真正合并的模型计算，但不是高效 varlen 或 paged attention 实现，也没有由此获得提速的测量结论。

## 统一提交与回收

Runner 按调度顺序取每个请求的最后一行 logits，greedy 选择 token，一次传回主机。Engine 先检查输出数量和类型，再提交整批请求的进度。

一项输出无效或模型抛错时，本轮所有已调度请求都失败，本轮没有部分输出提交；等待队列保留。Runner 必须在返回或抛错前完成设备工作，Engine 才能释放槽位。当前只能在同一线程、两轮之间调用取消。

请求正常结束或取消后，Engine 还会调用 `Runner.release()` 删除按请求保存的前缀记录。K/V 张量中的旧数据不必清零，新请求 prefill 会写入自己的输入；已经释放的前缀不能再用于 decode。

## 运行多个真实请求

按第二课准备环境和权重，重复传入 `--prompt`：

```bash
python -m minideepinfer.generate \
  --model weights/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu --dtype float32 --raw \
  --prompt "The capital of France is" --prompt "A KV cache" \
  --max-new-tokens 16 --cache-capacity 128 \
  --max-num-seqs 2 --max-batched-tokens 32
```

只有一个 prompt 时仍使用第一课的 Engine，多个 prompt 使用 BatchEngine。文本在整轮执行结束后打印，不是流式服务。比较固定 token 的 logits：

```bash
python -m minideepinfer.verify_batch \
  --model weights/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu --dtype float32 --output outputs/batch-cpu.json
```

测试覆盖混合 prefill/decode、请求顺序变化、脏槽位复用、取消、预算与缓存不足、异常和批次内的数值隔离。真实权重与目标设备的完成情况以[验证记录](VALIDATION.md)为准。

验证脚本保留两种参考：独立请求，以及按相同批次与缓存顺序执行的 Transformers 模型。后者通过仅调整存储顺序的 `DynamicCache` 适配器实现，mask 也单独构造。相同布局 FP32 使用 `atol=1e-4`，跨批次形状使用 `atol=2e-4`，两者均为 `rtol=2e-4`；阈值依据、初次未通过记录和最终结果均在验证文档中，避免用放宽容差掩盖布局错误。

## 对照 DeepInfer

本课沿着原版的“调度输出与逐请求元数据 → worker 模型执行 → 按调度结果提交”组织代码，入口见[源码映射](SOURCE_MAP.md)。Python FIFO、全量预留、合并张量的显式 mask，以及在一批里混合 prefill/decode 都是教学选择，不复现原版完整调度策略。

这里仍然每轮等待计算结束，再处理输出。后续学习 `WorkerStepState` 与 `DeferredRelease` 时，才会讨论上一轮结果尚未提交、下一轮批次已经变化的情况；当前立即回收的约定不能直接搬到异步执行中。
