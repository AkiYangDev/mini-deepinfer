# 第八课：计算已经发出，还能取消请求吗？

[返回首页](../README.md) · [第七课：真实模型结果](DEFERRED_RUNNER.md) · [源码对应](SOURCE_MAP.md)

可以。但“取消请求”和“设备停止计算”是两回事。

假设 A、B 已经一起送进 NPU。A 的用户这时断开连接，B 还在等待回答。引擎应该丢弃 A 这一轮的输出，让 B 继续；A 占用的缓存则要等本批计算结束后再归还。否则新来的 C 可能拿到相同槽位，而 NPU 还在往里面写 A 的数据。

第七课已经让模型返回一份待收取的结果。这一课把它接进现有 [`BatchEngine`](../minideepinfer/batch_engine.py)，由同一个调度器管理等待队列、请求进度和槽位。原同步路径继续保留。

## 先看一次完整运行

在仓库目录、已经安装 PyTorch 的环境中执行：

```bash
python -m minideepinfer.verify_deferred_engine \
  --device cpu --output outputs/deferred-engine-cpu.json
```

无需下载权重。程序创建一个随机初始化的小 Qwen3.5，用真实模型计算和混合缓存检查执行顺序。它不能用于聊天。

程序会打印六轮结果：A 先分块读取 prompt，B 随后加入；A、B 一起 decode 时取消 A，再让 C 使用归还的槽位；最后单独检查 EOS。同步与分步执行共享只读权重，各自使用独立缓存，每轮比较 logits 和请求进度。

## 把原来的 step 拆在哪里？

原来的 `step()` 一次完成调度、计算和结果提交。新模式把这次调用拆成两段：

```python
# 讲解片段：runner 是已经加载好模型的 ModelRunner。
engine = BatchEngine(DeferredRunner(runner), cache_capacity=32,
                     max_num_seqs=2, max_num_batched_tokens=4,
                     chunked_prefill=True, deferred=True)
request = engine.submit([2, 3, 4, 5, 6], max_new_tokens=4)

plans = engine.launch_step()
# 此处可以接收新请求，或取消已有请求。
outputs = engine.finish_step()
```

`launch_step()` 仍调用原来的 `_schedule()`，选择请求、预留槽位并提交模型计算。它返回本轮 Step 元组，方便观察调度结果；没有工作时返回空元组，不需要再收取结果。

新准入的请求此时已经是 `RUNNING`，但 `num_cached_tokens` 和 `output_ids` 仍表示上一次成功提交的进度。即使模型已经算完，这两个字段也不会自己变化。

`finish_step()` 等待结果可读，校验整批 token，再更新这些字段。中间 prompt 块只推进缓存计数，最后一块才产生首个输出；decode 消费上一轮 token，再追加新 token。EOS 和长度上限也在这里判断。

这两段之间，仍然由同一个 CPU 线程调用引擎。可以 `submit()` 和 `cancel()`，不能再 `launch_step()`。下一轮 decode 依赖本轮结果，当前版本没有为它猜一个输入。

需要原有的逐轮用法时，继续调用 `step()` 或 `run()`。在分步模式下，`step()` 会连续调用发起与收取，中间立即等待，因此这条便利路径本身不提供调度重叠。

## 取消 A 时，到底改什么？

```python
# 讲解片段：这一批已经发出，request 是其中的请求。
engine.cancel(request.request_id)
outputs = engine.finish_step()
```

`cancel()` 先把 A 标记为 `CANCELLED`，并通知本批结果丢弃 A 的采样。已经发出的设备计算照常完成，A 的槽位和线性状态继续保留。

收取整批结果后，A 对应的 `StepOutput` 包含 `token_id=None`、`finish_reason="cancelled"`。A 的输出列表和已提交缓存计数保持原值，随后删除它的 Runner 状态并归还槽位。B 正常提交自己的输出。C 只能在下一次调度时取得空闲槽位。

等待中的请求还没有设备状态，可以直接取消。在一批在途期间，引擎会把运行请求的清理统一留到该批安全结束后，代码因此不需要再维护一套并行回收队列。

`None` 不能单独说明发生了什么：中间 prompt 块也没有输出。要结合 `phase`、请求状态和 `finish_reason` 判断。把取消当成非法采样，会连带误杀同批的 B；把中间块当成生成结束，又会提前截断 prompt。

## 等待报错，不能当作已经结束

普通主机异常与“无法确认设备完成”需要分开处理。

如果模型提交后主机代码抛错，Runner 会先等待自己的流，再清理本批模型状态。确认完成后，引擎把本批未取消的请求标成 `FAILED`，归还槽位；等待队列中的其他请求仍可继续。

如果等待或排空设备工作本身失败，引擎继续持有本批句柄、张量和槽位，拒绝下一次发起。此时 `FAILED` 表示请求不能继续生成，并不表示设备资源已经空闲。

`abort_step()` 用于明确放弃整批：再次等待设备工作结束，成功后清理本批状态和槽位；等待仍失败就继续保留。它会删除本批已有的前缀，不能恢复到发起前接着生成。失败请求需要从 prompt 重新提交。

等待失败、排空失败和恢复清理用受控 CPU 错误测试；真实 Ascend 910C 上只验证了 forward 提交后的主机异常，没有验证设备故障恢复。

## 在 Ascend 910C 上的结果

复用第四课的固定 Qwen3.5-2B 权重，执行：

```bash
python -m minideepinfer.verify_deferred_engine \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/qwen35-2b-deferred-engine-fp32.json
```

本次环境为 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0，单逻辑设备、FP32。六轮共比较十组 logits，最大绝对差为 **0**，沿用 `atol=1e-4, rtol=2e-4`。

| 检查 | 本次观察 |
| --- | --- |
| 分块 prompt、动态加入、混合 decode | 两条路径每轮使用相同 Step；提交前请求进度和主机前缀不变 |
| A 在第三轮取消 | 本轮输出丢弃，原已提交进度不变；收取前槽位没有归还 |
| C 加入、B 继续 | C 在第四轮复用 A 的槽位，B、C 的输出和进度逐轮一致 |
| 长度与 EOS | B、C 按长度结束；另一个相同 prompt 请求在首个采样处遇 EOS |
| forward 后注入主机异常 | 本批请求失败，流完成后清理；32/32 槽位归还，前缀和线性状态均为 0 |

EOS 示例使用 B 已观测的首个输出 token 作为停止符，专门检查停止与回收，不代表模型遇到了自然的文本结束符。同步对照在第三轮先完成 A 再取消；分步路径在收取前取消，所以必须丢弃 A 本轮多算出的 token。后续只比较仍在继续的请求，不把这项预期差别写成精度问题。

实际 token IDs、六轮 Step、每个请求的停止原因及执行源码 SHA 见[原始报告](results/2026-10-24/qwen35-2b-deferred-engine-npu-fp32.json)。同步 Runner 是本轮接口变更的对照，模型数学对独立 Transformers 的验证沿用[已有记录](VALIDATION.md)。

离线生成也可以加上 `--deferred` 使用这条路径，例如：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "What is a KV cache?" --max-new-tokens 32 \
  --cache-capacity 128 --max-num-seqs 2 --max-batched-tokens 4 \
  --chunked-prefill --deferred
```

## 这一课借鉴 DeepInfer 的哪一部分？

固定版本 DeepInfer 的 [`LLMEngine.post_process()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py)通过 `WorkerStepState` 取得本轮元数据和采样输出，处理请求结果后收集待释放资源。源码明确注明，worker 在调用它前已经同步本轮采样完成事件。[`DeferredRelease`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L64)还要照顾调用方继续读取输出的生命周期。

本课沿用“执行结果需要由引擎提交、资源要等使用结束才能释放”这条主线。Python FIFO、一次预留最大槽位、单批在途和显式 `finish_step()` 是教学简化，尚未复现上游 ZOS、跨轮执行重叠或资源池。

下一步应在这两段之间放入确实不依赖新 token 的 CPU 工作，再用实际时间线确认是否与 NPU 计算重叠。本轮先验证接口和生命周期，没有吞吐或延迟结论。
