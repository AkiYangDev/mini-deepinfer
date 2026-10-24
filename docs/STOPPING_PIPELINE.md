# 第十二课：请求结束了，已经发出去的下一轮怎么办？

[返回首页](../README.md) · [上一课](DECODE_PIPELINE.md) · [验证记录](VALIDATION.md)

上一课让下一轮直接读取设备 token，CPU 随后再处理上一轮输出。固定长度时，顺序很清楚。加入 EOS 后，一个问题就出现了：CPU 刚读到“请求已经结束”，下一轮可能已经把这个 EOS 当作输入，写进了 KV Cache 和 DeltaNet 状态。

本课增加 [`StoppingDecodePipeline`](../minideepinfer/stopping_pipeline.py)，约 110 行，复用上一课的模型执行、采样、设备行映射和两轮状态持有。所有请求仍在开始时一起进入，但可以因 EOS、长度上限或取消而退出。已有 BatchEngine 和默认生成入口保持原样。

## 先分清结果和计算

请求 A 的第一轮采样得到 token 11751。我们把这个已观察到的 token 设为 A 的 EOS，专门检查停止流程；它不是模型原生 EOS，也不是一次自然文本结束的检查。

执行顺序如下：

```text
第 0 轮：A、B、C 做 prefill，采样 token 留在设备上
第 1 轮：C、B、A 读取上一轮设备 token，forward 已提交
CPU 收取第 0 轮：发现 A 的 token 等于设定的 EOS
```

A 的 EOS 应当返回给调用方。第 1 轮 A 的计算已经提交，不能靠修改 Python 状态把它撤销。因此我们让它完成，但丢弃它的输出。从第 2 轮开始不再选 A。

B 的取消稍有不同：在收取第 1 轮结果的回调里取消，此时第 2 轮 forward 已经提交。B 的第 0 轮输出早已返回，应当保留；正在收取的第 1 轮输出和已经发出的第 2 轮输出都丢弃。从第 3 轮开始不再选 B。

| 请求 | 停止时机 | 返回的输出 | 已经提交但不再返回的结果 |
| --- | --- | --- | --- |
| A | 第 0 轮结果是 EOS | 第 0 轮 token，含 EOS | 第 1 轮 |
| B | 第 1 轮提交结果前取消 | 先前已提交的第 0 轮 token | 第 1、2 轮 |
| C | 达到四个输出 token | 第 0—3 轮 | 无 |

CPU 进度只记录已经提交的结果。实际设备缓存可能走得更远：本次 B 的主机缓存进度保持 3 个 token，而设备已写到 5 个位置。请求结束后会释放整份状态，不把这两个额外位置当成有效输出，也不尝试恢复到旧状态继续生成。

## 把“停止选择”和“归还缓存”拆开

`_select()` 只保留 `RUNNING` 请求。全体请求的预设顺序仍可换行，筛掉已停止成员后再按请求 ID 映射前一轮设备 token：

```text
[A, B, C] → [C, B, A] → [B, C] → [C]
```

停止选择可以立即生效，归还缓存还要等最后一位设备使用者。`_retire()` 从保留的两轮状态中，找到最后包含这个请求的一轮，把它的 `ready` 事件记下来。这个事件记录在该轮 forward、采样和 D2H 之后，足以覆盖本实验对缓存的最后使用。

一个容易忽略的时刻是：CPU 在处理上一轮时，本轮已经 forward，但尚未执行采样，也就尚未记录本轮事件。如果直接把未记录的原始设备事件当成“已经完成”，就会提前释放缓存。本项目复用第六课的 `RecordedEvent`，未记录时 `query()` 始终返回 False。

```python
last = next(s for s in reversed(self._states) if request_id in s.request_ids)
self._retired[request_id] = last.ready

# 稍后收集；没有完成就继续保留。
if last_use.query():
    runner.release(request_id)
    pool.release(request_id)
```

停止请求所在的那一轮仍执行原来的 greedy 采样，只是后续不会提交其输出。这样无需引入另一套采样分支。记录事件后，存活请求可以继续；已经停止的请求等事件确认完成再归还自己的槽位。最后所有请求都停了，也要收取已经提交的尾轮。

本次日志里还有一个具体例子：B 的最后使用事件未记录时，空闲槽位从 39 增加到了 47。这并不是 B 被提前释放，而是 A 的 8 个槽位刚好可以归还。检查资源安全需要看“B 的槽位是否仍归 B”，不能只看全局空闲数。原报告同时保留了每个请求的持有状态和空闲数。

## 取消发生在什么地方

这个实验只允许在结果回调里调用 `cancel()`，所有调用在同一个 CPU 线程执行：

```python
def on_result(index, steps, tokens, logits):
    if index == 1:
        pipeline.cancel("B")
```

回调先于这批结果提交，所以取消 B 会阻止本轮 B 的输出进入 `Request.output_ids`。先前已提交的输出保留。重复取消返回 False，已经因 EOS 结束的请求也不会再改成取消状态。回调中可以读取诊断结果、取消请求，不能重入 `run()` 或调用 `close()`。

回调看到已停止行时，token 为 `None`，对应的 logits 仍可用于数值诊断。设备上确实算过这些行；`None` 表达的是它们不再作为输出提交。

如果回调抛异常，外层先排空 stream，再清理剩余状态。已经取消的请求保持取消状态，其余仍在运行的请求标记失败。如果连排空也失败，则保留两轮张量、槽位和最后使用事件，阻止再次运行。这个失败分支经过 CPU 注入检查，未声称验证真实设备故障恢复。

## 与 DeepInfer 对应到哪里

固定版本仍是 `8b161497`。[`Worker.execute_model_step()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py) 先执行本轮 forward，再处理上一轮输出。[`WorkerStepState`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/zos_status.py) 要求步骤元数据有效到下一轮结果处理；[`DeferredRelease`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L64) 分开收集待释放对象和实际释放，并另有 clone 完成事件保护。

本课借鉴的是“结果处理晚一轮，资源生命也要跟着使用者走”。这里的每请求最后使用事件、整请求状态释放和回调取消是教学版的选择，不等同于原版完整的调度、块管理、状态池或取消实现。

## Ascend 910C 上的实际结果

2026-10-24，复用已有 Qwen3.5-2B 权重，以 FP32 在单逻辑 Ascend 910C 上检查。软件为 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0；三个 raw prompt 长度为 5、3、8，各最多输出 4 个 token，容量 64，CPU 线程 1。

同步基线先完整跑四轮，随后实验共用只读权重、使用独立缓存。A 使用基线首 token 作为设定的 EOS；B 在第 1 轮结果回调中取消；C 到长度上限。比较保留了已经停止请求多算的那一行，共九组完整 logits。

| 检查 | 结果 |
| --- | --- |
| 四轮成员变化及请求行映射 | 与预期一致，A 输出 1 个、B 保留 1 个、C 输出 4 个 token |
| 九组完整 logits | 最大绝对差 `2.0980835e-5`；原容差 `atol=1e-4, rtol=2e-4` 通过 |
| greedy、位置和槽位 | 对照位置全部一致；提交前取消的 B 输出未被追加 |
| A、B 的未记录最后使用事件 | 槽位和 Runner 前缀均继续保留 |
| 三次正常请求回收 | 最后使用事件均已记录且完成 |
| 所有请求首轮即 EOS | 只执行两次 forward，尾轮输出全部丢弃，没有第三轮 |
| 取消后回调异常、随后复用 Runner | 清理通过；最终 64/64 槽位归还，前缀与线性状态为 0 |

批次缩小后，计算形状与完整批次基线不同，本课沿用原来的数值容差检查。[原始报告](results/2026-10-24/qwen35-2b-stopping-pipeline-npu-fp32.json)保留逐组误差、请求记录、释放事件和 13 份执行源码 SHA。

本次只启动一次验证程序、加载一次已有权重，没有下载或新采集 trace。同步基线 4 次 forward，混合退出 4 次、全部 EOS 2 次、异常清理 2 次、复用 1 次，共 13 次。结果收回时模型进程已退出，项目容器只剩原有保活进程。这里验证的是退出与回收，没有性能计时或加速结论。

新增 11 项 CPU 测试，覆盖逐步缩批、EOS 和长度边界、首次提交前取消、保留旧输出、尾轮收取、回调异常、排空失败保留资源以及非法调用。Transformers 5.12.1 环境 114 项全部通过；4.46.3 环境 102 项通过、12 项独立参考跳过；无模型依赖时 51 项通过、63 项跳过。

## 自己运行

在仓库目录、已有 PyTorch 的环境中，先用随机小模型运行相同流程，无需权重：

```bash
python -m minideepinfer.verify_stopping_pipeline --device cpu \
  --output outputs/stopping-pipeline-cpu.json
```

按[第四课](QWEN35.md)准备依赖和权重后，在匹配的 Ascend 环境中运行：

```bash
python -m minideepinfer.verify_stopping_pipeline \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/stopping-pipeline-npu.json
```

现在成员可以退出，还不能加入。下一步加入新请求时，需要把“从前轮设备结果取 token”和“用新 prompt 做 prefill”放在同一轮输入准备中，还要证明旧请求归还的槽位已经没有设备使用者。现有 BatchEngine 的单批限制继续保留。
