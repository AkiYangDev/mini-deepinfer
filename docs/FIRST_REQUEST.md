# 第一课：跟踪一个请求

[返回首页](../README.md) · [架构](ARCHITECTURE.md) · [开发路线](ROADMAP.md)

先运行，再沿着一次请求读代码。Python 3.10+，无需 PyTorch 或模型权重：

```bash
python -m minideepinfer
```

当前是控制流程模拟。`CountingRunner` 把最后一个输入 token 加一，方便逐步跟踪；它没有模型、tokenizer 或张量 KV。Ascend 910C 是后续真实执行路径的目标平台。

## 1. 提交请求

入口是 [`demo.py`](../minideepinfer/demo.py)：

```python
from minideepinfer import Engine
from minideepinfer.demo import CountingRunner

engine = Engine(CountingRunner(), cache_capacity=8)
request = engine.submit([10, 11], max_new_tokens=3)
for output in engine.run():
    print(output)
print(request.output_ids)  # [12, 13, 14]
```

提交时请求处于 `waiting`，只进入队列。`Engine.step()` 选择它后，预留最多需要的 4 个槽位，将状态改为 `running`。多个请求按 FIFO 顺序逐个运行；当前没有 batching。

为什么是 4 而不是 5？每次计算消费输入并预测下一个 token，最后一个预测结果直接返回，不再送进模型。因此最大计算输入数是 `len(prompt) + max_new_tokens - 1`。

如果最大需求超过总容量，`submit()` 立即报错。排队请求不占用槽位，前一个请求退出后才为后一个分配。

## 2. prefill 与 decode

[`Engine.step()`](../minideepinfer/engine.py) 构造不可变的 `Step`，交给 Runner：

| 步骤 | phase | input_ids | positions | slot_mapping | context_slots | 输出 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | prefill | `(10, 11)` | `(0, 1)` | `(0, 1)` | `(0, 1)` | `12` |
| 2 | decode | `(12,)` | `(2,)` | `(2,)` | `(0, 1, 2)` | `13` |
| 3 | decode | `(13,)` | `(3,)` | `(3,)` | `(0, 1, 2, 3)` | `14` |

`positions` 是序列中的绝对位置；`slot_mapping` 是本轮输入对应的写入位置；`context_slots` 按序列顺序给出历史与本轮输入的缓存位置。未来真实 Runner 可以据此写入新 K/V，并读取当前因果范围内的历史。

这里最容易混淆的是 **输出进度与缓存进度**。第三步结束时，输出是 `[12, 13, 14]`，成功计算过的输入是 `[10, 11, 12, 13]`。`14` 尚未成为计算输入，所以没有它的 KV。Engine 用 `output_ids` 和 `num_cached_tokens` 分别记录两种进度。

## 3. 提交、停止与回收

Runner 返回一个已经选出的 token，Engine 再推进计数并检查停止条件：

- 命中 `eos_token_id`，原因是 `eos`。演示保留 EOS 在 `output_ids` 中。
- 达到 `max_new_tokens`，原因是 `length`。
- 调用 `cancel(request_id)`，状态改为 `cancelled`，支持排队时和两步计算之间取消。
- Runner 抛错，状态改为 `failed`，保存错误摘要并向调用者重新抛出；调用者处理后可以继续运行剩余请求。

完成、取消和失败都会释放该运行请求的槽位。[`SlotPool`](../minideepinfer/kv_cache.py) 用空闲堆和归属表表达最基本的资源所有权。它没有张量，不是 paged attention；槽位位置也不保证在物理存储中连续。

整个接口只在单线程顺序调用。Runner 返回或抛错前必须完成本轮工作；未来加入异步设备执行时，需要先加入事件与延迟释放协议。终止后的 `num_cached_tokens` 是历史进度，不代表仍占有对应缓存。请求记录保留在 Engine 内，当前用于小型教学运行。

## 4. 验证执行逻辑

```bash
python -m unittest discover -s tests -v
```

[`tests/test_engine.py`](../tests/test_engine.py) 检查停止、取消、失败、容量与槽位复用。另一个测试 Runner 用槽位保存前缀和，逐步消费输入，再与每步完整重算对照。这样可以发现位置偏移、漏消费 token，以及旧槽位污染后续请求等问题。

这只证明当前元数据和生命周期逻辑。下一课将接入 DeepSeek-R1-Distill-Qwen-1.5B 的真实 forward 与张量 KV，并对照参考模型检查 logits；Ascend 910C 环境和基础运算也在这一阶段验证。
