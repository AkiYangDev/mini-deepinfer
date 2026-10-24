# 第七课：模型已经在算，为什么进度还没更新？

[返回首页](../README.md) · [第六课：NPU 事件](NPU_EVENTS.md) · [源码对应](SOURCE_MAP.md)

前两课用整数把事件和回收顺序讲清楚了。这次换成项目自己的 Qwen3.5 forward，让模型计算、混合缓存和采样真正走一遍。

过去调用 `run_batch()`，会等本轮计算完成，直接拿到 token。现在先拿到一份“这批结果”：里面保存本轮输入、输出张量和完成事件。等结果拷回 CPU，再读出 token、更新请求进度。代码在 [`deferred_runner.py`](../minideepinfer/deferred_runner.py)，原来的同步 Runner 继续作为对照。

## 先跑小模型，看进度在哪里变化

在仓库目录、有 PyTorch 的环境中执行：

```bash
python -m minideepinfer.verify_deferred \
  --device cpu --output outputs/deferred-tiny-cpu.json
```

不传 `--model` 时，程序临时创建一个随机初始化的四层 Qwen3.5，包含线性注意力和完整注意力。无需下载权重；它用于检查执行与缓存，不是聊天模型。

核心用法只有几步。下面是讲解片段，`steps` 沿用前几课的有序 Step 元组：

```python
deferred = DeferredRunner(runner)
pending = deferred.launch(steps)
# 这里可以做不读取本批结果、不修改模型或缓存的 CPU 工作。
pending.wait()
tokens = pending.result()
```

`launch()` 负责准备输入、执行模型、选 token 和提交结果拷贝。正常路径没有显式等待本批设备操作结束；调用返回时，小任务也可能已经完成。因此程序不能靠“它应该还没算完”判断资源是否可用，只能检查完成事件。

`wait()` 等 CPU 结果缓冲区可读，`result()` 才读取并校验整批 token，再提交每个请求的已计算前缀。即使 CPU 路径早已算完，不调用 `result()`，主机进度也不会自动更新。

这里更新的是 Runner 保存的前缀位置和“prompt 是否完成”。Engine 中的 `Request.num_cached_tokens`、`output_ids` 和等待队列仍由原同步引擎管理；本课没有替它们提交结果，验证程序直接构造 Step 来检查这层接口。

## 缓存写了，进度却没提交，这会不会矛盾？

它们描述的是不同的事。模型 forward 会直接写 K/V 和 DeltaNet 状态；这些设备写入已经发生，不能假装没有发生。主机保存的前缀记录则表示：这一批结果已经检查过，可以据此准备下一步了。

所以本版只允许一批在途。在读取结果或放弃整批之前，第二次 `launch()` 会报错，`release()` 也拒绝清理这批请求。否则下一批可能根据旧进度继续输入，而设备缓存其实已经向前走了一段。

包装后的 Runner 由 `DeferredRunner` 独占使用。不要同时绕过包装器调用原 Runner 的 forward、release，也不要修改模型权重或缓存。本例在专用 NPU 流开始前等待当前流已有的工作，保证权重和缓存初始化先完成；输入、logits、采样张量等引用保留到这批操作用完。

## prompt 还没算完，就不要提前输出

验证中的 A 有 5 个 prompt token，按 `[2, 2, 1]` 分块。第一轮 A、B 都只计算中间块，返回 `[None, None]`。这表示缓存向前推进了，但还没有用户可见的输出 token。

设备结果缓冲区用 `-1` 标记中间块。只有 `is_last_prompt_chunk=True` 的行才进入 argmax；`result()` 检查中间块确实保留 `-1`，再转换为 `None`。中间块不消耗输出长度，也不能触发 EOS。即使整批都没有采样，完成事件仍然放在模型计算和结果拷贝之后，不能提前提交缓存进度。

下一轮 B 先完成 prompt，A 继续算中间块；之后一轮同时执行 B 的 decode 和 A 的最后一个 prompt 块。这些计算复用原 Runner 的输入准备、模型和缓存，新增接口没有重新实现 attention 或 DeltaNet。

## 取消 A，B 还要继续

在本例的第四轮，A、B 都已经提交了计算，此时调用：

```python
pending.cancel("A")
pending.wait()
tokens = pending.result()
```

A 的这次输出变成 `None`，B 仍拿到自己的 token。为了区分“中间块没有 token”和“取消后丢弃 token”，还可以检查这批的 `steps` 与 `cancelled`。取消不会中断已经发出去的设备操作。

A 的设备状态可能已经推进，所以取消后不能直接用旧前缀恢复 A。包装器会拒绝再次提交 A，直到调用 `release("A")` 删除它的前缀和线性状态。逻辑槽位由前几课的 `DeferredRelease` / `SlotPool` 归还。

这个取消接口只处理已经发出的本批结果。等待队列中的请求怎样取消、请求状态何时变成 cancelled，仍由上层引擎决定。

验证程序在读取结果之前尝试回收 A，确实没有释放；读完后才归还 A 的槽位。C 随后使用其中的槽位做分块 prefill，B 同时继续 decode，两者仍与独立的同步 Runner 对齐。

## 发生错误时，先等设备，还是先清缓存？

先等设备。主机代码抛错时，前面提交的设备操作可能仍在读写缓存。这里不能因为 Python 已经退出 forward，就马上删除状态。

异常路径等待本包装器的专用流完成，再删除本批请求的前缀与线性状态。如果输入校验在模型执行前就失败，原有有效前缀会保留；如果已经进入可能写缓存的模型执行，本批请求的状态会清掉，其他请求不受这次清理影响。

`pending.close()` 表示放弃整批结果，也走等待和清理这条路径。它会删除本批请求已经保存的状态，包含之前的前缀；后续需要从 prompt 重新开始。它不是把缓存回滚到调用 `launch()` 前。

槽位里的旧 K/V 数值不必清零，但原请求的前缀记录不能继续使用。新请求获得槽位后，由自己的 prefill 覆盖相应位置。

如果等待本身报错，就继续保留这批对象，拒绝发起新批次或复用资源。不能把“无法确认完成”改写成“已经安全释放”。这种失败处理用 CPU 注入错误检查了控制流程，真实设备故障恢复尚未验证。

## 在 Ascend 910C 上验证真实权重

使用第四课已经准备好的 Qwen3.5-2B，本次执行命令为：

```bash
python -m minideepinfer.verify_deferred \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/qwen35-2b-deferred-fp32.json
```

本次实际环境是 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0，单逻辑设备、FP32，沿用已核验的[固定权重](results/2026-10-24/weights.json)。同步与新接口共享只读模型权重，各自使用独立缓存；输入和批次布局完全相同。

| 检查 | 本次结果 |
| --- | --- |
| 六轮分块 prefill / decode / 换序 | logits 最大绝对差为 0，比较门槛为 `atol=1e-4, rtol=2e-4` |
| 结果读取前的主机前缀 | 每轮都保持原值 |
| 取消 A、C 复用其槽位 | A 的输出丢弃，B、C 的计算继续与同步路径一致 |
| forward 已提交后注入主机异常 | 等待本流结束，再删除本批状态 |
| 最后清理 | 32/32 槽位归还，前缀记录和线性状态均为 0 |

完整输入 token、每轮比较和源码 SHA 见[原始报告](results/2026-10-24/qwen35-2b-deferred-npu-fp32.json)。这次对照专门检查执行与提交方式的变化，参考方是本项目原有同步 Runner；模型数学对独立 Transformers 的验证沿用[之前的记录](VALIDATION.md)，两种检查回答不同的问题。

主机异常是有意在真实 forward 提交后抛出的，不是 NPU 故障。当前只有一批在途，验证程序每轮等待并比较，没有测量调度重叠或性能收益。当前入口为独立实验，生成命令和 BatchEngine 仍走同步接口。

## 回到 DeepInfer 的主线

DeepInfer 的 [`ModelRunner`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py)区分输入准备、主模型执行与采样，[`WorkerStepState`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/zos_status.py)保存跨轮仍需使用的状态。本课把这些职责放进现有计算路径，让“设备已经执行”和“主机可以提交”在代码里分开。单批限制、Python 句柄和异常清理策略是本项目的教学选择，没有复现上游 ZOS。

下一步再决定调度器怎样保存这份结果、在等待期间准备哪些不依赖新 token 的工作。需要扩大并行范围时，还要重新检查张量使用者与释放事件，不能简单删掉单批限制。
