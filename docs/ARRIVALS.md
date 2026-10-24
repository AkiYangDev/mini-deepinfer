# 第十三课：旧请求继续生成，新请求怎么插进来？

[返回首页](../README.md) · [上一课](STOPPING_PIPELINE.md) · [源码映射](SOURCE_MAP.md)

前两课已经接通设备 token，并处理了 EOS 和取消。这次让新请求在运行途中加入。难点集中在两个地方：一批输入里，新旧请求的 token 来源不同；请求结束后，槽位也不能立刻交给下一位使用者。

代码继续扩展 [`StoppingDecodePipeline`](../minideepinfer/stopping_pipeline.py)，没有增加新的 pipeline 类。名字沿用上一课，它现在同时处理到达、排队和退出；固定批次实验和默认 BatchEngine 保留各自入口。

## 先排队，再分配槽位

结果回调中可以提交新请求：

```python
def on_result(index, steps, tokens, logits):
    if index == 0:
        pipeline.submit("C", prompt_ids, max_new_tokens=2)
```

`submit()` 检查 token、位置上限、最大缓存占用和请求 ID，然后把请求放进 FIFO 队列。这时状态为 `WAITING`，没有槽位，也没有模型状态。只有并发名额和空闲槽位同时满足，`_schedule()` 才分配完整占用，改成 `RUNNING`。

每条请求预留 `prompt 长度 + 最大输出数 - 1` 个槽位。最后一个输出只返回，不再送回模型，所以少一个。`max_num_seqs` 按仍拥有槽位的请求数计数，包括已经停止但设备还没用完的请求。队头不满足条件时，后面的小请求也等待；这是简单的 FIFO 选择，不做抢占和换出。

取消排队中的请求不需要等待设备，因为它还没有设备使用者。已经出现过的请求 ID 在本次运行中不能再次使用，避免把旧步骤的行映射误认成新请求。

## 这次让 C 复用 A 的位置

真实 2B 检查使用容量 16 的槽位池：A 的 prompt 长度为 5，B 为 3，各最多输出 4 个 token，因此先占用 8 和 6 个槽位，只剩 2 个。

C 在第 0 轮结果回调中加入，使用与 A 相同的五个 prompt token，最多输出 2 个，需要 6 个槽位。它先等待。A 的首 token 被设为受控 EOS，上一课的最后使用事件继续保护已经提交的后继轮次。

实际轨迹是：

| 轮次 | 提交给模型的输入 | CPU 随后处理 |
| --- | --- | --- |
| 0 | A prefill、B prefill | 暂存结果 |
| 1 | A decode、B decode | 第 0 轮：C 入队，A 遇 EOS |
| 2 | B decode | 第 1 轮：丢弃 A 多算的输出，归还 A 的槽位 |
| 3 | B decode、C prefill | 第 2 轮：提交 B 的输出 |
| 4 | C decode | 第 3 轮：B 到长度上限，提交 C 的首 token |
| 收尾 | 无新 forward | 收取 C 的最后一个输出 |

这里统一在结果处理阶段收集已经完成的释放事件，下一次调度再考虑新请求。没有为了抢一个更早的准入时刻在循环顶部轮询事件，轨迹也更容易核对。

C 最终得到槽位 `[0,1,2,3,4,5]`，属于 A 先前持有的位置。进入 C 的 prefill 前，A 的 Runner 前缀和线性状态已经删除，C 也没有旧状态。新 prompt 覆盖自己的位置，attention 只读取本请求声明的上下文；物理存储复用不代表继承 A 的计算历史。

## 一批 token 有两种来源

第 3 轮前，上一批只有 B；本批是 `[B, C]`，请求行映射为 `[0, -1]`。B 使用前一批设备采样结果的第 0 行，C 使用新 prompt。`-1` 表示新请求，不能直接送进 `index_select()`。

公共的 [`_launch()`](../minideepinfer/decode_pipeline.py) 先找出 decode 行，再从前一批取它们的设备 token。新 prompt 和 decode 的零占位一起构造打包输入，然后只把 decode 位置替换为设备采样值：

```text
准备的打包输入：[0, C0, C1, C2, C3, C4]
设备上替换之后：[B 的前轮 token, C0, C1, C2, C3, C4]
```

输入的行数是 2，但打包 token 数是 6。目的位置按前面各请求的输入长度累加，不能把请求行号直接当成 token 偏移。CPU 测试还检查了 prefill 在前、decode 在后的排列，以及缺少前轮来源的 decode 在 gather 前被拒绝。

纯 decode 仍直接 gather 前轮设备 token，纯 prefill 仍按原方法准备。只有混合批次走新增的 `ModelRunner.prepare_mixed_batch()`；它与原接口共享位置、槽位、前缀和隔离 mask 的检查。这个设备输入接口是内部受控路径，张量由有效 prompt 和前轮 argmax 构成。

这一过程没有先读回 B 的 token 再上传。CPU 回归在混合准备期间禁止读取前轮的主机 token，检查仍通过。不过 prompt、位置、映射行号和 mask 仍要准备，eager 提交也可能等待设备，这里没有加速结论。

## 循环不能再只数全局轮次

旧实验的请求一起开始，可以统一跑四轮。新请求晚到，可能需要在初始请求结束后继续运行。本课最初用 `_issued` 分别记录每个请求已经提交了几次计算，调度不会为它提交超过最大输出数的输入。[第十四课](CHUNK_PIPELINE.md)将它扩展为输入数和采样步骤数两份计数，以容纳不采样的中间 prompt 块。

这个计数与 `Request.num_cached_tokens` 不同：前者记录已经发给设备的工作，后者在 CPU 提交结果时更新。不能用落后一轮的 CPU 进度来规划下一轮位置。

暂时选不出新批次时，循环先收取仍在途的结果，再尝试准入。这既让最后一个输出正确提交，也能释放等待请求需要的槽位。最后一个结果回调仍可加入新请求；单请求运行中，D 只输出一个 token 后，E 再完成两轮计算的检查也已通过。

请求到达和取消目前都限定在同一个 CPU 线程的结果回调里。没有后台线程、网络服务或无限输入流；本课轨迹将整个 prompt 一次 prefill，不使用 token 预算或分块。后续能力见第十四课；默认 BatchEngine 的单批限制保持不变。

## 对照 DeepInfer 的哪部分

固定参考版本仍为 `8b161497`。[`MetadataZOS`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/zos_status.py) 将当前请求映射到前一轮，`-1` 表示新行，并传递前轮设备采样结果；[`Worker.execute_model_step()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py) 保留本轮 forward、上一轮结果处理、本轮采样的顺序。

本课借鉴这些输入依赖与步骤生命周期。FIFO 准入、整份缓存预留、同批混合 prefill/decode 和回调到达是教学版的具体取舍，不代表复现了上游调度策略、长度修正、图执行或完整 ZOS。

## Ascend 910C 验证

2026-10-24，复用已有 Qwen3.5-2B 权重，以 FP32 在单逻辑 Ascend 910C 上检查。软件为 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0；CPU 线程 1，tokenizer 并行关闭，槽位容量 16。

同步 A/B 四轮先完整执行，C 另用独立的全新缓存运行两轮。随后实验共用只读权重，按上面的到达轨迹执行。A 使用基线首 token 11751 作为设定的 EOS，不是自然文本终止。

| 检查 | 结果 |
| --- | --- |
| 五轮、八组完整 logits，包括 A 多算的一行 | 最大绝对差 `1.9073486e-5`，原 `atol=1e-4, rtol=2e-4` 通过 |
| A、B、C 输出 | 分别 1、4、2 个 token；greedy、位置和槽位与参考一致 |
| C 的混合输入 | 映射 `[0,-1]`，六个复用槽位来自 A，旧前缀和线性状态已清除 |
| C 的前两次等待观察 | `WAITING`，未分配槽位 |
| 正常释放 | A、B、C 的最后使用事件均已完成 |
| 尾轮加入、排队时回调异常、Runner 复用 | 均通过；最后 16/16 槽位归还，前缀与线性状态为 0 |

[原始报告](results/2026-10-24/qwen35-2b-arrivals-npu-fp32.json)保留批次、逐组误差、行映射、槽位归属和 13 份执行源码 SHA。一次程序加载一次已有权重：同步基线 6 次 forward，混合到达 5 次，尾轮加入 3 次，异常清理 2 次，复用 1 次，共 17 次。没有下载或采集 trace；进程退出后项目容器只剩原有保活进程。

新增 9 项 CPU 测试。Transformers 5.12.1 环境 123 项全部通过；4.46.3 环境 111 项通过、12 项独立参考跳过；无模型依赖时 51 项通过、72 项跳过。固定执行和退出实验的两个 CPU 对照入口也分别通过，保留原来的结果边界。

## 自己运行

在仓库目录、已有 PyTorch 的环境中，无需权重先跑随机小模型：

```bash
python -m minideepinfer.verify_arrivals --device cpu \
  --output outputs/arrivals-cpu.json
```

按[第四课](QWEN35.md)准备依赖和权重后，在匹配的 Ascend 环境中运行：

```bash
python -m minideepinfer.verify_arrivals \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/arrivals-npu.json
```

[第十四课：发出去了，不等于已经提交了](CHUNK_PIPELINE.md)接着共用 Step 构造，明确已发出、CPU 已提交和最后使用完成的区别，再加入 token 预算与分块 prompt。
