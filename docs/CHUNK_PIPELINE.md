# 第十四课：发出去了，不等于已经提交了

[返回首页](../README.md) · [上一课：新请求到达](ARRIVALS.md) · [源码映射](SOURCE_MAP.md)

上一课让新 prompt 和旧请求的 decode 合批，但 prompt 仍一次算完。长一点的输入会占住整轮工作；要把它切开，先得回答一个问题：CPU 还在处理前一轮，下一块应该从哪个位置开始？

这次沿用同一个 [`StoppingDecodePipeline`](../minideepinfer/stopping_pipeline.py)，加入每轮输入 token 预算和分块 prefill。先给已有 decode 各一个 token，再计算已有请求的 prompt 块，最后按 FIFO 接纳新请求。没有再增加一个 pipeline 类。

## 一条请求，有三种进度

假设 A 的 prompt 有五个 token，每轮预算四个。第一轮发出前四个，第二轮发出最后一个。第二轮 forward 已经提交时，CPU 才开始处理第一轮结果：

| 此刻看什么 | A 的值 | 能用来做什么 |
| --- | --- | --- |
| `Request.num_cached_tokens` / `output_ids` | 0 / 空 | CPU 已经确认并提交的请求历史 |
| `IssuedProgress.cached_tokens` / `sampled_tokens` | 5 / 1 | 已发出的输入数、将产生采样结果的步骤数；规划后续位置与输出上限 |
| `_last_use[A]` | 第二轮的完成事件 | 判断设备是否已用完 A 的状态，决定何时归还槽位 |

这就是实机回调中记录的值。**已发出五个输入，不表示设备已经完成这五个输入。** 同样，`sampled_tokens=1` 表示已提交一轮需要采样的计算，此时还没向用户提交首 token，当前轮采样也可能尚未发出。

如果拿 CPU 的 0 去规划第二轮，就会重新从 prompt 开头计算。若反过来把已发出的 5 当成完成状态，又可能提前归还设备还在访问的缓存。位置规划、输出提交和资源回收各自需要自己的依据。

[`planning.py`](../minideepinfer/planning.py) 用不到张量。`IssuedProgress` 是不可变快照：每发出一个合法连续步骤，生成下一份计数。中间 prompt 块只增加输入数；最后一块和 decode 才增加采样步骤数。

## 先共用“这一步输入什么”

同步 [`BatchEngine`](../minideepinfer/batch_engine.py) 和跨轮实验现在共用一个小函数：

```python
plan_step(request, slots, start, budget, decode_token=0)
```

它根据明确的起点切 prompt，或构造一个 decode token，并返回位置、写入槽位、上下文和 `is_last_prompt_chunk`。同步引擎传入 CPU 已提交的位置与 token；跨轮循环传入已发出的位置，decode 的零占位随后用前轮设备 token 替换。

这里共用的是 Step 构造。两边仍分别管理准入、停止和在途结果：BatchEngine 保留单批在途限制，跨轮实验保留两轮张量的有效期。分清这层接口后，才容易判断下一步该合并什么，而不是把两个循环硬塞到一起。

## 九轮里，四次没有新 token

实机使用 A、B 两条原始文本，prompt 长度分别为 5、8，各输出四个 token。C 在第 0 轮结果回调中加入，prompt 与 A 相同，输出两个 token。槽位池容量 24，并发上限 2，每轮输入预算 4。

A 预留 8 个槽位，B 预留 11 个，只剩 5 个；C 需要 6 个，所以先排队。实际提交轨迹如下：

| 轮次 | 提交给模型的输入 | 本轮输入数 |
| --- | --- | --- |
| 0 | A prompt 前 4 个，中间块 | 4 |
| 1 | A prompt 最后 1 个；B prompt 前 3 个，中间块 | 4 |
| 2 | A decode；B prompt 再 3 个，中间块 | 4 |
| 3 | A decode；B prompt 最后 2 个 | 3 |
| 4 | A decode；B decode | 2 |
| 5 | B decode | 1 |
| 6 | B decode；C prompt 前 3 个，中间块 | 4 |
| 7 | C prompt 最后 2 个 | 2 |
| 8 | C decode | 1 |

当前轮 forward 发出后处理上一轮结果，再采样当前轮；收尾再处理最后一轮。A 结束后，C 最终复用 A 的 `[0,1,2,3,4,5]`，旧前缀和线性状态已清除。

预算约束的是**本轮输入 token 总数**。它既不等于请求数，也不是四个 token 就耗时相同的保证：decode 读取的历史长度、模型状态和批次形状仍会影响计算。

## 中间块不能拿来继续 decode

中间 prompt 块会产生 logits，但不能把块尾预测当成回答。它后面还有用户原本输入的文本。这里保持每个请求一行结果，中间块用 `-1` 占位，回调看到 `None`，CPU 只提交缓存进度，不追加输出，也不判断 EOS 或输出长度。

批次里可能有的行需要采样、有的行不用。混合时，需要采样的行号在 forward 前上传，由本轮对象保留；forward 后只对这些行做 argmax，再填入完整结果向量。所有行都需采样时，直接使用原路径；整批都是中间块时，不执行 argmax，仍记录拷贝完成事件。采样索引不留到 forward 后才准备，延续[第十课](SAMPLER_INPUTS.md)的职责划分。

下一轮只有 decode 行才能取前轮设备 token。若它的来源还是中间 prompt 块，输入准备立即拒绝，避免把占位 `-1` 送进 embedding。测试也检查无效占位不会推进 CPU 进度。

## 最后一次使用，单独记住

分块和 token 预算会让请求的执行间隔变得不规则。不能只在当前保留的两轮里搜索“谁最后用过 A”：A 的最近一轮可能已经离开这两个对象，但它仍拥有缓存。

因此每次成功发出步骤，都把该请求的 `_last_use` 更新到本轮事件。停止时保存这份引用，事件确认完成后再删除 Runner 前缀、线性状态和槽位。两轮对象负责持有计算所需张量，每请求事件负责确定最后使用边界，两者分工不同。

取消发生在中间块时，也遵守同样的顺序。排空成功再清理；若排空失败，进度、事件、在途张量和槽位继续持有，不能把 Runner 当成空闲状态重用。

## 对照 DeepInfer，验证到哪里

固定参考仍为 `8b161497`。[`Worker.execute_model_step()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py) 提供跨轮执行顺序，[`WorkerStepState` / `MetadataZOS`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/zos_status.py) 提供在途状态和行映射的参考。[`prepare_sampler_inputs()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/model_runner.py) 则帮助区分 prompt 完成与采样。本课的两份进度、FIFO、整份缓存预留和 decode 优先，是教学版自己的具体表达，不是上游完整调度策略或 ZOS 的复现。

2026-10-24，在单逻辑 Ascend 910C 上复用 Qwen3.5-2B 权重，以 FP32、关闭 HF32 检查。Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0；CPU 线程 1，tokenizer 并行关闭。

同步参考直接构造整段 prompt 的 Step，不经过新增规划函数或分块路径：A/B 四轮，C 用独立冷缓存两轮。实验中的中间块 logits 对照整段参考的对应前缀位置。

| 检查 | 结果 |
| --- | --- |
| 九轮、十四组完整 logits | 最大绝对差 `2.6702880859375e-5`，原 `atol=1e-4, rtol=2e-4` 通过 |
| 四行中间 prompt 块 | 原始 token 为 -1，回调为 None，不追加输出 |
| A、B、C 输出 | 4、4、2 个，greedy、位置和槽位与参考一致 |
| 采样行准备、C 复用 A 的槽位 | 九轮准备顺序通过；旧模型状态清除 |
| 中间块取消后注入主机异常、再次复用 Runner | 均通过，最终 24/24 槽位归还，前缀和线性状态为 0 |

[原始报告](results/2026-10-24/qwen35-2b-chunk-pipeline-npu-fp32.json)保留轨迹、逐组误差、三种进度、回收事件及十五份相关源码 SHA。一次程序加载一次权重：同步参考 6 次 forward，主轨迹 9 次，取消异常 2 次，复用 1 次，共 18 次。没有下载权重、重新采集 trace 或测量性能。

新增十一项 CPU 测试。Transformers 5.12.1 环境 134 项全部通过；4.46.3 环境 122 项通过、12 项参考跳过；无模型依赖时 54 项通过、80 项跳过。原到达检查与 BatchEngine 分步检查的 CPU 示例也通过。模型数学、缓存布局和 BF16 结论沿用已有边界。

## 自己运行

已有 PyTorch 时，无需权重先跑随机小模型：

```bash
python -m minideepinfer.verify_chunk_pipeline --device cpu \
  --output outputs/chunk-pipeline-cpu.json
```

按[第四课](QWEN35.md)准备依赖和权重后，在匹配的 Ascend 环境中运行：

```bash
python -m minideepinfer.verify_chunk_pipeline \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/chunk-pipeline-npu.json
```

新请求和取消仍从单线程结果回调进入。下一步先检查不同 prompt 长度与到达时刻下的执行轨迹，找到具体的等待，再决定哪些调度职责值得合并或测量。九轮顺序正确，并不能单独说明吞吐提高。
