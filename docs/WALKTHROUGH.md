# 跟着 A、B、C 跑完一次调度

[返回首页](../README.md) · [English](WALKTHROUGH.en.md) · [自己的文本](PIPELINE_TEXT.md)

A 的 prompt 较长，B 只生成两个 token，C 在运行中加入。这个例子用项目自己的 Qwen3.5 前向计算，走已有的 `StoppingDecodePipeline`，观察它们怎样合批、等待和交还缓存。

## 运行

安装过兼容 PyTorch 时，在仓库目录直接运行；还没准备环境，先看[安装指南](START_HERE.md#cpu-小模型无需下载权重)。

```bash
python -m minideepinfer.pipeline_demo
```

它在 CPU 上随机初始化小模型，不下载权重。输入和输出都是 token ID，没有文本含义。为固定演示轨迹，关闭 EOS，分别生成 6、2、3 个 token，然后按长度结束。需要保存每次观察的原始数据时：

```bash
python -m minideepinfer.pipeline_demo --output outputs/pipeline-demo.json
```

每轮输入预算为 4 个 token，最多两个请求持有槽位，总容量 20。请求进入时就预留完整的 `prompt 长度 + 最大输出数 - 1` 个槽位；最后一个输出不再送回模型，所以减 1。

| 请求 | prompt 长度 | 输出数 | 需要槽位 | 何时排队 |
| --- | --- | --- | --- | --- |
| A | 9 | 6 | 14 | 开始时 |
| B | 2 | 2 | 3 | 收取第 0 轮时 |
| C | 4 | 3 | 6 | 收取第 1 轮时 |

## 为什么 CPU 是 0，已发出却是 8

开头会看到：

```text
collect 0 | issued through 1
  A prefill 0..3: partial prompt; no sample
  arrived: B
  A: running; cached CPU/issued=0/8; slots=14
  B: waiting; cached CPU/issued=0/0; slots=0
  free slots: 6/20
```

`collect 0` 表示第 0 轮结果已拷回，CPU 即将处理它。此时第 1 轮计算已经提交，因此 A 的两块输入共 8 个 token 都已发出。观察发生在 CPU 提交结果之前，所以 `cached CPU` 仍是 0；处理完这次回调后才变成 4。

`issued` 只表示已提交计算，不表示设备已经完成。在 CPU 上计算实际同步完成；这个例子展示执行顺序，不能据此判断 NPU 的时间重叠。

A 的 prompt 还有第 9 个 token。前两个中间块只更新缓存，不生成回答，因此显示 `partial prompt; no sample`。B 在这次回调中才加入，不会进入已经发出的第 1 轮。

## B 开始生成，C 还在等

收取第 1 轮时，C 加入。A 和 B 已占用 17 个槽位，只剩 3 个；C 需要 6 个，而且两个并发名额都已占用，所以它只能排队。等待中的 C 没有分配槽位或模型状态。

第 2 轮同时处理 A 的最后一个 prompt token 和 B 的两个 prompt token，两行都可以采样。第 3 轮让 A、B 各 decode 一次。收取第 3 轮时，B 的第二个输出还没提交，所以快照里 B 仍是 `running`。回调返回后，CPU 保存这个输出，才把 B 标为结束。

每轮计算由已发出进度规划，CPU 的结果记录按收取顺序更新。把两个进度混成一个，下一轮就可能重复计算，或读错上一轮 token。

## B 归还了槽位，为什么空闲数变成 0

随后会看到：

```text
collect 4 | issued through 5
  A decode 10..10: sample available
  A: running; cached CPU/issued=10/12; slots=14
  B: finished; cached CPU/issued=3/3; slots=0
  C: running; cached CPU/issued=0/3; slots=6
  observed released: B
  reused B -> C: slots [14, 15, 16]
  free slots: 0/20
```

B 原来的槽位 14、15、16 已交给 C，另外三个原先空闲的槽位也被 C 拿走。因此 A 占 14 个、C 占 6 个，空闲数是 0。只看空闲总数，会漏掉这次交接。

`observed released` 是两次观察之间的归属变化，`reused` 是根据历次槽位归属核对出的交接，都不是设备事件的精确时间。真正释放仍由执行循环检查请求的最后使用事件。这个固定长度例子可以预先知道 B 不会再参与后续轮；真实 EOS 则可能晚一轮才被 CPU 发现，已发出的额外计算和回收规则见[退出处理](STOPPING_PIPELINE.md)。

C 的 prompt 长度为 4，但 A 的 decode 先用掉一个输入预算，所以 C 第一块只处理 3 个 token。到下一轮再处理最后一个，之后才进入 decode。

## 最后核对什么

结束时应看到：

```text
drained | observed released: C
  A: length; 6 output tokens
  B: length; 2 output tokens
  C: length; 3 output tokens
whole-prompt reference outputs match: True
free slots: 20/20; clean: True
```

程序还用同一份只读模型、独立缓存，逐个请求运行原始同步 `Engine`，整段处理 prompt，比较最终输出。这里检查输出 token 一致、长度结束和缓存清理，不比较完整 logits；已有完整 logits 的 CPU/NPU 检查仍由[文本验证入口](PIPELINE_TEXT.md#实机检查)负责。

[保存的完整输出](results/2026-10-24/pipeline-demo-cpu.txt)来自实际 CPU 运行。演示代码约 130 行，直接读 [`pipeline_demo.py`](../minideepinfer/pipeline_demo.py)：`observe()` 只加入 B/C 并记录状态，调度、采样、停止与回收仍在原循环里。

## 换成 Ascend 910C 上的真实文本

按[安装指南](START_HERE.md#ascend-910c真实权重)准备配套环境和已有 Qwen3.5-2B 权重后，在 Bash 中运行：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "Explain prefill, decode, and why a request needs a KV cache." \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --prompt "用两句话解释为什么不同请求的缓存需要隔离。" \
  --max-new-tokens 64 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill \
  --pipeline --stagger-prompts --trace-rounds
```

真实文本使用聊天模板、自然 EOS，以及统一的最大输出数 64。分词长度、结束顺序和复用轮次取决于模型输出，不会复制上面固定整数例子的九轮轨迹。这条命令复用已有文本入口；本次新增演示只在 CPU 运行，没有为它重新加载 2B 权重或测量性能。
