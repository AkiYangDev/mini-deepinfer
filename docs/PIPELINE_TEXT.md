# 用自己的文本运行跨轮执行

[返回首页](../README.md) · [安装指南](START_HERE.md) · [分块与进度](CHUNK_PIPELINE.md)

前面的跨轮实验使用固定 prompt，便于核对每一步的数值。现在 `generate` 也能选择同一个执行循环，输入自己的文本，观察新请求怎样进入批次。

这项功能位于 `main`，包版本为 `0.2.0.dev0`。已发布的 [v0.1.0](https://github.com/AkiYangDev/mini-deepinfer/releases/tag/v0.1.0) 保持原样。

## 先看一个不需要权重的例子

已有 PyTorch 时，在仓库目录运行：

```bash
python -m minideepinfer.verify_pipeline_text --device cpu --trace-rounds \
  --output outputs/pipeline-text-cpu.json
```

程序随机初始化一个小型 Qwen3.5，依次比较同步、跨轮和逐条加入 prompt 的结果。三种方式共用只读权重，各自使用独立缓存。这里的 token 是整数，输出只用于检查执行过程。

逐条到达的轨迹开头是：

```text
collect round 0 [before CPU commit; issued through round 1]
  req-0 prefill 0..3 partial; sample=-
  arrived: req-1
  waiting: req-1
  free slots: 16/24
```

它表示第 0 轮结果已经可以读，但 CPU 还没提交这轮进度；第 1 轮计算已经发出。此时才加入 `req-1`，因此它不会进入已发出的第 1 轮，最早在第 2 轮参与调度。

`partial` 是中间 prompt 块。这一块更新模型缓存，还不能产生回答，所以 `sample=-`。最后一个 prompt 块和 decode 才会给出采样 token。

## 换成真实文本

按[安装指南](START_HERE.md)准备环境和已有 Qwen3.5-2B 权重后，运行：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill \
  --pipeline --stagger-prompts --trace-rounds
```

没有 NPU 时可使用 `--device cpu` 和 Qwen3.5-0.8B 权重，CPU 仍需加载完整文本模型。默认使用 checkpoint 的聊天模板并关闭思考；`--raw` 改为原始文本补全，`--thinking` 传给支持这一选项的模板。

三个新选项各做一件事：

| 选项 | 行为 |
| --- | --- |
| `--pipeline` | 用设备上的前轮 token 提交下一轮计算，再处理上一轮结果；当前限定 FP32 |
| `--stagger-prompts` | 首条文本先进入，其余文本每收取一轮结果加入一条；省略时，所有文本一开始就排队 |
| `--trace-rounds` | 打印收取的轮次、已发出轮次、输入位置、采样结果及观察到的缓存归属 |

逐条加入按收取轮次触发，没有模拟毫秒级到达率，也没有后台输入线程。分词和打印仍在同一个 CPU 线程上执行。

去掉这三个选项，就回到原同步路径。`--deferred` 和 `--prepare-during-step` 使用单批在途的执行方式，与 `--pipeline` 分开选择；参数冲突会在加载权重前报错。

## 缓存回收要看归属

观察输出中可能出现：

```text
collect round 5 [before CPU commit; issued through round 6]
  req-1 decode 8..8; sample=6
  waiting: -
  observed released: req-0
  free slots: 5/24
```

`req-0` 的槽位已经归还，但空闲数仍只有 5，因为新的 `req-2` 已经拿到了其中一部分。仅看空闲总数，容易误以为没有回收。

`observed released` 表示：上一次观察时请求还持有槽位，这一次已经不再持有。它不是设备事件发生的精确时间。输出只读取 CPU 上的归属和已拷回的 token，不会为了打印再次读取设备采样张量。

请求遇 EOS 时，下一轮可能已经发出。那一轮继续完成，输出丢弃，缓存等最后一次使用完成后回收。最终会看到：

```text
pipeline drained
  observed released: req-2
  free slots: 24/24
```

这表示循环已排空。各请求的完整文本随后打印，仍采用原来的停止原因、输出数和空闲槽位格式。

## 代码从哪里读

[`generate.py`](../minideepinfer/generate.py) 负责参数和执行方式选择。同步与跨轮都通过 `run_generation()` 接收已加载的 Runner 和 tokenizer，因此数值检查可以共用一次权重加载。

[`pipeline_text.py`](../minideepinfer/pipeline_text.py) 负责文本编码、回调到达和观察记录。它调用原有 [`StoppingDecodePipeline`](../minideepinfer/stopping_pipeline.py)，没有另建调度循环。trace 的观察点在结果提交前；真正的停止判断和回收仍由原循环执行。

[`verify_pipeline_text.py`](../minideepinfer/verify_pipeline_text.py) 比较完整 logits 和最终输出。同步参考遇 EOS 后已经停止，跨轮路径多算的一步另用冷缓存整段重算作对照，避免误把“没有参考行”当成模型错误。打开 trace 会增加主机打印开销，不能用它判断吞吐或提速。

## 实机检查

2026-10-24，在 Ascend 910C 的一个逻辑设备上复用 Qwen3.5-2B 权重，以 FP32、关闭 HF32 检查新前端。软件沿用 Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6 和 CANN 9.1.0。

| 路径 | 检查结果 |
| --- | --- |
| 三条原始文本，同步参考 | 整段 prompt 计算，各输出 4 个 token |
| 全部文本先排队的跨轮执行 | 11 轮、16 组完整 logits 对照，最大差约 `2.67e-5`，24/24 槽位归还 |
| 每收取一轮加入一条文本 | 11 轮、16 组对照，最大差约 `2.29e-5`，输出与同步一致，24/24 槽位归还 |
| 真实聊天模板与 EOS | 输出 `1 plus 1 equals 2.`，9 个 token 后遇 EOS；11 组对照含多算的一步，最大差约 `4.94e-5`，256/256 槽位归还 |

共 43 组完整 logits 对照，在原 `atol=1e-4, rtol=2e-4` 内通过。成功程序加载一次权重，共 51 次 forward：原始文本同步 8、两种跨轮方式各 11、聊天同步 9、聊天跨轮 11、EOS 多算一步的冷缓存参考 1。

首次实机程序在计算完成后，因对照脚本缺少 EOS 多算一步的参考而报 `KeyError`，未生成通过报告。补上独立重算和受控 EOS 的 CPU 回归后才做第二次检查。因此本轮实际加载了两次权重；另有显存门槛拦下的请求，没有加载模型。这次失败属于验证脚本，不计为模型数值失败，也不作性能结论。

[原始报告](results/2026-10-24/qwen35-2b-pipeline-text-npu-fp32.json)保留每次观察、完整输出和十八份相关源码 SHA。它调用与 CLI 相同的 `run_generation()`，用预加载权重避免为每个方式重复加载模型；参数分派另有 CPU CLI 检查。可在干净检出中核对：

```bash
python scripts/check_evidence.py --index docs/results/v0.2/evidence.json
```

本地 149 项测试在 Transformers 5.12.1 环境全部通过；4.46.3 环境通过 137 项、跳过 12 项独立参考；无模型依赖时通过 59 项、跳过 90 项。已发布 v0.1 的原报告和源码索引继续保留。
