# top-k / top-p：先决定从哪里抽，再验证候选

[返回首页](../README.md) · [接受与补采样](SAMPLING.md) · [双模型文本入口](SPECULATIVE_TEXT.md)

温度改变分数的差距，top-k 和 top-p 决定哪些 token 还能被抽到。加进双模型推理后，过滤也会改变接受概率：小模型抽候选时用了哪份分布，验证时就必须用那份分布。

本课使用固定版本 [v0.5.0](RELEASE_v0.5.md)。保存的开发阶段报告仍保留原字段与源码哈希，按实际运行的固定提交核对。

## 先跑一个四 token 的例子

从仓库根目录执行，需要已有 PyTorch，不下载权重：

```bash
python -m minideepinfer.sampling_filters_demo --output outputs/sampling-filters-cpu.json
```

先看温度为 1 时的四个 token：

| 阶段 | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| 原始概率 | 45% | 30% | 15% | 10% |
| top-k=3，保留前三个并归一化 | 50% | 33.33% | 16.67% | 0 |
| 再做 top-p=0.7，归一化 | 60% | 40% | 0 | 0 |

top-p 看的是上一步归一化后的概率。A 只有 50%，还没到 70%；加上 B 就够了，因此保留 A、B。它保留的是一组 token，随后仍要从中随机抽取。

实现顺序固定为 **温度 → top-k → top-p → 归一化**。top-k 为 0、top-p 为 1 时关闭对应过滤，保留原有温度采样路径。k 大于词表大小时按整个词表处理；p 必须在 `(0, 1]`。

本项目采用明确的边界规则：分数相同时按 token ID 从小到大排序；top-k 最多保留 k 个；top-p 保留累计概率首次达到或超过 p 的那个 token。p 再小也保留第一项。这些约定方便复现，不保证与其他框架在并列分数、浮点边界处完全一致。

## 小模型猜到的 token，大模型可能根本不允许

假设过滤后，target 和 draft 的概率分别是：

```text
          A      B      C
target   2/3    1/3     0
draft     0     1/3    2/3
```

draft 抽中 C 时，target 的概率为 0，接受率就是 0。拒绝后，正差额 `max(p-q, 0)` 只剩 A，于是补采样得到 A。draft 抽中 B 则全部接受。

如果仍拿过滤前的 q 计算接受率，或者拒绝后直接按 p 重抽，这条规则就被破坏了。这里的 p、q 都是同一个前缀下，**各自模型经过完整过滤并归一化的概率**。使用同样的 top-k / top-p 参数，并不意味着两模型会保留相同的 token。

[`TemperatureSampler.probabilities()`](../minideepinfer/sampling.py) 负责温度和过滤；[`ModelDraft`](../minideepinfer/model_draft.py) 把实际 q 与候选一起保存；原有 [`verify_sampled()`](../minideepinfer/sampling.py) 继续处理接受、改正和 bonus。缓存回退、流式提交与取消仍走原循环。

## 用已有权重生成

先按[运行指南](START_HERE.md)准备环境和两套权重。以下是 Bash 命令：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B \
  --speculative --device npu:0 --dtype float32 \
  --temperature 0.7 --top-k 20 --top-p 0.9 --seed 2026 \
  --draft-width 3 --prompt "The capital of France is" --raw \
  --max-new-tokens 16 --cache-capacity 128 --stream \
  --output outputs/filtered-text.json
```

过滤参数目前属于单请求投机采样入口，必须同时指定正温度；不改变缺省的 greedy 模式，也不能与 `--check-greedy` 同用。报告记录实际过滤参数和执行顺序。target、draft 分别使用请求内的随机数对象，默认 seed 为 2026、2027。

模型计算使用 CPU 或 Ascend 910C FP32，过滤和概率计算仍在 CPU FP64，保留完整词表向量。这是可阅读的参考实现，没有设备端融合采样算子，也没有新增性能结论。

## 怎样核对

演示还会创建两套未训练的 Qwen3.5 小模型，以 top-k=7、top-p=0.8 运行普通采样、投机采样、取消和指定 EOS。它们用于检查执行与状态，不用于展示回答能力。

[CPU 记录](results/2026-10-25/sampling-filters-cpu.json)中，39 组条件概率与 11 组冷缓存对照通过。普通采样与独立整段前缀重算输出一致；投机路径用同一前缀重算 p、q，并复用原随机数核对提交决定。两个模型结束后各归还 8 个槽位。完整程序包含参考计算，共 79 次 forward、333 个输入 token。

测试另用独立的 Python 标量实现核对过滤顺序，覆盖相同分数、累计概率恰好到达门槛、极小 p、零概率拒绝和默认行为。模型检查复用原有验证循环；它验证固定输入的数值和状态，不是所有输入上的分布证明。算法依据沿用 [Leviathan 等人的拒绝采样](https://proceedings.mlr.press/v202/leviathan23a.html)，top-k / top-p 的通用接口说明可参考 [Transformers 文档](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TopPLogitsWarper)。

## Ascend 910C 记录

[实机报告](results/2026-10-25/qwen35-filters.json)使用真实 Qwen3.5-0.8B / 2B、FP32、HF32 关闭，温度 0.7、top-k=20、top-p=0.9。raw 输入 `The capital of France is` 共五个 token，输出上限 6、宽度 3，两边各有 10 个槽位。环境为 PyTorch 2.10.0、torch_npu 2.10.0.post6、CANN 9.1.0，CPU 线程 1。

24 组条件概率和 5 组冷缓存检查通过。概率比较使用 `atol=1e-6, rtol=2e-4`，最大绝对差约 `4.62e-6`；是否通过同时取决于绝对和相对误差。普通采样与独立整段重算输出一致，冷概率复用原随机数后，提交决定也一致。

本次投机请求接受 3/3 个候选，未走拒绝回退；过滤后的拒绝与恢复由 CPU 两种缓存测试覆盖，不把旧实机拒绝记录算作本次覆盖。取消请求交付完整一组后共有 5 个输出 token，没有启动下一轮；指定 EOS 使用第一枚实际采样 ID，只验证停止条件。所有缓存池各归还 10 个槽位。完整程序包含冷参考，共 44 次 forward、247 个输入 token，两套已有权重各加载一次。

命令入口为 `python -m minideepinfer.verify_sampling --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B --device npu:0 --top-k 20 --top-p 0.9 --output outputs/qwen35-filters.json`。这次检查没有计时、采集 trace 或额外执行真实文本流式 CLI。[原始日志](results/2026-10-25/qwen35-filters.txt)保留环境 warning。

## English usage

Run `python -m minideepinfer.sampling_filters_demo --output outputs/sampling-filters-cpu.json` with PyTorch installed; no checkpoints are required. The fixed four-token example shows top-k followed by top-p. Untrained Qwen3.5 models then exercise filtered proposal probabilities, cold-prefix references, rollback, cancellation and configured EOS.

For the existing `generate --speculative` text entry, add `--temperature 0.7 --top-k 20 --top-p 0.9 --seed 2026`. Both models apply temperature, top-k, then top-p to their own logits. Stable descending order breaks ties by ascending token ID; top-k keeps at most k entries, and top-p keeps the first token reaching the threshold. Zero k and p=1 disable filtering. Filtered probabilities remain CPU FP64, even with NPU FP32 model execution. There is no fused sampling kernel or speedup claim. See the Chinese sections for the runnable Bash command and recorded validation scope.

The saved NPU check passes 24 conditional-probability and five cold-state comparisons with real 0.8B/2B weights. It accepts 3/3 candidates, so filtered rejection is covered on CPU, not by this NPU run. All pools return 10 slots. The full check costs 44 forwards and 247 input tokens, including references; no timing or additional NPU streaming-CLI run is claimed.
