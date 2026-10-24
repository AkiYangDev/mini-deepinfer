# 猜错的 token，缓存怎么办？

[返回首页](../README.md) · [学习路线](LEARNING_PATH.md) · [DeepInfer 来源](SOURCE_MAP.md)

前面的执行循环里，已发出的计算可以领先于 CPU 已提交的结果。投机解码把这个区别变得更直接：模型已经算过几个猜测 token，但它们未必都能成为输出。猜错以后，下一轮必须沿着被接受的前缀继续，不能读到错误分支的缓存。

这个实验在 main 的 v0.3.0.dev0 中加入。用一个无需权重的 CPU 小模型，把候选、验证、提交与回退走完。主生成示例仍是 Ascend 910C 上的 Qwen3.5-2B；本章先使用已有的纯注意力 Qwen2 实现，因为它的 KV 前缀容易检查。Qwen3.5 的线性状态不能套用这里的回退方式。

## 先运行

在仓库目录，已有兼容 PyTorch 时：

```bash
python -m minideepinfer.spec_demo --output outputs/spec-demo.json
```

没有模型依赖时，先按[运行指南](START_HERE.md)安装；这个入口只需要 CPU PyTorch，不下载 checkpoint。`--help` 不导入模型依赖。它是独立的教学入口，没有接入 `generate --pipeline`。

例子先用普通 greedy 算出十个输出，再拿这份答案构造三种候选：首个候选故意改错、第二个候选改错、全部正确。这让每个分支都能稳定出现。真实的 draft 模型不会提前知道答案，因此这一步是测试工具，不用于证明加速。

实际普通 greedy 输出：

```text
[41, 66, 63, 27, 60, 39, 54, 72, 57, 15]
```

这些是随机小模型的 token ID，不是一段自然语言。完整[控制台输出](results/2026-10-24/spec-demo-cpu.txt)和[结构化记录](results/2026-10-24/spec-demo-cpu.json)都保留在仓库。

## 一行 logits 预测哪一个 token

prompt 是 `[2, 3, 4]`。prefill 后，target 提交第一个输出 `41`，但此时缓存只写了 prompt：`41` 还没有作为输入计算过。它就是下一轮的待计算 token。

假设 draft 提出后面三个 token `[66, 63, 27]`，验证时一次把下面四个 token 交给 target：

| target 本轮输入 | 这一行 logits 预测什么 |
| --- | --- |
| 待计算的 `41` | 第一个候选是否应为 `66` |
| 候选 `66` | 第二个候选是否应为 `63` |
| 候选 `63` | 第三个候选是否应为 `27` |
| 候选 `27` | 全部接受后，额外输出什么 |

因果 mask 保证每一行只看它之前的前缀。四行可以一次计算，但只有连续匹配 target argmax 的候选才能接受。遇到第一次不匹配后，后面的行是沿错误前缀计算的，不能再接着提交。

[`verify_greedy()`](../minideepinfer/speculative.py) 只处理这个决定，不依赖 PyTorch。全部匹配时，最后一行提供 bonus token；不匹配时，第一处拒绝所对应的 target argmax 提供改正 token。本章只使用 greedy，不实现随机投机采样的接受概率与残差重采样。

## 看一次拒绝与回退

故意把第一个候选 `66` 改成 `67`，实际输出为：

```text
输入 [41, 67, 63, 27] → target [66, 47, 27, 60]
接受 0 个候选；提交 [66]；最后来源=correction
KV 暂存 7 → 保留 4；失效槽位 [4, 5, 6]；待计算 token=66
```

验证前缓存有三个 prompt token。这次又写入 `41、67、63、27` 的 KV，暂时达到七个位置。target 第一行要求 `66`，所以 `67` 被拒绝。输出只增加改正后的 `66`。

应保留的缓存是 `[2, 3, 4, 41]`，共四个位置。新输出 `66` 仍待下一轮计算；错误的 `67、63、27` 所在槽位失效。下一轮先把 `66` 写到第 4 个位置，再沿有效前缀继续。

[`CachedTarget.rollback()`](../minideepinfer/speculative.py) 删除的是有效输入前缀的尾部记录，没有把张量清零。每次 forward 构造的 `context_slots` 只包含保留前缀和本轮新写入位置。旧字节可以暂留，但不能再出现在可读上下文里；失效位置重新使用前会被覆盖。

这里的回退也不等于把整个预留空间还给其他请求。这个单请求实验预先保留 `prompt长度 + 最大输出长度 - 1` 个槽位，运行结束才统一归还。物理槽位归属与其中哪一段 KV 有效，是两个不同的问题。

## 全部接受时还有一个输出

候选全部正确时，实际第一轮是：

```text
输入 [41, 66, 63, 27] → target [66, 63, 27, 60]
接受 3 个候选；提交 [66, 63, 27, 60]；最后来源=bonus
KV 暂存 7 → 保留 7；失效槽位 []；待计算 token=60
```

前三行确认三个候选，最后一行给出 `60`。提交了四个输出，却只新计算了四个输入：旧的待计算 token `41`，加三个候选。`60` 是新输出，仍没有自己的 KV。

因此每一轮提交后都检查同一个约定：

```text
有效 KV 对应 prompt + 已提交输出，但不含最后一个待计算 token。
```

接近长度上限时，候选数量裁剪为 `min(width, 剩余输出数 - 1)`，留出 target 最后一行的位置。只剩一个输出时不再提候选，直接让 target 消费待计算 token。报告将这种直接输出和 bonus 分开统计；初始 prefill 输出也属于直接 target 输出。

EOS 若出现在接受的候选中，到这里就停止，不再提交后面的改正或 bonus。若 target 的最终输出本身是 EOS，也在这里结束。结束时最后一个输出的 KV 同样不需要保留。

## 这和 DeepInfer 怎样对应

本轮阅读固定版本 `8b16149741a4935e1dec8cea27853f273b374c5b`，对应关系如下。源码没有复制进教学模块。

| 上游入口 | 保留的学习问题 | 本章的简化 |
| --- | --- | --- |
| [`Sampler._verification()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/sampler.py#L411) | 验证候选、得到连续接受前缀和最终输出行 | `verify_greedy()` 比较 argmax；不复现概率验证、残差采样、结构化约束或 golden 覆盖 |
| [`LLMEngine._process_model_outputs()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py#L687) | 接受的候选与最终输出分开交给结果处理，推进请求状态 | `generate()` 在一个 CPU 请求中提交二者，分别记录接受、改正、bonus 和直接输出 |
| [`Worker._correct_seqlens_qk()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py#L191) | 后续计算长度需要服从已接受进度 | `CachedTarget` 用一个显式有效前缀排除拒绝分支，不复现 ZOS 元数据、变长设备修正或上游资源管理 |

原版还用 MTP 等组件产生下一轮预测。这里的回调只是受控输入，不是 MTP、DSpark 或另一个已训练的模型。有效前缀回退是本项目对纯注意力缓存的教学实现，不声称与原版采用相同的缓存操作。

## 观察工作量，而不是计算加速倍数

十个输出的实际记录如下，target 次数包含各实验自己的 prefill：

| 方式 | target forward | 实际处理的输入 token | 接受 / 提出候选 | 改正输出 | bonus | 直接 target 输出 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 greedy | 10 | 12 | 不适用 | 不适用 | 不适用 | 10 |
| 首个候选拒绝 | 10 | 33 | 0 / 21 | 8 | 0 | 2 |
| 第二个候选拒绝 | 6 | 19 | 4 / 11 | 4 | 0 | 2 |
| 全部接受 | 4 | 12 | 6 / 6 | 0 | 2 | 2 |

每种实验的“接受候选 + 改正 + bonus + 直接输出”都等于十个最终输出。提出数量包含整次实际给出的候选，不把未接受的尾部从分母删掉。

构造候选之前，额外运行了一次十 forward 的 greedy 答案准备；这份成本单独记录在 `oracle`，没有藏进 target 次数。这不是能用于部署的加速方法。即使 target forward 变少，也没有测量吞吐或延迟，更不能直接把 `10/4` 写成速度提升。

## 怎么检查回退真的正确

测试不只看最后答案。两种随机 seed、四种输出长度、四种 draft 宽度和三种候选模式，组成 96 条端到端轨迹，与使用独立缓存的原 `Engine` greedy 对照，检查输出、停止长度、有效缓存前缀与最终槽位回收。

另一个数值检查把分块验证的完整 logits 与冷缓存整段重算比较，包括错误候选分支；使用非连续物理槽位。回退后将失效槽位填入 NaN，再逐步继续计算。若下一轮读到了它们，结果会出错；本次仍在 `atol=1e-5, rtol=1e-4` 下与重算一致。

边界覆盖首次、中途和末尾拒绝、全部接受、EOS 出现在 prefill/候选/最终输出、输出长度裁剪、零宽度、非法候选、容量不足，以及观察回调失败时释放槽位。混合注意力模型会明确拒绝接入这个缓存实现。

公开示例使用 CPU、FP32、固定 seed 的随机小模型，没有真实权重、NPU 运行或性能测量。报告绑定七份相关执行源码，[索引](results/speculative/evidence.json)固定报告本身的 SHA。核对保存的数据，无需执行模型：

```bash
python scripts/check_evidence.py --index docs/results/speculative/evidence.json
```

下一阶段需要分别处理真实 draft 的成本、Qwen3.5 线性状态的恢复，以及 NPU 资源生命周期。完成这些前，本实验保持独立，已有文本生成入口继续按原来的路径运行。

## English summary

`python -m minideepinfer.spec_demo` runs a tiny CPU FP32 full-attention model without downloading weights. It demonstrates controlled greedy draft verification, prefix acceptance, correction or bonus output, and logical KV rollback. The draft reads a precomputed greedy answer to force specific cases; this is a test fixture, not a learned draft model or a speedup technique.

Each verification consumes the pending last output plus the candidates. After commit, the valid cache covers the prompt and committed outputs except the new pending last token. Rejected tensor bytes remain in storage but are excluded from future context and overwritten before reuse. Tests compare outputs with an independent greedy engine and full logits with cold recomputation, including noncontiguous slots and NaN-poisoned rejected storage.

The fixed DeepInfer references above distinguish verification, accepted versus final outputs, and subsequent length correction. This lesson simplifies them to one greedy CPU request. It does not implement probability-preserving speculative sampling, MTP/DSpark, hybrid-state rollback or an NPU pipeline. The main text model remains Qwen3.5-2B. This is a v0.3 development entry point on main; v0.2.0 release artifacts do not contain it.
