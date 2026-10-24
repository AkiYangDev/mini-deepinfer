# 不提前知道答案，候选从哪里来？

[返回首页](../README.md) · [验证与状态回退](SPECULATIVE.md) · [后续路线](ROADMAP.md)

前一个实验先算好 greedy 答案，再故意改错某个候选，用来检查接受、拒绝和回退。这样容易重现分支，却不能回答一个实际问题：没有答案时，draft 能提出什么？

这里加一个只读已提交上下文的 `LookupDraft`。它不加载第二个模型，而是在前文找相同片段，把那次出现后面的 token 作为候选。验证仍由原来的 target 完成；Qwen3.5 猜错后的状态恢复也沿用上一章。

## 跑起来看

已有 CPU PyTorch 时，无需下载权重：

```bash
python -m minideepinfer.spec_demo --draft lookup --hybrid \
  --output outputs/lookup-draft.json
```

去掉 `--hybrid` 会运行已有的纯注意力小模型，用来观察“始终找不到续文”的情况。`--help` 不需要模型依赖。

与受控候选不同，这个模式先完成 lookup 生成，再用独立缓存运行普通 greedy。报告中的 `baseline` 只用于事后对照，不传给 draft；候选生成对象只收到 `prefix` 和本轮允许的最大候选数。

## 怎样找，找不到怎么办

例如，已提交 token 是：

```text
1 2 7 8 1 2
        └─┘ 当前末尾
└─┘         前面也出现过 1 2
    7 8 1   最多取三个已经存在的 token 当候选
```

默认先找最长四个 token 的后缀，找不到就逐渐缩短到一个。同样长度出现过多次时，选最近一次；那次出现后只剩一个 token，就只提一个，不补齐。历史续文可能与当前后缀重叠，但其中每个 token 都必须已经提交。

找不到就返回空元组。验证器这时只消费上轮待计算的末尾 token，提交 target 的下一个输出，继续正常生成。为此，draft 回调现在返回 **不超过**本轮上限的候选；超过上限、非法 token ID 仍报错并清理。原来的受控候选接口继续可用。

这段查找没有读取 logits、未来 token、greedy 答案或另一个请求的缓存。它只是从已知历史里提出一段可能重复的续文。即使找到了相同片段，后面的生成也未必相同，所以每个候选仍要验证。

`LookupDraft` 是本项目新增的小参考实现，不是从 DeepInfer 移植的 draft 算法。DeepInfer 的来源对应仍是上一章中的验证、输出提交与长度修正；这里用独立候选把这条流程串起来，没有实现它的 MTP。

## 成本怎样记录

报告保存每轮的匹配长度、历史位置、允许的宽度和实际候选；总计查找调用、命中调用、提出的 token、检查的历史窗口和逐 token 比较次数。draft 的模型 forward 是零，查找本身仍消耗 CPU。

比较次数不是完整 CPU 成本，不含输入校验、切片、张量准备、同步和状态复制，也不能换算成毫秒。target 的验证、拒绝后的重算、线性状态快照，以及额外的冷缓存检查，继续分别统计。

这个直接扫描的实现适合短上下文教学。上下文变长时，每轮还要遍历历史；没有索引或长上下文性能保证。当前同步参考即使本轮没有候选，也沿用同一条快照路径，这些复制成本照常计入报告。

## CPU 实际发生了什么

两个例子都使用 FP32 随机权重、seed 2026，prompt 是 `[2, 3, 4]`，输出固定十个 token，最大候选宽度三：

| 模型 | target forward | target 输入 token | 接受 / 提出候选 | 查找调用 / 命中 | token 比较次数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 随机 Qwen3.5 | 6 | 12 | 4 / 4 | 5 / 4 | 17 |
| 随机 Qwen2 | 10 | 12 | 0 / 0 | 9 / 0 | 160 |

两者均与事后独立 greedy 输出相同，归还全部 12 个槽位。Qwen3.5 的随机模型恰好连续输出十个 `4`，非常容易在历史里找到重复；它不是自然语言负载，也不能代表真实接受率。Qwen2 则一直不提候选，仍付出了查找成本。这里的 `0 / 0` 表示没有候选，不定义为零接受率或百分之百接受率。

Qwen3.5 做了五次状态快照，每份 5,376 字节，没有拒绝重算；五次提交后冷缓存检查均通过 `atol=1e-5, rtol=1e-4`。其完整演示还包含十次 baseline 和五次冷缓存检查，共 21 次模型 forward。纯注意力例子没有额外冷缓存检查，共 20 次模型 forward。没有计时或加速结论。

测试另外覆盖两种模型、两种 prompt、三种宽度、三种长度以及 EOS 开关，共 72 条端到端轨迹；每次提交检查有效缓存。还有 200 组固定随机历史，确认候选只能是已提交历史中的连续切片。过长候选、无匹配、零宽度和短续文分别检查，不只比较最终答案。

## Ascend 910C 的真实权重结果

使用已下载的 Qwen3.5-2B 权重，可运行同样的入口：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m minideepinfer.spec_demo \
  --draft lookup --model /path/to/Qwen3.5-2B --device npu:0 \
  --output outputs/qwen35-2b-lookup-draft.json
```

本次使用单逻辑 Ascend 910C、FP32、HF32 关闭、线程数一。权重固定为 revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，原始输入是 `The capital of France is`，五个 token，不加聊天模板，固定输出十个 token，不按 EOS 提前结束。一次现有权重加载，没有下载、trace 或计时。

lookup 和事后独立 greedy 都输出 `[11751, 13, 198, 32, 13, 2912, 198, 33, 13, 3439]`。这次提出五个候选，没有一个接受：

| 工作量 | 普通 greedy | lookup 路径 |
| --- | ---: | ---: |
| target forward | 10 | 12，含两次恢复后重算 |
| target 输入 token | 14 | 21，含两个重算输入 |
| 接受 / 提出候选 | 不适用 | 0 / 5 |
| 状态快照次数 | 不使用这条路径 | 9 |

九次查找中，两次找到历史片段，共检查 211 个窗口、比较 215 对 token。比如末尾 token `13` 曾在前文出现，draft 据此提出 `[198, 32, 13]`，target 的下一输出却是 `2912`，第一个候选就被拒绝。另一次提出 `[32, 13]`，同样首个拒绝。状态恢复后各重算一个有效输入，继续生成。

每份线性状态快照 20,201,472 字节，累计复制 181,813,248 字节。九次提交后的冷缓存对照均通过预设 `atol=1e-4, rtol=2e-4`，检查有效 K/V、卷积历史、递推矩阵和前缀末尾完整 logits；请求最终归还全部 14 个槽位。完整演示另含十次 baseline 和九次冷缓存检查，共 31 次模型 forward；冷参考处理 90 个输入 token。

这个短输入里，lookup 没减少 target 工作量，还增加了查找、复制和重算。没有计时，不能换算成慢了多少；也没有更换输入反复寻找漂亮的接受数字。这条路径作为教学入口保留，不接入主文本入口或跨轮 pipeline。下一步若研究模型 draft，应先确定权重、词表与目标模型的关系，再比较候选生成成本；当前结果不构成继续优化这个查找器的性能依据。

两份 CPU 报告和一份实机报告分别保存：[混合小模型](results/2026-10-24/lookup-draft-hybrid-cpu.json)、[纯注意力小模型](results/2026-10-24/lookup-draft-qwen2-cpu.json)、[真实 Qwen3.5-2B](results/2026-10-24/qwen35-2b-lookup-draft.json)。它们绑定八份或十份执行源码，核对入口为：

```bash
python scripts/check_evidence.py --index docs/results/lookup-draft/evidence.json
```

本次是本项目模型与独立缓存的对照，没有新增 Transformers 数值结论。前一章的受控候选和混合状态报告继续按各自固定 commit 校验，不代表当前 lookup 路径的测量。

## English summary

`python -m minideepinfer.spec_demo --draft lookup --hybrid` uses committed token history to propose candidates without another model or a precomputed answer. It searches longest suffixes first, chooses the latest earlier match, and copies only existing continuation tokens. A miss produces no candidates; target generation continues normally. The independent greedy baseline runs afterwards.

The draft interface now accepts up to the requested width. Reports separate lookup comparisons, proposed and accepted tokens, target work, state snapshots, replay, and cold-cache checks. CPU random Qwen3.5 and Qwen2 examples both match greedy output, but one emits repeated token IDs while the other never proposes a candidate. These are mechanism checks, not acceptance-rate benchmarks or evidence of a speedup. This small lookup implementation is separate from DeepInfer MTP; verification and state handling reuse the previous lesson.

One real Qwen3.5-2B FP32 run on Ascend 910C matched the subsequent greedy baseline, with all nine cold-cache state checks passing at `atol=1e-4, rtol=2e-4`. None of its five proposed tokens was accepted. Target work increased from 10 forwards / 14 input tokens to 12 forwards / 21 input tokens, including two replay calls. Nine linear-state snapshots copied 181,813,248 bytes in total. No latency or throughput was measured. The full demonstration used 31 model calls including the independent baseline and cold checks. This result supports a working independent-draft teaching path, not a default acceleration feature.
