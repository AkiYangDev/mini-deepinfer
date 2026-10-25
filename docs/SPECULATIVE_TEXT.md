# 用自己的文本跑双模型生成

[返回首页](../README.md) · [双模型实现](MODEL_DRAFT.md) · [逐轮讲解](MODEL_DRAFT_WALKTHROUGH.md)

固定输入适合看缓存怎样恢复，但也应该能换成自己的问题。`generate --speculative` 将文本编码后，直接交给已有的 `ModelDraft` 和 `speculative.generate()`：小模型提候选，target 验证，最后只解码实际提交的 token。

这项功能在 main 的 `0.5.0.dev0` 中，v0.4.0 安装包不包含它。已有安装环境可以切到 main 后更新项目本身，无需重装 PyTorch、torch_npu 或 CANN：

```bash
git fetch origin
git switch main
git pull --ff-only
python -m pip install -e . --no-deps
```

## 从一个问题开始

先按[运行指南](START_HERE.md)准备匹配的 Ascend 910C 环境。下面的两个路径是已经下载好的本地权重目录；运行时不会下载模型。

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B \
  --speculative --device npu:0 --dtype float32 \
  --prompt "Reply with exactly one word: Paris." \
  --max-new-tokens 16 --cache-capacity 128 --draft-width 3 \
  --check-greedy --trace-rounds --output outputs/speculative-text.json
```

默认使用 target 权重附带的聊天模板，关闭 thinking；需要时加 `--thinking`。如果想续写原始文本，加 `--raw`，例如把 prompt 换成 `The capital of France is`。两边使用同一组 token ID，因此加载权重之前先核对 tokenizer 文件和特殊 ID；不一致就停止，不能只比较词表大小。

输出先打印生成文本，再显示停止原因、接受的候选数和两边归还的槽位。命中 tokenizer 的 EOS 就结束；没有命中时在长度上限停止。末尾预测出的 token 不再输入模型，所以实际预留的容量为 `prompt token 数 + 最大输出数 - 1`。`--cache-capacity` 是允许使用的上限，两边各自分配这份请求需要的容量，不共享张量或状态。

`--draft-width` 默认为 1，可选 1 到 8。它只是每轮候选上限：接近输出长度上限时会缩短，最后只剩一个输出时 draft 不再计算。更大的宽度会增加候选计算和拒绝后的恢复，不能从接受数推断速度。

## 哪些输出来自验证

`--check-greedy` 会在双模型生成结束后，用同一套已加载的 target 权重和一份新缓存，再跑一次普通 greedy。两次使用完全相同的编码输入、EOS 和输出上限，比较输出 ID 与停止原因。它会增加计算，默认关闭。

`--trace-rounds` 复用现有只读观察器，打印 draft 提出什么、target 接受什么，以及下一轮怎样对齐。它不改变候选，也不是设备时间线。关闭它后，模型工作量和输出应保持相同。

JSON 在生成及可选对照完成后写入，可以查看：

| 字段 | 含义 |
| --- | --- |
| `output_ids` / `text` | 实际提交的 ID 与解码文本 |
| `finish_reason` | `eos` 或 `length` |
| `greedy_check.status` | 默认 `not_run`；显式对照通过后为 `passed` |
| `target_forward_calls` / `target_input_tokens` | target 的计算量，包含恢复重算 |
| `draft_work` | draft 的计算、快照与最终清理状态 |
| `free_slots` / `cache_capacity` | target 归还的槽位与本次请求容量 |
| `trace` | 仅开启逐轮观察时保存 |
| `source_sha256` | 这次生成对应的执行源码 |

只有显式 greedy 对照通过时才写 `passed: true`。这个文件记录成功生成与可选的输出检查；固定实验的冷缓存数值检查、异常报告和计时仍使用原来的 `model_draft_demo`。

## 本次 Ascend 910C 检查

上面的聊天问题编码后有 20 个输入 token。FP32 运行输出 `Paris`，ID 为 `[57590, 248046]`，第二个 token 是 tokenizer 的 EOS；独立 greedy 的 ID 和停止原因完全相同。两份请求缓存各预留、归还 35 个槽位，draft 的线性状态与快照均已清空。[原始 JSON](results/2026-10-25/qwen35-speculative-text.json)和[控制台输出](results/2026-10-25/qwen35-speculative-text.txt)保留完整记录。

draft 提出 3 个候选，target 全部拒绝并改正为 EOS。target 共 3 次 forward、25 个输入 token，包含一次恢复重算；draft 共 3 次、23 个输入 token。独立 greedy 另做 2 次 forward，完整程序共 8 次，两套已有权重各加载一次。这次没有计时或冷缓存 logits 对照，不能当成新的速度、长上下文或数值精度结论。raw、不同输出上限、纯 KV 与混合缓存、EOS、回调异常及中断由 CPU 测试覆盖。

当前是单请求、同步 greedy、FP32。它不接受批处理、分块 prefill、deferred 或 pipeline 参数。验证和回退沿用[投机参考](SPECULATIVE.md#这和-deepinfer-怎样对应)中对应 DeepInfer 的设计；这里新增的是文本接入，没有复现上游 MTP 或完整生产引擎。

## English usage

On main (`0.5.0.dev0`), `generate --speculative --draft-model PATH` accepts your own text with the target checkpoint's chat template, or raw text with `--raw`. The command above runs a single FP32 request using local Qwen3.5-0.8B and 2B checkpoints. v0.4.0 packages do not contain this frontend.

The frontend reuses the existing model draft and greedy verification loop. Both models own separate request caches. EOS and the output limit apply to committed target output. Tokenizer files and special IDs must match before either checkpoint loads.

`--check-greedy` optionally runs target-only greedy with a fresh cache after generation, comparing exact output IDs and the finish reason. It adds computation and is off by default. `--trace-rounds` records logical cache changes without adding model calls. A successful JSON report includes text, work counters, cleanup state, and source hashes; `passed: true` appears only after the explicit greedy check succeeds.

One Ascend 910C FP32 check of the chat prompt above returned `Paris` followed by EOS, exactly matching independent target greedy. Both caches returned 35 slots. Acceptance was 0/3; target and draft each made three forward calls, with two more for the independent baseline. No timing, new cold-logit comparison, or speedup is established by this check.
