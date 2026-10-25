# 用自己的文本跑双模型生成

[返回首页](../README.md) · [双模型实现](MODEL_DRAFT.md) · [逐轮讲解](MODEL_DRAFT_WALKTHROUGH.md)

本文命令默认 greedy。main 另有正温度的[投机采样](SAMPLING.md)，复用同一循环；Ascend 910C 数值与回退检查已通过，概率在 CPU FP64 上计算。

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

`--check-greedy` 会在双模型生成结束后，用同一套已加载的 target 权重和一份新缓存，再跑一次普通 greedy。正常结束时，两次使用相同的编码输入、EOS 和输出上限，比较输出 ID 与停止原因。它会增加计算，默认关闭。主动取消后的对照范围见下一节。

`--trace-rounds` 复用现有只读观察器，打印 draft 提出什么、target 接受什么，以及下一轮怎样对齐。它不改变候选，也不是设备时间线。关闭它后，模型工作量和输出应保持相同。

JSON 在生成及可选对照完成后写入，可以查看：

| 字段 | 含义 |
| --- | --- |
| `output_ids` / `text` | 实际提交的 ID 与解码文本 |
| `finish_reason` | `eos`、`length` 或 `cancelled` |
| `greedy_check.status` | 默认 `not_run`；显式对照通过后为 `passed` |
| `target_forward_calls` / `target_input_tokens` | target 的计算量，包含恢复重算 |
| `draft_work` | draft 的计算、快照与最终清理状态 |
| `free_slots` / `cache_capacity` | target 归还的槽位与本次请求容量 |
| `trace` | 仅开启逐轮观察时保存 |
| `source_sha256` | 这次生成对应的执行源码 |

只有显式 greedy 对照通过时才写 `passed: true`。这个文件记录成功生成与可选的输出检查；固定实验的冷缓存数值检查、异常报告和计时仍使用原来的 `model_draft_demo`。

## 边生成边接收，或在提交后取消

加 `--stream` 后，已提交的 token 组会立即交给 Transformers 的 [`TextStreamer`](https://huggingface.co/docs/transformers/v4.46.3/en/internal/generation_utils#transformers.TextStreamer)。终端按词、换行和中文字符输出；半个词可能暂存在解码缓冲里，不能把一次打印当成一个模型 token。结束时刷新余下文本，不再把整段答案打印第二遍。最终解码与流式解码均关闭空格清理，避免两种方式自动改出不同的空格。

下面的命令在提交数量达到 2 时请求取消，便于观察停止边界：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B \
  --speculative --device npu:0 --dtype float32 \
  --prompt "The capital of France is" --raw \
  --max-new-tokens 10 --cache-capacity 128 --draft-width 3 \
  --stream --cancel-after-tokens 2 --trace-rounds --check-greedy \
  --output outputs/stream-cancel.json
```

取消门槛不截断一个已经提交的组。比如先输出 `A`，下一轮验证后提交 `B C D E`，就完整交付这五个 token，再停止；不会把已经交给调用者的 `C D E` 撤回。如果同一组已经命中 EOS 或长度上限，仍记录正常的 `eos` 或 `length`，不改成取消。

Python 调用者把一个零参数函数传给 `run_speculative_text(..., should_cancel=...)`，例如 `threading.Event.is_set`，上层调用 `event.set()` 就能请求停止。检查发生在 prefill 之前、每个提交回调之后；已经开始的一轮先完成验证和回退，不在算子中途释放缓存。`on_commit` 只接收真正提交的 ID，返回值不控制取消。普通异常和 Ctrl+C 仍会传播并执行清理，不伪装成正常取消。

取消返回 `finish_reason: cancelled`，CLI 报告的 `status` 也为 `cancelled`。两边同步、清理完毕后才返回；等待设备失败仍抛异常，不能提前宣称槽位已归还。

若同时开启 `--check-greedy`，只对照已经交付的前缀，参考运行的输出上限设为这段前缀的长度，报告注明 `greedy_check.scope: committed_prefix`。因此参考以 `length` 结束、实际请求以 `cancelled` 结束是正常情况；检查通过不表示剩余生成或 EOS 已被验证。开始前就取消时没有输出，greedy 检查记为 `not_run`，不跑零长度请求。

## Ascend 910C 检查

最初的文本入口检查对应提交 `30e3b48`；下面保留当时的结果，原始报告按该版本核对。

上面的聊天问题编码后有 20 个输入 token。FP32 运行输出 `Paris`，ID 为 `[57590, 248046]`，第二个 token 是 tokenizer 的 EOS；独立 greedy 的 ID 和停止原因完全相同。两份请求缓存各预留、归还 35 个槽位，draft 的线性状态与快照均已清空。[原始 JSON](results/2026-10-25/qwen35-speculative-text.json)和[控制台输出](results/2026-10-25/qwen35-speculative-text.txt)保留完整记录。

draft 提出 3 个候选，target 全部拒绝并改正为 EOS。target 共 3 次 forward、25 个输入 token，包含一次恢复重算；draft 共 3 次、23 个输入 token。独立 greedy 另做 2 次 forward，完整程序共 8 次，两套已有权重各加载一次。这次没有计时或冷缓存 logits 对照，不能当成新的速度、长上下文或数值精度结论。raw、不同输出上限、纯 KV 与混合缓存、EOS、回调异常及中断由 CPU 测试覆盖。

当前是单请求、同步 greedy、FP32。它不接受批处理、分块 prefill、deferred 或 pipeline 参数。验证和回退沿用[投机参考](SPECULATIVE.md#这和-deepinfer-怎样对应)中对应 DeepInfer 的设计；这里新增的是文本接入，没有复现上游 MTP 或完整生产引擎。

本轮另跑了上面的流式取消命令：5 个原始输入 token，第一组提交 `[11751]`，下一组接受两个候选、由 target 改正第三个，提交 `[13, 198, 32]`。达到取消门槛后，共交付 4 个 token；target 在回退到 8 个有效输入后停止，没有再启动一轮。两边各归还 14 个槽位，draft 的线性状态与快照清空。独立 greedy 的前四个 ID 完全相同，检查范围标为 `committed_prefix`。

target 共 3 次 forward、12 个输入 token，包含恢复后重算 3 个输入；draft 共 3 次、8 个输入，独立前缀对照另做 4 次 forward。完整程序共 10 次，两套已有权重各加载一次，没有计时或冷缓存 logits 复查。[本轮 JSON](results/2026-10-25/qwen35-stream-cancel.json)和[原始输出](results/2026-10-25/qwen35-stream-cancel.txt)均保留。开始前取消、首 token 后取消、全部接受的组、取消与 EOS/长度同时发生、消费者异常和拆字解码由 CPU 测试覆盖。

## English usage

On main (`0.5.0.dev0`), `generate --speculative --draft-model PATH` accepts your own text with the target checkpoint's chat template, or raw text with `--raw`. The command above runs a single FP32 request using local Qwen3.5-0.8B and 2B checkpoints. v0.4.0 packages do not contain this frontend.

The frontend reuses the existing model draft and greedy verification loop. Both models own separate request caches. EOS and the output limit apply to committed target output. Tokenizer files and special IDs must match before either checkpoint loads.

`--check-greedy` optionally runs target-only greedy with a fresh cache after generation, comparing exact output IDs and the finish reason. It adds computation and is off by default. `--trace-rounds` records logical cache changes without adding model calls. A successful JSON report includes text, work counters, cleanup state, and source hashes; `passed: true` appears only after the explicit greedy check succeeds.

One Ascend 910C FP32 check of the chat prompt above returned `Paris` followed by EOS, exactly matching independent target greedy. Both caches returned 35 slots. Acceptance was 0/3; target and draft each made three forward calls, with two more for the independent baseline. No timing, new cold-logit comparison, or speedup is established by this check.

`--stream` now forwards committed groups to Transformers' `TextStreamer`, buffering incomplete words and characters and flushing once at the end. `--cancel-after-tokens N` requests cancellation when a whole committed group reaches the threshold; the delivered count can exceed N. The Python API accepts a zero-argument `should_cancel` predicate, polled before prefill and after each commit callback. EOS and length completion take precedence. Callback exceptions and Ctrl+C remain errors, rather than normal cancellation.

A cancelled request returns its committed prefix after both caches drain and close. Optional greedy checking covers only that prefix and records `scope: committed_prefix`; its reference finishes by length, not cancellation. A pre-cancelled empty request skips the greedy check. One additional Ascend 910C run of the raw command above stopped at four committed tokens with a threshold of two, accepted 2/3 candidates, and returned all 14 slots in each pool. The four-token prefix matched independent greedy. There were ten total forwards and no timing or new cold-logit checks.
