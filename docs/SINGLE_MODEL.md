# 一份权重，也能采样、流式输出和取消

[返回首页](../README.md) · [温度采样](SAMPLING.md) · [top-k / top-p](SAMPLING_FILTERS.md) · [双模型生成](SPECULATIVE_TEXT.md)

学投机解码前，可以先看普通生成：模型读入 prompt，抽出第一个 token，再把它送回模型，得到下一个。小模型还没出场，缓存、采样和停止条件已经能完整走一遍。

`generate --single` 只加载一份模型，支持温度、top-k / top-p、流式输出、取消和 JSON 报告。它在 **main 的 0.6.0.dev0** 中；固定学习版本 v0.5.0 保留原样，不包含这个新入口。

## 先用一份 Qwen3.5 权重

已有环境先更新项目代码，不重装 PyTorch、torch_npu 或 CANN：

```bash
git switch main
git pull --ff-only
python -m pip install -e . --no-deps
```

按[运行指南](START_HERE.md)准备环境与本地权重后，从仓库目录执行：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --single --device npu:0 --dtype float32 \
  --prompt "Explain what a KV cache does in two sentences." \
  --temperature 0.7 --top-k 20 --top-p 0.9 --seed 2026 \
  --max-new-tokens 32 --cache-capacity 256 --stream \
  --output outputs/single-model.json
```

也可以使用 Qwen3.5-0.8B。权重与内存足够时，将设备改为 `cpu` 就能跑 CPU；命令不会下载模型。删掉温度和过滤参数，使用 greedy。`--seed`、`--top-k`、`--top-p` 都需要显式的正温度。

增加 `--cancel-after-tokens 8`，交付第八个 token 后停止。每次提交一个 token，因此单模型不会像投机模式那样跨过门槛后再交付一整组。若此时恰好命中 EOS 或长度上限，仍记录正常完成。取消只在提交边界检查，不打断正在执行的算子；Ctrl+C 和异常仍会传播并清理缓存。

`--stream` 使用 tokenizer 的文本缓冲。提交一个 token，不保证屏幕马上显示一个字：一个汉字可能占多个 token，完整文本也可能等到空格或结束时才刷新。引擎提交的 ID 可以从报告的 `output_ids`、`rounds` 中查看。

## 为什么没有再写一套生成循环

[`run_target_text()`](../minideepinfer/target_text.py)负责文本编码和请求参数，然后调用已有的 [`speculative.generate()`](../minideepinfer/speculative.py)，把候选宽度设为零。没有候选时，每轮只计算上一个已提交 token，再直接从 target 分布抽取下一个；不创建 `ModelDraft` 或 draft 缓存。

比如 prompt 有 5 个 token，要输出 6 个：

| 计算 | 本轮输入 | 交付输出后，缓存里有什么 |
| --- | --- | --- |
| prefill | 5 个 prompt token | prompt；第一个输出尚未送回模型 |
| 第一次 decode | 第一个输出 | prompt + 第一个输出 |
| 最后一次 decode | 第五个输出 | prompt + 前五个输出；第六个已经交付 |

完整生成需要 6 次 forward，缓存最多容纳 `5 + 6 - 1 = 10` 个输入。最后一个输出不需要为了“补齐缓存”再计算一次。取消也一样：交付后不再启动下一轮，清理状态并归还槽位。

`cache-capacity` 是请求上限，实际分配按 `prompt 长度 + 输出上限 - 1` 计算。报告同时保留 `cache_budget` 和实际 `cache_capacity`。温度与过滤继续使用现有 CPU FP64 采样器，每个请求有自己的随机数状态；这里是便于检查的 FP32 同步路径。

双模型模式仍通过同一个循环提交输出、处理停止和归还缓存。它增加的是候选、验证和回退，不改变“只交付已经决定的 token”这条规则。两条路径的区别因此能直接从代码里读出来。

## 没有候选，还需要保存回退快照吗

Qwen3.5 的线性注意力会修改卷积和递归状态。验证多个候选时，后面的计算可能被拒绝，提前留一份状态副本，才能恢复到正确前缀。这份副本就是回退快照。

单模型每轮只送回上一个已经交付的 token。这个输入一定保留，没有候选需要撤销，因此不用复制状态。即使新抽出的 token 是 EOS，停止时也只是释放缓存，不需要先回退。现在循环按**实际候选是否为空**决定要不要保存快照：候选宽度设为 3，但本轮返回空列表，同样可以省掉复制；一旦有候选，仍保留原来的恢复过程。

CPU 测试用相同权重、输入和随机种子，对照“每轮都留副本”的旧方式。除了快照次数和字节数，输出、缓存进度、模型计算量与逐轮记录一致；有候选和空候选交替出现时，也逐轮核对独立冷缓存。

Ascend 910C 的[新报告](results/2026-10-25/qwen35-target-no-snapshot.json)沿用下文实测的输入与参数。六输出的正常生成从 5 次快照、复制 101,007,360 字节变为零；取消和配置 EOS 的路径也不再复制快照。与[旧报告](results/2026-10-25/qwen35-target-text.json)相比，除源码哈希和三项快照统计外，全部字段相同。这次只检查复制量与正确性，没有计时，不能据此换算提速比例。

## 怎样核对

CPU 测试使用未训练的小型 Qwen2 和 Qwen3.5，逐 token 与独立完整前缀重算对照，覆盖 greedy、过滤采样、聊天模板、取消、EOS、异常清理和 CLI 单次加载。无需下载权重：

```bash
python -m unittest discover -s tests -p test_target_text.py -v
```

同一份真实权重可运行独立验收：

```bash
python -m minideepinfer.verify_target_text \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/verify-single-model.json
```

检查器只加载一次模型，比较完整前缀采样、带缓存生成、两个 token 后取消和人为指定 EOS 的停止结果。后两项检查使用相同随机种子，只比较实际交付的前缀；普通采样的报告不会仅因生成成功就自动写入 `passed: true`。

Ascend 910C 上已用 Qwen3.5-2B 跑过这条 API 路径：[原始报告](results/2026-10-25/qwen35-target-text.json)、[运行日志](results/2026-10-25/qwen35-target-text.txt)。raw 输入 5 个 token、输出上限 6、温度 0.7、top-k 20、top-p 0.9、seed 2026，FP32、HF32 关闭。缓存输出与独立重算的 6 个 ID 完全一致；取消交付前两个，配置 EOS 交付第一个，三次均归还 10 个槽位。包括参考在内共 15 次 forward、66 个输入 token，没有额外加载、计时或 NPU 文本流式 CLI 实测。

核对报告与源码：

```bash
python scripts/check_evidence.py --index docs/results/target-text/evidence.json
```

上面的首次报告按 `4ad9655` 保存，原文件不改。去除多余快照后的记录使用 `--index docs/results/target-snapshots/evidence.json` 核对；[新运行日志](results/2026-10-25/qwen35-target-no-snapshot.txt)保留设备告警。

这是单请求接口，不接入批次、分块 prefill 或跨轮 pipeline；这些入口保持各自已有的执行方式。本次不增加 HTTP 服务，也不把概率从主机移到设备。

## English quick start

On **main / 0.6.0.dev0**, `generate --single` loads one checkpoint and supports temperature, top-k/top-p, streaming, cancellation and JSON reports. Update the project from main before running the commands above; the fixed v0.5.0 tag does not include this entry point.

The text frontend calls the existing synchronous generation loop with draft width zero. There is no draft model or cache. Each commit contains one token; EOS and length completion take precedence over cancellation. The final delivered token remains pending and needs no extra forward. Sampling uses request-local randomness and CPU FP64 probabilities, with FP32 models on CPU or Ascend 910C.

The verifier compares cached generation with independent full-prefix sampling, then checks cancellation and an artificial EOS. It measures correctness for one short input, not performance or arbitrary-model distribution equivalence. The CPU tests also exercise the CLI's single load, text streaming and report saving.

The saved Ascend 910C Qwen3.5-2B run passed all three API cases: six sampled tokens, cancellation after two tokens, and configured EOS after one. All ten slots were returned each time. The checkpoint loaded once; the complete check used 15 forwards and 66 input tokens, including independent references. NPU CLI text streaming was not measured.

A later check removed unnecessary recurrent-state snapshots when no candidates are supplied. Normal six-token generation now makes zero snapshots instead of copying 101,007,360 bytes across five snapshots. All other report fields, except source hashes, match the preserved earlier run. This is a reduction in copied state, not a measured latency improvement.
