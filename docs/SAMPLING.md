# 随机采样时，候选为什么还能被接受

[返回首页](../README.md) · [双模型文本入口](SPECULATIVE_TEXT.md) · [缓存回退](SPECULATIVE.md)

greedy 验证只问一个问题：小模型猜的 token，是不是大模型分数最高的那个？随机采样则不同。一个 token 不是第一名，也可能被正常抽中。继续比较 argmax，就会改变生成的概率。

main 的 CPU 参考现在支持正温度采样。它复用已有的双模型循环、缓存回退、逐组提交和取消，只替换候选的生成与接受规则。正式版 v0.4.0 不包含这个入口；当前开发版本仍为 `0.5.0.dev0`。

## 先运行，不下载权重

从仓库根目录执行。已有 PyTorch 的环境只需最后一条命令：

```bash
python -m pip install -e ".[model]"
python -m minideepinfer.sampling_demo --output outputs/sampling-cpu.json
```

这个例子先做两 token 的小词表实验，再运行两个未训练的随机 Qwen3.5 模型。target 与 draft 的权重不同、输出头独立；它们用于观察计算和状态，不用于展示回答质量。

保存的[CPU 记录](results/2026-10-25/sampling-cpu.json)中，一次完整接受三个候选，提交四个 token；下一轮接受一个、拒绝一个，target 缓存从 11 个输入恢复到 9 个，再继续生成。18 次冷缓存对照通过，结束后两边各归还 12 个槽位。不同软件版本的浮点差异可能改变随机轨迹，检查重点是规则和状态。

## 接受与补采样

用 `p` 表示 target 的分布，`q` 表示 draft 的分布。它们都经过相同的温度变换，并且必须对应同一个前缀。

假设词表只有 A、B：target 希望 A 占 25%、B 占 75%，draft 却把 A 猜成 75%。draft 抽中 A 时，只接受其中的三分之一；抽中 B 时全部接受。拒绝的那部分还需要补回来。

代码按以下规则执行：

- draft 按 `q` 抽候选 `x`，同时保存当时完整的 `q`。
- target 一次计算候选各位置的 `p`。以 `min(1, p[x] / q[x])` 接受候选。
- 第一次拒绝后，从归一化的 `max(p-q, 0)` 抽一个改正 token，本轮到此结束。
- 全部接受时，从最后一行 target 分布再抽一个 bonus。接受到 EOS 则直接停止。

上面的例子拒绝 A 后，差额分布只剩 B。如果直接重新按原来的 `p` 抽，A 就会被重复分配概率。实现见 [`verify_sampled()`](../minideepinfer/sampling.py)，算法依据是 [Leviathan 等人的 Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a.html)。

这里保存的是每个候选**实际被抽出时**的分布。不能用别的前缀重算的 `q`、未除以温度的 logits，或候选中最高的分数来代替。

## 普通采样如何作参照

`TemperatureSampler` 在 CPU 上把 FP32 logits 转成 FP64 概率，用请求自己的随机数对象采样。温度必须为有限正数；本轮没有 top-k、top-p、重复惩罚或概率截断。

演示中的宽度 0 不生成候选，每次直接从 target 抽一个 token。它与独立的整段前缀重算使用相同 seed，输出一致，用于核对普通采样的计算与缓存。

双模型路径会额外抽取候选、接受随机数和补采样结果，消耗随机数的次序已经不同。同一 seed 的两条路径不需要逐 token 相同。小词表实验进一步检查两个连续输出的联合分布：第二个 token 的概率依赖第一个**实际提交**的 token，而不是被拒绝的候选。

保存记录中，目标四种组合的概率为 `[[0.2, 0.05], [0.075, 0.675]]`，一万次观察为 `[[0.2021, 0.0493], [0.0764, 0.6722]]`。这是一项固定小例子的经验检查；一般分布的依据来自算法推导，单次模型输出不能证明这一点。

## 接入已有文本入口

已有本地 CPU 权重时可以运行：

```bash
python -m minideepinfer.generate \
  --model /path/to/Qwen3.5-2B \
  --draft-model /path/to/Qwen3.5-0.8B \
  --speculative --device cpu --temperature 0.7 --seed 2026 \
  --draft-width 3 --prompt "Explain a KV cache briefly." \
  --max-new-tokens 16 --cache-capacity 128 --stream \
  --output outputs/sampled-text.json
```

两套 tokenizer 仍须逐文件一致。`--trace-rounds` 会列出接受概率与抽数，`--cancel-after-tokens N` 仍在完整提交组后生效。target 和 draft 使用独立的请求随机数对象；默认 seed 分别为 2026、2027，不改变 PyTorch 全局随机数状态。

没有指定 `--temperature` 时保留原来的 greedy 行为。随机采样不能同时使用 `--check-greedy`；文本报告也不会因正常生成而自动标成通过了分布验证。

## 和 DeepInfer 的对应及当前边界

固定版本 DeepInfer 的 [`Sampler._verification()`](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/sampler.py#L411) 把采样、候选验证和有效输出整理放在同一条执行路径中。本项目保留这个职责划分：概率规则决定提交几个 token，原来的循环负责回退与资源归属。这里使用独立小模型和经典的顺序拒绝采样，不声称复现上游 MTP、置信度机制或融合 kernel。

CPU 检查覆盖小词表概率、两种缓存、候选条件前缀、普通采样、EOS、取消和失败清理。正温度路径也已通过下面的 Ascend 910C 检查。概率仍留在主机并保留完整词表，没有融合采样 kernel 或吞吐、延迟收益结论。

## Ascend 910C：验证概率和回退

使用已有本地权重执行：

```bash
python -m minideepinfer.verify_sampling \
  --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B \
  --device npu:0 --output outputs/qwen35-sampling.json
```

这条命令会执行冷缓存重算，比普通生成多做计算。已有模型各加载一次，之后各项检查复用只读权重，每个请求使用独立缓存。它不下载权重，也不计时。

[实机报告](results/2026-10-25/qwen35-sampling.json)使用 PyTorch 2.10.0、torch_npu 2.10.0.post6、CANN 9.1.0，FP32、HF32 关闭、CPU 线程 1。raw 输入 `The capital of France is` 共五个 token，温度 0.7，target/draft seed 为 2026/2027，宽度 3、输出上限 6、各有 10 个槽位。

先独立重算整段前缀，验证普通温度采样。然后对 draft 实际保存的每一行 `q`、target 验证用的每一行 `p`，分别用同一前缀的冷模型计算作对照；复用本轮原始随机数，还原接受、改正或 bonus 的决定。28 组概率比较通过，最大绝对差约 `3.38e-6`。门槛为 `atol=1e-6, rtol=2e-4`，绝对差不单独决定是否通过。

投机请求接受 3/5 个候选，生成六个 token 后按长度停止。拒绝后 target 重算两个输入，draft 恢复状态后重算一个输入；六组有效缓存与线性状态的冷对照通过，使用 `atol=1e-4, rtol=2e-4`。这次实机确实走过了 draft 分支恢复，未把 CPU 反例当成实机覆盖。

另一个请求在完整提交组三个 token 后取消，没有启动下一轮；指定第一个抽中 ID 为 EOS 的请求只交付一个 token，draft 不计算。后一项验证的是配置停止条件，不代表自然语言回答遇到了模型自己的 EOS。所有请求结束后各池都归还 10 个槽位。

完整检查共 54 次 forward、294 个输入 token，包含普通生成、两种停止检查、条件概率参考和冷缓存检查，不只统计投机主循环。没有计时、profiling 或提速结论。短输入也不能证明所有模型输入的分布等价；数学规则、小词表实验和这次实机数值检查各自回答不同的问题。

文本入口现在允许 `--speculative --device npu:0 --temperature 0.7`，概率仍在主机计算。这里保存的实机记录来自数值验收入口；新温度参数下的 NPU 文本流式 CLI 没有另跑一遍，CPU 的文本、流式与取消测试继续维护。

张量转换采用先 `logits.detach().cpu()`、再在 CPU 转 FP64 的顺序，并以三个数做过当前环境的实测，避免让 NPU 承担 FP64 转换。[官方版本对应表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)用于核对版本组合；接口文档页面未返回可用的方法表，实际搬运行为以这次固定环境的小输入实测为准。[原始运行日志](results/2026-10-25/qwen35-sampling.txt)保留了环境告警，没有隐藏 warning。

## English usage

Run `python -m minideepinfer.sampling_demo --output outputs/sampling-cpu.json` with PyTorch installed. It needs no checkpoint and generates no meaningful language: both Qwen3.5 models are untrained.

The CPU reference reuses the existing speculative loop and rollback. The draft stores the actual conditional proposal distribution at each position. Verification uses classic rejection sampling; see the linked Algorithm 1 and `sampling.py`. Width zero is checked against independent full-prefix temperature sampling. A separate two-token toy experiment checks conditional output frequencies. Neither test claims a proof for all model inputs.

For local CPU text generation, add `--temperature 0.7 --seed 2026` to the existing `generate --speculative` command and use `--device cpu`. Matching tokenizer files remain mandatory. Streaming, whole-group cancellation and round traces remain available. The draft uses seed 2027 by default. These request-local generators do not change PyTorch's global RNG. Ordinary and speculative sampling consume random numbers differently, so identical seeds need not give identical tokens.

Omit temperature to retain greedy behavior. `--check-greedy` is incompatible with random sampling. Dense probabilities stay in CPU FP64: FP32 logits transfer to the host before conversion.

Ascend 910C sampling has a separate [saved validation](results/2026-10-25/qwen35-sampling.json). Run `python -m minideepinfer.verify_sampling --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B --device npu:0 --output outputs/qwen35-sampling.json`. Each existing checkpoint loads once. The environment uses PyTorch 2.10.0 / torch_npu 2.10.0.post6 / CANN 9.1.0, FP32 with HF32 disabled and one CPU thread. The check uses one raw five-token prompt, temperature 0.7, seeds 2026/2027, width 3 and six output tokens.

All 28 conditional probability comparisons and six cold-state checks pass. Cold probabilities reproduce verification decisions using the original random draws. The sampled case accepts 3/5 proposals; both models replay retained inputs after rejection. Cancellation delivers one whole three-token commit prefix. A separate configured-EOS case stops on its first sampled ID; it does not test natural model EOS. All pools return 10 slots. The complete check costs 54 forwards and 294 input tokens, including references. There is no timing, profiling, speedup claim, general distribution proof, or additional NPU streaming-CLI run. CPU text/stream/cancellation tests remain in place.
