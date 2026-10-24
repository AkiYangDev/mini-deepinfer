# mini-DeepInfer v0.2.0

[返回首页](../README.md) · [A、B、C 演示](WALKTHROUGH.md) · [运行指南](START_HERE.md)

v0.2 把已有跨轮执行接到文本生成入口。现在可以输入自己的文本，观察请求进入、分块 prompt、生成结束和缓存交接。新加的 A、B、C 演示用随机小模型走同一个循环，不需要下载权重。

## 从哪开始

已有兼容 PyTorch 时，运行：

```bash
python -m minideepinfer.pipeline_demo
```

A 的九个 prompt token 分块处理，B 先生成结束，C 等待并复用 B 的槽位。[中英文讲解](WALKTHROUGH.md)解释每次观察发生在哪个位置，以及 CPU 已提交进度为什么落后于已发出进度。程序会与逐请求整段 prompt 的同步输出比较，并检查全部 20 个槽位归还。

真实文本通过 `generate --pipeline` 选择跨轮循环。`--stagger-prompts` 每收取一轮加入一条后续文本，`--trace-rounds` 打印执行与缓存归属；完整命令见[文本入口](PIPELINE_TEXT.md)。同步和单批 deferred 路径继续保留。

## 检查结果

共 150 项测试：Transformers 5.12.1 环境全部通过；4.46.3 环境通过 138 项、跳过 12 项；基础环境通过 59 项、跳过 91 项。新增演示检查独立缓存的同步输出、等待请求零分配、预算、槽位独占及 B 到 C 的实际复用。安装包在仓库目录外运行基础例子和小模型入口。

文本入口的 Ascend 910C 结果沿用本次开发阶段保存的 Qwen3.5-2B FP32 报告：同步、全部先排队和逐条到达输出一致；真实聊天自然遇 EOS。43 组完整 logits 对比最大绝对差约 `4.94e-5`，通过原 `atol=1e-4, rtol=2e-4`，缓存全部归还。

[v0.2 证据索引](results/v0.2/evidence.json)绑定原报告与十八份相关源码。演示和发布整理未改动这些执行文件，直接复用已有实机证据，没有再加载 2B 权重、采集 trace 或计时。原首次实机对比脚本缺少 EOS 后额外一步参考的问题及两次权重加载成本仍记录在[验证说明](PIPELINE_TEXT.md#实机检查)里。

```bash
python scripts/check_evidence.py --index docs/results/v0.2/evidence.json
```

## 适用范围

跨轮文本路径限定 FP32、单设备、eager、greedy 和短上下文。回调到达按轮触发，没有 HTTP 服务或真实请求到达率。打印和同步 CPU 观察不能证明 NPU 重叠或稳定提速。BF16 跨扫描和调度差异仍见[已有记录](BF16_REFERENCE.md)。v0.1.0 的 tag 和安装包保留。

## English release notes

v0.2.0 connects the existing cross-round pipeline to text generation. Use `generate --pipeline` with your own prompts, `--stagger-prompts` for callback-driven arrivals, and `--trace-rounds` to inspect rounds and ownership.

The new [A/B/C walkthrough](WALKTHROUGH.en.md) runs a tiny random CPU model without a checkpoint. A has a chunked prompt, B finishes first, and C waits before reusing B's slots. The program checks final outputs against independent-cache, whole-prompt synchronous execution. It adds no scheduling loop.

All 150 tests pass with Transformers 5.12.1; 138 pass and 12 skip with 4.46.3; the base environment passes 59 and skips 91 model checks. Package checks run outside the checkout. Preserved Qwen3.5-2B FP32 results on Ascend 910C cover 43 full-logit comparisons, with maximum absolute error about `4.94e-5`, matching output and clean caches. The eighteen related source hashes remain unchanged, so release preparation reuses those results without another hardware run.

Cross-round text generation remains single-device, eager, greedy, FP32, and intended for short teaching examples. Arrival callbacks and printed snapshots do not establish a throughput improvement. The existing v0.1.0 release is preserved.
