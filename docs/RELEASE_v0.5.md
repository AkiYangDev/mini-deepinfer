# mini-DeepInfer v0.5.0

[返回首页](../README.md) · [安装指南](START_HERE.md) · [采样教程](SAMPLING_FILTERS.md)

v0.5 把双模型教学接到自己的文本：可以逐组看输出，在提交边界取消，也可以用温度、top-k / top-p 改变采样范围。候选产生、验证、提交和缓存恢复仍走原来的同步循环，读代码时不用再追另一套引擎。

这一版继续使用 Qwen3.5-0.8B / 2B，保留 CPU 小模型演示和 Ascend 910C 的真实权重路径。十四课的历史编号不变，采样教程接在双模型讲解之后。

## 三条学习路线

| 想看什么 | 从哪里开始 |
| --- | --- |
| 请求怎样进入批次，缓存何时归还 | [安装指南](START_HERE.md) → [A、B、C 演示](WALKTHROUGH.md) |
| 小模型猜错后怎样恢复 | [双模型逐轮演示](MODEL_DRAFT_WALKTHROUGH.md) → [自己的文本](SPECULATIVE_TEXT.md) |
| 过滤候选后，接受概率怎样计算 | [随机采样](SAMPLING.md) → [top-k / top-p](SAMPLING_FILTERS.md) |

已有兼容 CPU PyTorch 时，可以先运行不下载权重的过滤例子：

```bash
python -m minideepinfer.sampling_filters_demo --output outputs/sampling-filters-cpu.json
```

它先把四个 token 的概率从 `45/30/15/10` 过滤为 `60/40/0/0`，再用两套随机小模型检查条件概率、回退、取消和停止。随机模型只用于观察计算与状态，不展示回答能力。

## 相比 v0.4 新增了什么

- **自己的文本。** `generate --speculative` 使用本地 tokenizer 与聊天模板，也支持 `--raw`。加载前核对两模型的 tokenizer 文件与特殊 ID。
- **逐组流式输出与取消。** `--stream` 输出实际提交的 token；`--cancel-after-tokens` 在完整组提交后生效，因此可能超过指定门槛。两份缓存各自完成同步与回收。
- **正温度采样。** draft 保存每个候选实际抽出时的 q，target 用同一前缀下的 p 验证；拒绝后从正差额分布补采样。
- **top-k / top-p。** 两模型分别对自己的 logits 过滤并归一化，顺序为温度、top-k、top-p。相同分数按 token ID 排序；默认关闭过滤，保留 greedy 和原温度路径。

这些是单请求、FP32、eager 的教学入口，和跨轮文本调度分别保留。概率与过滤在 CPU FP64 上运行，模型可使用 CPU 或 Ascend 910C FP32；没有新增融合采样算子、MTP、图执行或分布式能力。

## 记录覆盖到哪里

| 已保存的 Ascend 910C 记录 | 实际覆盖 |
| --- | --- |
| [流式取消](SPECULATIVE_TEXT.md#ascend-910c-检查) | 交付 4 个 token 后取消，匹配独立 greedy 前缀，两边缓存全回收 |
| [温度采样](SAMPLING.md#ascend-910c验证概率和回退) | 28 组概率、6 组冷状态对照；实际走过 draft 拒绝后的线性状态恢复 |
| [top-k / top-p](SAMPLING_FILTERS.md#ascend-910c-记录) | 24 组概率、5 组冷状态对照；候选 3/3 全接受，过滤后的拒绝恢复由 CPU 测试覆盖 |

上面的实验均来自开发阶段的固定提交，不在打包时重跑。原始报告、环境、源码 SHA、输入和工作量保持原样；[证据索引](EVIDENCE.md)按当时的代码核对。指定 EOS 的采样检查使用第一枚采样 ID，只验证停止条件，不代表自然语言 EOS。没有新增性能计时或提速结论。

251 项测试在 Transformers 5.12.1 环境通过；4.46.3 环境通过 239 项、跳过 12 项；无模型依赖环境通过 79 项、跳过 172 项。发布包在仓库外检查版本、基础入口、双模型文本与采样测试，以及演示输出。发布整理仅修改包版本和文档，运行模块与功能提交 `f40f6af` 一致。

## 安装固定版本

从源码开始：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
git checkout v0.5.0
python -m pip install -e . --no-deps
python -m minideepinfer.step_demo
```

需要模型例子时，按[中英文安装指南](START_HERE.md)准备依赖；已有 Ascend 环境沿用匹配的 PyTorch、torch_npu、CANN，不因安装本项目而替换它们。

[发布页](https://github.com/AkiYangDev/mini-deepinfer/releases/tag/v0.5.0)提供 wheel、源码包和 `SHA256SUMS`。下载 wheel 后，可用 `python -m pip install --no-deps ./mini_deepinfer-0.5.0-py3-none-any.whl` 安装。项目没有上传 PyPI。源码包保留教程、测试、素材和原始结果；历史证据检查需要 Git 检出及对应提交。

旧版 v0.4.0 的 tag、正文与附件保持不变。角色素材的来源与权利说明见[素材说明](assets/ARTWORK.md)，不随代码许可证重新授权。

## English release notes

v0.5.0 brings the existing two-model loop to custom text, streamed committed groups, cooperative cancellation, positive-temperature sampling and top-k/top-p filtering. Draft and target use their actual conditional, transformed distributions; verification, rollback and cache ownership stay in the same synchronous loop. Default greedy behavior remains available.

Start with the checkpoint-free CPU filtering command above, or follow the [English setup guide](START_HERE.en.md) at the fixed `v0.5.0` tag. The Qwen3.5-0.8B / 2B examples continue to run FP32 models on CPU or Ascend 910C, while dense probabilities and filtering remain CPU FP64.

The release preserves development records at immutable commits. Temperature sampling passes 28 probability and six cold-state checks, including actual draft recovery. The filtered NPU run passes 24 probability and five cold-state checks with 3/3 accepted candidates; filtered rejection is covered on CPU, not in that NPU run. Configured EOS is a stopping-condition test. No NPU experiment is repeated for packaging and no speedup is claimed.

The test matrix has 251 passing tests with Transformers 5.12.1; 239 pass and twelve skip with 4.46.3; the dependency-free environment passes 79 and skips 172. Installed-package checks run outside the checkout. Runtime modules are unchanged from `f40f6af`; this release updates metadata and documentation.

Assets are a wheel, source distribution and SHA256SUMS. No PyPI upload. Older releases remain unchanged; artwork rights are documented separately. MTP/DSpark, SWA bounded replay, fused sampling kernels, graphs and distributed serving are outside this release.
