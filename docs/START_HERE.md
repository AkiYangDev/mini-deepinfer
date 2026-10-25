# 从安装到第一次运行

[返回首页](../README.md) · [English setup](START_HERE.en.md) · [学习路线](LEARNING_PATH.md)

没有 NPU 或权重，也可以从第一节开始。基础包不依赖 PyTorch，整数例子装好就能运行；后面两节再分别准备 CPU 小模型和 Ascend 真权重环境。

以下多行命令使用 Bash；在 Windows PowerShell 中，请将一条命令写成一行。安装和运行都使用同一个 Python 环境。

下面使用固定的 v0.4.0，便于按同一版本跟读。已有 v0.3 环境时，可按后面的[双模型接续步骤](#v04让小模型提出候选)升级；原来的整数、跨轮和 lookup 例子仍可运行。

## 先看请求怎样执行

整数例子支持 Python 3.10+。如果还要继续跑 CPU 小模型，下面用 Python 3.12 创建环境，与本项目的 CPU 检查版本一致。

只看整数例子时，可用已有 Python 3.10+ 执行 `python -m venv .venv`；下面分别给出 Python 3.12 的创建和激活命令。

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
git checkout v0.4.0
```

Linux/macOS 创建并激活环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 创建并激活环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

后续命令在仓库目录执行：

```bash
python -m pip install -e .
python -m minideepinfer
python -m minideepinfer.batch_demo
python -m minideepinfer.step_demo
```

前两个例子用整数代替模型输出，展示单请求循环和批次调度。第三个例子手动推进完成事件，观察请求换行和取消后，缓存为什么还要保留。接着可以从[第一课](FIRST_REQUEST.md)跟读代码。

## CPU 小模型：无需下载权重

在上面的 Python 3.12 环境中，只需安装 CPU PyTorch。Linux/Windows 使用：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

macOS 使用：

```bash
python -m pip install torch==2.5.1
```

下载源按 [PyTorch 2.5.1 官方说明](https://pytorch.org/get-started/previous-versions/#v251)区分平台；本项目 CPU 数值检查覆盖 Windows/Linux，未新增 macOS 实测。已有兼容 PyTorch 时跳过安装。下面的随机模型直接用项目代码计算，不需要 Transformers、tokenizer、safetensors 或 checkpoint：

```bash
python -m minideepinfer.pipeline_demo
```

真实权重的模型接口依赖放在后面的 Ascend 一节，不用为了跑这个例子提前安装。

程序随机初始化一个小型 Qwen3.5 模型，保留线性注意力和完整注意力。A 的长 prompt 分块计算，B 先结束，C 等待后复用 B 的槽位。最后应看到 `whole-prompt reference outputs match: True` 和 `free slots: 20/20; clean: True`。每轮输出的解释见[A、B、C 演示](WALKTHROUGH.md)。

接着通过同一个文本前端，检查完整 logits 和两种到达方式：

```bash
python -m minideepinfer.verify_pipeline_text --device cpu \
  --output outputs/pipeline-text-cpu.json
```

随机模型不使用自然 EOS；真实权重模式才增加聊天模板与自然 EOS 对照。浮点误差会随设备和软件版本略有变化，以报告中的容差检查为准。

## v0.3：候选猜错后，怎样接着算

CPU 小模型依赖准备好后，按顺序运行：

```bash
python -m minideepinfer.spec_demo --output outputs/spec-controlled.json
python -m minideepinfer.spec_demo --hybrid --output outputs/spec-hybrid.json
python -m minideepinfer.spec_demo --draft lookup --hybrid --output outputs/spec-lookup.json
```

三步分别观察候选验证、混合状态回退、独立候选来源。最后一个模式在生成完成后才运行普通 greedy 对照。JSON 应有 `passed: true`，并记录输出、槽位归还和各类计算成本。例子的完整解释与实机范围见[v0.3 说明](RELEASE_v0.3.md)；这里没有计时或性能承诺。

以第三个命令生成的 `outputs/spec-lookup.json` 为例，先看这几项：

| 字段 | 这个 CPU 例子应看到什么 |
| --- | --- |
| `passed` | `true`，整体检查通过 |
| `cases.lookup.matches_greedy` | `true`，生成输出与事后独立 greedy 相同 |
| `cases.lookup.free_slots` | `12`，请求结束后全部归还 |
| `cases.lookup.target_forward_calls` | `6`，这次生成实际调用 target 的次数，包含重算 |
| `baseline.forward_calls` | `10`，事后对照另外计算了十次 |
| `reference_forward_calls` | `5`，五次提交后的冷缓存检查 |
| `total_forward_calls` | `21`，上述三部分之和 |

`target_forward_calls` 统计生成路径，`total_forward_calls` 还包含验证例子是否正确的额外计算。这里没有测量耗时，不能把 `10/6` 当作加速倍数。随机小模型在这个 seed 下连续输出相同 token，容易被 lookup 命中；真实权重的 0/5 接受结果另见[独立候选说明](LOOKUP_DRAFT.md#ascend-910c-的真实权重结果)。

## v0.4：让小模型提出候选

这一节需要前面装好的 CPU PyTorch。先在仓库目录切到 v0.4.0，并重新安装项目，让当前环境的包信息也跟上源码：

```bash
git checkout v0.4.0
python -m pip install -e . --no-deps
python -c "from importlib.metadata import version; print(version('mini-deepinfer'))"
python -m minideepinfer.model_draft_demo --output outputs/model-draft-cpu.json
```

当前版本号应为 `0.4.0`。这个例子随机初始化两个 Qwen3.5 小模型，让其中一个提出候选，另一个验证；不读取权重，也不下载文件。默认只检查正确性，不计时。最后应看到 `greedy outputs and both caches match: True`，并分别打印宽度 1 和 3 的结果。

打开生成的 JSON，先看宽度 1 的这一组：

| 字段 | 当前 CPU 例子应看到什么 |
| --- | --- |
| `passed` | `true`，输出、缓存和清理检查通过 |
| `checked_runs[0].target_forward_calls` | `6`，生成路径的实际调用数；发生重算时也计入 |
| `checked_runs[0].draft_work.forward_calls` | `4`，draft 的计算也要计入成本 |
| `checked_runs[0].draft_work.snapshot_count` | `0`，只提一个候选，没有临时输入需要回退 |
| `checked_runs[0].draft_work.closed` | `true`，draft 已结束并清理 |
| `checked_runs[0].draft_work.free_slots` | `12`，draft 自己的槽位全部归还 |

随机模型这次会重复输出相同 token，不能用它的接受率推断真实模型性能。两个模型怎样接续状态、真实 0.8B→2B 的成本见[双模型说明](MODEL_DRAFT.md)。如果只想观察多余快照被省掉的前后差别，另行运行：

```bash
python -m minideepinfer.model_draft_demo --compare-draft-snapshots --output outputs/draft-snapshot-cpu.json
```

两份输出保存在不同文件里。第二份报告比较旧、新快照方式，仍不计时；各阶段的模型工作量必须相同，才能通过检查。

想看到候选被拒绝后的完整过程，按[逐轮演示](MODEL_DRAFT_WALKTHROUGH.md)运行 `--cpu-example rejection --trace-rounds --width 3`；这组未训练随机模型不共享输入嵌入与输出头，不需要下载权重。

## Ascend 910C：真实权重

这一节从已有的 Ascend 环境开始。本项目实际使用的软件组合如下，安装时先保留配套的 PyTorch、torch_npu 和 CANN：

| 组件 | 本项目实测版本 |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0.post6 |
| CANN | 9.1.0，aarch64 |
| 模型接口 | Transformers 5.12.1，safetensors 0.8.0 |

这里的 PyTorch 版本确实带有 `+cpu` 后缀，NPU 后端由 torch_npu 注册。按整套环境核对版本即可，不需要因为这个后缀更换 PyTorch。

在这套环境中，先保留设备依赖，再安装 Python 模型接口与项目：

```bash
python -m pip install transformers==5.12.1 safetensors==0.8.0
python -m pip install -e . --no-deps
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

最后应能看到 NPU 可用。随后准备固定 revision 的权重；已经有同一份权重时直接复用，运行阶段只读本地文件，不自动下载。

```bash
hf download Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --local-dir weights/Qwen3.5-2B
```

运行前确认所选逻辑设备可用，并为 FP32 权重、缓存和工作区留出空间。下面使用 `npu:0`，请按当前容器或机器的设备映射选择编号。

先运行真实生成：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

这个入口走同步 BatchEngine。输出末尾会打印停止原因、输出长度和空闲槽位；示例两个请求都遇 EOS 的[原始输出](results/2026-10-24/qwen35-2b-generation-npu-fp32.txt)已保存。最后一个预测 token 不再送回模型，因此请求预留 `prompt 长度 + 最大输出数 - 1` 个槽位。容量不足时请求被拒绝或排队，不会按设备空闲内存自动扩容。

要让自己的文本走跨轮执行，在上面的生成命令末尾加 `--pipeline`。加上 `--stagger-prompts` 后，其余文本每收取一轮结果加入一条；`--trace-rounds` 打印每轮输入和观察到的缓存回收。当前跨轮入口限定 FP32，与 `--deferred`、`--prepare-during-step` 分开选择，详见[运行过程](PIPELINE_TEXT.md)。

无需权重也能先观察同一套文本接入和逐轮输出：

```bash
python -m minideepinfer.verify_pipeline_text --device cpu --trace-rounds \
  --output outputs/pipeline-text-cpu.json
```

原有固定短轨迹检查继续保留：

```bash
python -m minideepinfer.verify_chunk_pipeline \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/chunk-pipeline-npu.json
```

它会执行参考与实验路径，比生成例子多做计算。已有公开结果只想核对源码时，不必重跑实机：

```bash
python scripts/check_evidence.py
```

脚本按索引读取报告对应的固定 Git commit，或核对未指定版本的当前文件；历史通过不表示新代码重新跑过实机。v0.3 投机结果使用 `--index docs/results/v0.3/evidence.json`。完整规则和源码包不含历史的区别见[证据说明](EVIDENCE.md)。

## 让自己的文本用双模型生成

main 另有[`generate --speculative`](SPECULATIVE_TEXT.md)，可以让自己的文本使用独立小模型提出候选。它是单请求 FP32 入口，不能直接拼接上面的批处理参数；固定 v0.4.0 安装包不包含该功能。

## 常见卡点

| 现象 | 先检查什么 |
| --- | --- |
| `No module named torch` | 当前环境只装了基础包；按 CPU 路径安装模型依赖，或切回已有 NPU 环境 |
| `No module named minideepinfer.model_draft_demo` | 双模型入口需要 v0.4.0 或更新源码；按上面的接续步骤切换并重新安装，核对当前 Python 打印的包版本 |
| NPU 不可用或导入 torch_npu 失败 | PyTorch、torch_npu、CANN 和驱动是否来自匹配环境，设备是否映射进当前容器 |
| whole prompt 超过预算 | 开启 `--chunked-prefill`；每请求最大占用仍需不超过缓存容量 |
| 长 prompt 提示容量不足 | `--cache-capacity` 约束逻辑槽位，不等于设备剩余 MiB；同时核对最大输出长度 |
| 旧 Transformers 无法识别 Qwen3.5 tokenizer | 本例固定使用 5.12.1；4.46.3 只用于旧 Qwen2 回归 |
| BF16 不同分块得到不同文本 | 先读[BF16 说明](BF16_REFERENCE.md)，区分扫描方式和状态精度；v0.1 主例使用 FP32 |

模型依赖测试运行 `python -m unittest discover -s tests -v`；未装模型依赖时，模型项明确跳过。安装包包含运行模块，教程、原始报告、测试和 `scripts` 从源码仓库读取；各版本通过仓库或 [GitHub Releases](https://github.com/AkiYangDev/mini-deepinfer/releases)安装，尚未发布到 PyPI。
