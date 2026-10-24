# 从安装到第一次运行

[返回首页](../README.md) · [English setup](START_HERE.en.md) · [学习路线](LEARNING_PATH.md)

没有 NPU 或权重，也可以从第一节开始。基础包不依赖 PyTorch，整数例子装好就能运行；后面两节再分别准备 CPU 小模型和 Ascend 真权重环境。

以下多行命令使用 Bash；在 Windows PowerShell 中，请将一条命令写成一行。安装和运行都使用同一个 Python 环境。

## 先看请求怎样执行

Python 3.10+。建议在新虚拟环境里安装，避免影响已有推理环境：

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m venv .venv
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
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

在普通 CPU 虚拟环境中安装模型依赖：

```bash
python -m pip install -e ".[qwen35]"
python -m minideepinfer.verify_chunk_pipeline --device cpu \
  --output outputs/chunk-pipeline-cpu.json
```

这个入口只需要 PyTorch；上面的 extra 同时准备后续真实模型使用的 tokenizer 和 safetensors。已有兼容 PyTorch 时，只安装基础包也能运行随机模型检查。

程序随机初始化一个小型 Qwen3.5 模型，保留线性注意力和完整注意力。它先计算整段 prompt 作为参考，再运行分块、到达、取消和复用的轨迹。完成后应看到 `passed: true`、`free_slots: 24`，剩余前缀和线性状态均为 0。浮点误差会随设备和软件版本略有变化，以报告中的容差检查为准。

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

再学习跨轮顺序时，运行独立的固定短轨迹检查：

```bash
python -m minideepinfer.verify_chunk_pipeline \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/chunk-pipeline-npu.json
```

它会执行参考与实验路径，比生成例子多做计算。已有公开结果只想核对源码时，不必重跑实机：

```bash
python scripts/check_evidence.py
```

脚本比较当前文件与实机运行时保存的源码 SHA，并处理 Windows 换行符的差别。如果你改过相关源码，它会指出不一致的文件；此时需要针对改动补验证。各课原始结果对应的版本见[验证记录](VALIDATION.md)。

## 常见卡点

| 现象 | 先检查什么 |
| --- | --- |
| `No module named torch` | 当前环境只装了基础包；按 CPU 路径安装模型依赖，或切回已有 NPU 环境 |
| NPU 不可用或导入 torch_npu 失败 | PyTorch、torch_npu、CANN 和驱动是否来自匹配环境，设备是否映射进当前容器 |
| whole prompt 超过预算 | 开启 `--chunked-prefill`；每请求最大占用仍需不超过缓存容量 |
| 长 prompt 提示容量不足 | `--cache-capacity` 约束逻辑槽位，不等于设备剩余 MiB；同时核对最大输出长度 |
| 旧 Transformers 无法识别 Qwen3.5 tokenizer | 本例固定使用 5.12.1；4.46.3 只用于旧 Qwen2 回归 |
| BF16 不同分块得到不同文本 | 先读[BF16 说明](BF16_REFERENCE.md)，区分扫描方式和状态精度；v0.1 主例使用 FP32 |

模型依赖测试运行 `python -m unittest discover -s tests -v`；未装模型依赖时，模型项明确跳过。安装包包含运行模块，教程、原始报告、测试和 `scripts` 从源码仓库读取；v0.1 没有发布到 PyPI。
