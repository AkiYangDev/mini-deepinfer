# 第四课：Qwen3.5 的两种缓存

[返回首页](../README.md) · [第三课：同步批处理](BATCHING.md) · [验证记录](VALIDATION.md)

这一课使用 2026 年发布的 Qwen3.5 小模型。0.8B 用于开发验证，2B 作为主要运行示例。先只处理文本，使用项目自己的前向计算和状态缓存；视觉编码器、MTP、量化与专用融合算子不在本课范围内。

## 为什么 KV Cache 不够用

Qwen3.5-0.8B 和 2B 都有 24 层，每组三层 Gated DeltaNet 后接一层完整注意力。两种层记住历史的方式不同：

| 层 | 保留什么 | 怎样找到它 | 随什么增长 |
| --- | --- | --- | --- |
| 完整注意力 | 每个历史 token 的 K、V | 层号与逻辑 token 槽位 | 已计算 token 数 |
| Gated DeltaNet | 卷积窗口与递推矩阵 | 请求 ID 与层号 | 活跃请求数；单请求状态大小固定 |

[`HybridCache`](../minideepinfer/hybrid_cache.py) 只为六层完整注意力分配 KV 张量。线性层状态首次使用时创建，请求完成、取消或失败后删除。`max_num_seqs` 限制活跃请求数；逻辑 token 槽位预算仍控制准入和最大上下文。它还不是按设备空闲内存自动计算容量的管理器。

在这两个型号中，每个请求仅 FP32 递推矩阵就约占 18 MiB，另有卷积窗口。状态不随上下文长度增长，但多个并发请求仍各自需要一份。

## 沿一层 DeltaNet 读计算

实现位于 [`GatedDeltaNet.forward()`](../minideepinfer/models/qwen3_5.py)。先对合并后的 token 做投影，再按请求边界分别计算卷积和递推。令 `S` 为上一 token 的状态，`k`、`v` 为当前输入：

```text
衰减历史：S = exp(g) * S
预测当前值：prediction = kᵀ S
计算修正：delta = beta * (v - prediction)
写入状态：S = S + k deltaᵀ
读出结果：output = qᵀ S
```

`q`、`k` 先归一化，`q` 再按 head dimension 缩放；递推保持 FP32。读出后做带 SiLU gate 的归一化与输出投影。完整注意力另有 Q/K 归一化、部分 RoPE 和输出 gate，不能直接复用 Qwen2 attention。

这里逐 token 展开递推，便于观察状态如何改变。它没有采用上游的分块并行 DeltaNet 内核，也不承诺与成熟框架相同的吞吐。

## 两个请求为什么不会串状态

Runner 把每个 `Step` 的请求 ID 和 token 边界传给模型。线性层分别读取 `(request_id, layer)` 对应的状态，不能把 `[A 的 token, B 的 token]` 当成一段连续历史扫描。完整注意力仍通过隔离 mask 阻止跨请求读取 KV。

prompt 分块时，中间块更新两类缓存，但不采样；最后一块才产生第一个输出 token。取消一个尚未完成 prompt 的请求，也必须同时删除递推状态和归还逻辑槽位。相关对照见 [`test_qwen3_5.py`](../tests/test_qwen3_5.py)。

## 准备权重

CPU 环境可以安装本课固定的参考依赖。Ascend 环境先保留已经匹配的 PyTorch、torch_npu 与 CANN，再安装本项目；不要为了模型接口直接替换现有设备环境。

```bash
python -m pip install -e ".[qwen35]"

hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir weights/Qwen3.5-0.8B

hf download Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --local-dir weights/Qwen3.5-2B
```

官方 checkpoint 同时包含文本、视觉和 MTP 参数。加载器读取原始 safetensors 索引，只加载文本部分，并恢复共享 embedding / LM head；不需要先转换权重。

## 运行与对照

两个 `--prompt` 会进入同一个同步批处理引擎。默认关闭思考模板并采用 greedy：

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device cpu --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

可先将模型改为 0.8B，减少加载和调试成本。NPU 的 dtype 与实际验证范围见[验证记录](VALIDATION.md)；CPU 也需要加载完整文本模型。

固定 token 对照比较项目缓存、项目完整重算，以及 Transformers 的缓存与重算：

```bash
python -m minideepinfer.verify \
  --model weights/Qwen3.5-0.8B --device cpu --dtype float32 \
  --prefill-chunk-size 2 --output qwen35-cpu.json

python -m minideepinfer.verify_batch \
  --model weights/Qwen3.5-0.8B --device cpu --dtype float32 \
  --output qwen35-batch-cpu.json
```

Qwen3.5 的批次对照使用独立的 Transformers 请求，各自保存状态。普通 attention mask 无法隔离线性层递推，不能直接套用 Qwen2 的 packed 参考方法。

BF16 需要另外说明扫描方式与状态存储精度。固定轨迹已在逐 token / FP32 状态参考下零容差通过，原生分块差异保留；命令与边界见[BF16 对照说明](BF16_REFERENCE.md)。

## 与 DeepInfer 的关系

模型计算对照 Qwen / Transformers；引擎的输入准备、执行、采样、结果提交和资源归属继续按照 [DeepInfer 职责映射](SOURCE_MAP.md)展开。本课展示同一请求生命周期怎样管理不同模型状态，不把 Qwen3.5 的 DeltaNet 称为 DeepInfer 特有机制。
