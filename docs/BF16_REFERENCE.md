# BF16 对照：计算路径和缓存精度要一起说明

[返回首页](../README.md) · [第四课](QWEN35.md) · [原始验证记录](VALIDATION.md)

Qwen3.5 的权重使用 BF16，不意味着所有中间量和缓存也都应该使用 BF16。本项目逐 token 计算 DeltaNet，并以 FP32 保存递推状态。Transformers 5.12.1 的默认路径则采用分块 prefill，默认线性缓存会按卷积输入的 dtype 分配递推存储。

此前直接比较这两条路径，0.8B 有最大 0.25 的 logits 差异。只看到这个数，无法判断是模型公式、运算顺序还是状态保存出了问题。

## 四组实验，分别改变一个条件

固定同一份 0.8B 权重、同一设备和同一 token 轨迹：5-token prompt 按 `[2, 2, 1]` 切分，再做 4 步固定 decode。参考始终使用 Transformers 自己的投影、归一化、注意力和 MLP，只改变 DeltaNet 的扫描方式与递推状态存储类型。

| 参考扫描方式 | 参考递推存储 | 项目与参考缓存的最大差异 | 参考自身缓存与重算的最大差异 |
| --- | --- | --- | --- |
| 原生分块 | 原生 BF16 | 0.25 | 0.203125 |
| 原生分块 | FP32 | 0.15625 | 0.15625 |
| 逐 token | 原生 BF16 | 0.203125 | 0.203125 |
| 逐 token | FP32 | **0** | **0** |

完整逐步数据见[四组对照](results/2026-10-24/bf16-factorial-diagnosis.json)。这说明同一用例同时受到扫描顺序和状态存储精度影响，不能只改其中一个就宣告对齐。

例如，FP32 中的 `1 + 2^-15` 写入 BF16 后会变成 `1`。下一步即使再转回 FP32，丢失的残差也不会回来。递推内部用 FP32 计算，与跨步保存 FP32 状态，是两个不同的条件。

## 显式选择同语义参考

验证工具新增 `--reference-mode recurrent-fp32`。它调用 Transformers 5.12.1 自己的逐 token 递推公式，并用独立缓存保存 FP32 状态。实现见 [`reference_qwen3_5.py`](../minideepinfer/reference_qwen3_5.py)。

适配只作用于验证进程中的参考模型，调用结束或抛错都会恢复原函数，不修改安装的 Transformers 文件。生成路径和本项目模型计算不使用这个适配器。版本限定为 5.12.1，拒绝混用原生和 FP32 参考缓存。

```bash
python -m minideepinfer.verify \
  --model weights/Qwen3.5-0.8B --device npu:0 --dtype bfloat16 \
  --prefill-chunk-size 2 --reference-mode recurrent-fp32 \
  --atol 0 --rtol 0 --output qwen35-bf16.json

python -m minideepinfer.verify_batch \
  --model weights/Qwen3.5-2B --device npu:0 --dtype bfloat16 \
  --reference-mode recurrent-fp32 --atol 0 --rtol 0 \
  --output qwen35-bf16-batch.json
```

在已记录的 Ascend 910C / PyTorch 2.10.0 / torch_npu 2.10.0.post6 / CANN 9.1.0 环境中：

| 检查 | 实际结果 |
| --- | --- |
| 0.8B 分块 prefill + 4 步 decode | [最大差异 0，零容差通过](results/2026-10-24/qwen35-0.8B-npu-bf16-recurrent.json) |
| 2B 分块 prefill + 4 步 decode | [最大差异 0，零容差通过](results/2026-10-24/qwen35-2B-npu-bf16-recurrent.json) |
| 2B 双请求、交错槽位 | [最大差异 0，零容差通过](results/2026-10-24/qwen35-2b-batch-npu-bf16-recurrent.json)，结束后线性状态为 0 |

JSON 明确记录 `reference_mode`、适配范围、实际 token、依赖和适配器源码 SHA。省略该参数时仍检查 `native` 路径，保留原来的行为与失败结果。

## 结果说明到哪里

这些结果验证了固定轨迹上的模型计算和缓存续算，不能扩展成“与所有参考内核逐位一致”。原生分块路径的差异仍存在；完整 greedy 生成也要单独检查，因为很小的 logits 变化就可能改变后续 token。

2B 的[完整生成检查](results/2026-10-24/qwen35-2b-generation-npu-bf16.json)保留了一个反例。两请求均遇 EOS，槽位与线性状态全部回收；英文回答一致，中文回答则在第 38 个输出 token 分岔。整段执行与独立递推参考均输出 49 token，分块执行输出 47 token，因此该报告的 `passed` 保持 `false`。

随后固定相同前缀，分别比较项目与参考的整段和 `[16, 5]` prompt 分块执行，[逐项结果](results/2026-10-24/bf16-generation-divergence.json)显示：

- 整段执行中，“显”（token 96547）与“后续”（token 104218）的 logit 都为 `16.75`；`argmax` 返回词表中较早的“显”。`topk` 对并列元素的排列不代表 `argmax` 的选择顺序。
- 分块后“后续”为 `16.75`，“显”为 `16.625`，预测转向“后续”。独立 Transformers 递推参考也在相同位置翻转。
- 该位置项目整段与参考整段的最大误差为 0；项目与参考分块路径仍有最大 `0.234375` 差异，未通过此前的 `atol=0.125, rtol=0.01`。参考自身整段/分块的最大差异为 `0.1875`，也没有通过这个门槛。

这些事实支持该位置存在低精度舍入与并列候选敏感性，同时保留了较长分块轨迹的额外数值差异。不能把“参考也分岔”当成所有布局数值对齐已经完成，更不能将短轨迹的零误差推广为跨调度完全一致。

复现完整生成检查：

```bash
PYTHONPATH=. python docs/results/2026-10-24/check_generation_bf16.py \
  --model weights/Qwen3.5-2B --device npu:0 --output generation-bf16.json
```

这是零差异生成检查；上述固定版本和用例会记录不一致并以非零退出码结束，结果文件仍会保存。它不是性能基准，也不是要求任意低精度计算布局都必须产生相同文本。

验证时需要分别回答三个问题：同一计算与存储语义是否一致，另一种实现的舍入差异有多大，以及这些差异是否改变完整生成。不能用其中一个问题的结果替代另外两个。
