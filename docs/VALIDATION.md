# 验证与实验方法

[返回首页](../README.md)

本文记录实际结果和对应的模型、输入及源码版本。当前 Qwen3.5-0.8B / 2B 已通过 CPU 与 Ascend 910C 的 FP32 数值检查；BF16 仍有未通过的对照。此前 Qwen2 记录按各自历史版本保留。

## 2026-10-24：Qwen3.5 纯文本与混合缓存

0.8B 用于开发验证，2B 为主要示例。两个模型的 revision、文件大小与完整 SHA256 见[权重记录](results/2026-10-24/weights.json)。权重通过镜像传输，下载后与固定 Hugging Face revision 的 LFS SHA256 逐字节哈希核对，两者均一致。

环境为 Atlas 800I A3 / Ascend 910C，Python 3.12.13、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6、CANN 9.1.0、Transformers 5.12.1、safetensors 0.8.0。单个逻辑设备、eager、FP32，CPU 计算线程为 8。参考检查显式关闭 NPU matmul / convolution 的 HF32 近似模式。[依赖记录](results/2026-10-24/environment.json)与每份报告的源码 SHA 分开保存。

固定 prompt 为 `The capital of France is`，实际 5 token 按 `[2, 2, 1]` 分块，再执行 4 步固定 token decode。每步同时比较独立 Transformers 缓存、完整重算，以及项目自身完整重算。

| 模型与设备 | 各项检查的最大绝对误差 | 结果 |
| --- | --- | --- |
| 0.8B / CPU FP32 | `3.4332275390625e-5` | [通过](results/2026-10-24/qwen35-0.8B-cpu-fp32-final.json) |
| 2B / CPU FP32 | `3.910064697265625e-5` | [通过](results/2026-10-24/qwen35-2B-cpu-fp32-final.json) |
| 0.8B / Ascend 910C FP32 | `2.002716064453125e-5` | [通过](results/2026-10-24/qwen35-0.8B-npu-fp32.json) |
| 2B / Ascend 910C FP32 | `2.288818359375e-5` | [通过](results/2026-10-24/qwen35-2B-npu-fp32.json) |
| 2B 双请求 / Ascend 910C FP32 | `2.288818359375e-5` | [通过](results/2026-10-24/qwen35-2b-batch-npu-fp32.json)，结束后前缀记录与线性状态均为 0 |

检查位置的 top-1 全部一致。单请求沿用 `atol=1e-4, rtol=2e-4`，批次沿用跨形状的 `atol=2e-4, rtol=2e-4`；没有为本次结果放宽门槛。双请求使用交错、非连续槽位，各做 prefill 和 4 步固定 decode。Qwen3.5 参考请求各自保存状态，未构造一个会串联 DeltaNet 历史的 packed HF 假参考。

2B 还完成双请求分块生成：中文 KV Cache 解释输出 47 token，英文 `1 plus 1` 回答输出 9 token，两者均遇 EOS 结束，512/512 逻辑槽位回收。[完整输出](results/2026-10-24/qwen35-2b-generation-npu-fp32.txt)按实际结果保留。该结果是运行示例，不是通用回答质量或性能评测。

本地 PyTorch 2.5.1 与服务器 2.10.0 环境均通过 44 项 CPU 测试，新增 8 项 Qwen3.5 对照覆盖整段/分块/缓存、GQA、请求重排与隔离、脏槽位复用、取消与异常回收，以及原始多模态 checkpoint 的纯文本加载和共享权重。BF16 归一化回归测试也做了反例检查：恢复旧计算顺序后测试确实失败。

### BF16：保留失败，不以 top-1 代替数值验收

真实 0.8B 的[首轮 BF16 对照](results/2026-10-24/qwen35-08b-npu-bf16-initial.json)没有通过原 `atol=0.125, rtol=0.01`。权重逐项一致，[卷积隔离与逐层诊断](results/2026-10-24/bf16-initial-diagnosis.json)排除了首层卷积差异。原实现先转 FP32 再归一化 Q/K，而固定的 Transformers 5.12.1 先按投影 dtype 归一化；现已对齐计算顺序并补回归测试。

修正后的[BF16 对照](results/2026-10-24/qwen35-08b-npu-bf16.json)仍未全部通过，最大误差 0.25；检查位置 top-1 一致，项目缓存与重算在该输入上完全一致。参考自身缓存/重算也有最大 0.203125 差异，不能据此直接宣布项目通过。

进一步[扫描隔离](results/2026-10-24/bf16-scan-diagnosis.json)显示相同首层输入的分块与逐 token 输出一致，末状态有约 `5.96e-8` 的 FP32 运算顺序差异；[逐层记录](results/2026-10-24/bf16-layer-diagnosis.json)中前 17 层输出一致，第 18 层开始出现 BF16 舍入差异。Transformers 5.12.1 的默认线性缓存还按卷积 dtype 分配递推存储，本项目则保存 FP32。以上确定了需要分别核对的精度与缓存语义，尚不能把全部误差归为单一原因。

因此，当前示例明确使用 FP32，BF16 继续作为实验路径。没有通过调整门槛抹去失败，也没有吞吐、延迟或相对成熟框架的性能结论。

## 2026-10-21：分块 prefill 验证

真实权重与 CPU 环境沿用下方记录。固定 5-token prompt 按 `[2, 2, 1]` 分块计算，再执行 4 步固定 token decode。7 个步骤与 Transformers 相同输入块的缓存路径 logits 最大绝对差为 **0**；相对完整重算的最大差仍为 `7.557868957519531e-5`，检查位置 top-1 全部一致，未调整原 FP32 容差。原始[逐步结果](results/2026-10-21/qwen2-chunked-cpu-fp32.json)保留输入块、最后一块标记和源码 SHA。

双请求的每轮 token 预算从 32 降到 2，开启 `--chunked-prefill`，各生成 16 token；[完整文本](results/2026-10-21/generation-chunked-cpu.txt)与先前整段 prefill 的输出逐字符一致，结束后 128/128 槽位回收。它仅验证这组输入下的输出与回收行为，没有性能结论。

本地与服务器 CPU 均通过 36 项测试，补充了中间块不输出、decode 预算保留、部分 prompt 取消及复用、过早 decode 拒绝，以及数值脚本完整/分块模式的参考对照。只有最终 prompt 块才允许提交首个输出 token；中间块也不触发 EOS。

## 2026-10-20：同步批次验证

使用同一版本 DeepSeek-R1-Distill-Qwen-1.5B 权重，在 CPU FP32 上执行两个请求。输入分别为 `The capital of France is`（5 token）和 `A KV cache`（3 token），prefill 后各执行 4 步固定 token decode。每轮合并为一次模型调用，使用非连续、交错分配的槽位。

| 检查 | 实际结果 |
| --- | --- |
| 与 Transformers 相同批次布局对照 | 10 个请求步骤的 logits 最大绝对差为 **0**，检查位置 top-1 全部一致 |
| 批次与独立请求对照 | 最大绝对差 `1.4591217041015625e-4`；同布局 Transformers 批次与它自身独立请求也复现了相同差异 |
| 双请求生成 | 各生成 16 token，均按长度停止，128/128 槽位回收；仅为生成流程演示 |
| CPU 自动检查 | 本地与服务器均通过 29 项；包含 40 组固定 seed、共 2,400 次混合生命周期操作及排空后的历史/回收检查 |

原始[最终数值报告](results/2026-10-20/qwen2-batch-cpu-fp32.json)包含实际 token IDs、依赖版本、源码 SHA 和全部逐步误差；[双请求输出](results/2026-10-20/generation-batch-cpu.txt)也一同保存。运行命令与批次布局见[第三课](BATCHING.md)。环境沿用下面单请求记录中的服务器 CPU 组合，计算线程为 8；没有 NPU 或性能结论。

该批次报告的源码对应 [e51e9a12](https://github.com/AkiYangDev/mini-deepinfer/tree/e51e9a12ebad1b45e95e8c3c7882ad9b86c2c3ac)。后续代码演进时，历史 SHA 按对应版本核对。

初次批次对照沿用单请求的 `atol=1e-4, rtol=2e-4`，有一步未通过，见[初次报告](results/2026-10-20/qwen2-batch-cpu-initial.json)。普通 padding 的参考批次只出现约 `9.60e-5` 差异，不能直接代替相同布局对照。随后保留 Transformers 的模型运算，独立构造 mask，并用缓存顺序适配器对齐 packed 布局，复现了相同的 `1.46e-4` 跨形状差异，同时与项目实现逐步对齐为 0。

据此，本例跨批次形状的 FP32 对照使用 `atol=2e-4, rtol=2e-4`；相同布局的 `reference_packed` 仍使用 `atol=1e-4`。原单请求脚本的阈值不变。BF16 容差仍待真实 NPU 对照检验，不能由 FP32 结果推定。适配器仅服务验证，不参与引擎执行，已在 Transformers 4.46.3 与 5.12.1 测试。

新增模型测试检查混合 prefill/decode、请求重排、邻居输入与未来 token 的隔离、非连续槽位复用、完整重算与 greedy 输出一致，以及模型在写入部分 K/V 后抛错时整批失败且不泄漏前缀记录。

## 2026-10-20：单请求模型验证

模型固定为 DeepSeek-R1-Distill-Qwen-1.5B，revision 为 `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`。权重共 3,554,214,621 字节，完整 SHA256 与 Hugging Face LFS 元数据一致，见[权重记录](results/2026-10-20/weights.json)。

| 检查 | 实际范围与结果 |
| --- | --- |
| CPU 单元测试 | 16 项通过；包含两层随机权重的模型对照、非连续槽位、因果遮罩、旧槽位复用和 checkpoint 加载 |
| 真实权重 CPU FP32 | prefill 5 token，加 4 步固定 token decode；项目缓存路径与 Transformers 缓存路径的 logits 最大差为 0 |
| 缓存与完整重算 | 最大绝对差为 `7.557868957519531e-5`；相同差异也出现在 Transformers 自身的缓存/重算对照中，所有检查位置 top-1 一致 |
| CPU 离线生成 | 真实模型生成 96 token，按长度停止，512 个逻辑槽位全部回收；输出为尚未结束的思考片段，不计作回答质量验证 |
| NPU 先行探针 | 单逻辑设备的 BF16 matmul、index_copy_ 和 index_select 通过；缩放顺序调整前的小尺寸随机权重模型也曾通过同设备参考对照 |
| 真实权重 NPU | 加载阶段因设备被其他任务占用而 OOM，尚未进入模型数值检查；最终代码需在空闲设备重新验证 |

真实权重 CPU 环境为 Python 3.12.13、PyTorch 2.10.0+cpu、Transformers 5.12.1、safetensors 0.8.0，使用 8 个计算线程。本地随机权重测试另在 Python 3.12.14、PyTorch 2.5.1+cpu、Transformers 4.46.3、safetensors 0.5.3 通过。

原始[数值结果](results/2026-10-20/qwen2-cpu-fp32.json)与[输入及源码哈希](results/2026-10-20/case.json)一同保存。没有吞吐或延迟结论。

输入记录已按实际运行环境核对：Transformers 5.12.1 将本例 prompt 编码为 5 token，本地 4.46.3 的默认编码会多加一个 BOS。初版 `case.json` 误用了本地的 6-token 编码，现已更正并注明版本；原始数值报告未变。跨版本复现时应同时核对 token IDs。

初次真实权重 FP32 对照未通过 `atol=2e-5` 的设置。逐层检查发现第一层 attention 已有差异：参考实现将分数乘以预计算倒数，原实现除以平方根。改为相同计算顺序后，prefill 与前两步 decode 对齐；后两步的剩余差异与参考模型自身缓存/重算一致。当前本例采用 `atol=1e-4, rtol=2e-4`，报告仍保留实际误差，不能把该阈值推广到全部模型和输入。

NPU 探针使用的环境为 Atlas 800I A3、设备名 `Ascend910_9362`、CANN 9.1.0、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post6。真实权重加载前后的资源状态发生变化，新增服务占用全部逻辑设备。测试进程已退出，完整模型的结果待在空闲设备上补齐。

## 当前可运行检查

在仓库根目录执行：

```bash
python -m unittest discover -s tests -v
python -m minideepinfer
```

当前共 36 项测试：`test_engine.py` 的 8 项与 `test_batch_engine.py` 的 11 项仅依赖标准库；`test_model.py` 的 8 项与 `test_batch_model.py` 的 9 项需要模型依赖，未安装时明确跳过。真实权重的单请求与批次数值对照命令见[第二课](REAL_GENERATION.md)和[第三课](BATCHING.md)。

本轮本地检查环境为 Python 3.14.5；Python 3.10—3.13 与 3.14 的标准库检查由仓库 CI 配置覆盖，执行结果以对应 Actions 记录为准。

## 正确性检查

由小到大建立三层对照：

1. 固定输入下的算子结果与数学参考实现。
2. 单层输出与相同模型版本的参考模型。
3. 固定输入 token 与固定后续 token 轨迹下的整模型 logits/logprob。

teacher forcing 对照在每一步输入同一个 token，避免生成分叉后无法定位误差。greedy 生成与短文本演示是补充；低精度近似下，某个接近并列的 argmax 分叉要回到 logits 判断。

容差根据 dtype、计算路径和误差测量确定，记录绝对/相对误差及必要的分布指标，不预先指定通用于全部算子的阈值。

## 系统边界

覆盖空输入处理、不同输入/输出长度、页边界、EOS、长度上限、请求加入、完成、取消、容量不足，以及资源重用。重点检查缓存串读、读未写区域与在途资源过早释放。

投机实验必须覆盖全部拒绝、部分接受和全部接受，分别检查输出提交、bonus token 及缓存回退。先明确采用的 greedy 或概率采样算法，再验证其正确性条件；模型 MTP/DSpark 的验证独立于可控 draft 实验。

bounded replay 实验需说明模型/层级前提，对照可见 key、位置、槽位与输出。根据[官方部署报告](https://zhuanlan.zhihu.com/p/2095560295930508953)，有限回放用于近似恢复 SWA 状态。应分开验证同一语义下 mask/compact 实现的一致性，以及近似回放相对完整重算的数值和任务质量差异；前者通过不能证明后者严格等价，也不能只依据滑窗长度推断整模型等价。

## 可复现实验记录

每轮保存以下信息：

| 类别 | 必须记录 |
| --- | --- |
| 模型 | 模型 ID、权重 revision、配置、tokenizer、dtype/量化 |
| 环境 | 源码 commit、镜像 digest 或依赖版本、驱动、CANN、PyTorch、`torch_npu` |
| 设备 | 设备数、物理卡与逻辑设备映射、单设备可用内存、并行方式 |
| 负载 | 实际输入/输出 token 数、请求数、并发、请求速率、数据集/seed |
| 缓存 | 复用是否开启、预热/flush 方式、实际命中与统计定义 |
| 区间 | 预热、稳定测量、全部请求窗口分别记录，给出起止时间 |
| 结果 | 原始输出、成功/失败数、计时定义、摘要与比较结论 |

请求总数、并发和发送速率分别记录。固定输出长度的容量实验与允许 EOS 的自然生成实验分开比较。

## 指标定义

- **TTFT，ms**：客户端从发出请求到收到第一个生成 token 的时间。若记录服务端指标，单列其起止点。
- **TPOT，ms/token**：对输出 token 数大于 1 的请求，按 `(E2E - TTFT) / (输出 token 数 - 1)` 计算；明确聚合方法。
- **E2E，ms 或 s**：客户端从发出请求到流完整结束的时间。
- **输出吞吐，token/s**：测量窗口内按既定口径统计的生成 token 数除以窗口时长；明确失败请求的处理。
- **延迟分布**：至少保存均值、P50、P90、P99；样本不足时说明解释范围。
- **内存**：分别记录权重、KV、工作区，以及 allocated/reserved 或设备侧观测的口径。
- **投机指标**：声明接受长度是否含 bonus token，并同时给出 target/draft 工作量。

若报告每卡吞吐，同时报告系统总量和分母；不要混用物理卡、逻辑设备或仅 decode 设备数量。

## NPU 计时

NPU 异步执行下，主机函数返回不代表计算完成。微基准采用验证过的设备事件或明确同步的计时方法；服务基准使用客户端计时，并单独解释服务端拆分。

冷启动、编译、权重加载、预热、通信准备与正式测量分开记录。优化的投入也应包含这些时间，避免反复加载只为重复已知结果。

## 对照与失败分类

先比较本项目参考路径与优化路径。与 SGLang 等外部引擎比较时，对齐模型、设备、负载、缓存和采样设置，并保留未能对齐的差异。

编排检查、解析或结果收集失败与模型、通信、性能问题分别归类。先核对原运行是否完成并保留已有结果；有效环境中的收集错误优先修复收集逻辑。只有能改变结论的实验才安排重跑。

## CI 与硬件回归

普通 CI 运行 CPU 上的调度、缓存状态及小尺寸机制检查；NPU 功能、数值和性能回归运行在受控环境。公开 PR 默认不自动获得实验机器执行权限。

只有对应检查完成后，才将能力标为“已验证”。功能通过、数值通过与性能改善分别陈述。
