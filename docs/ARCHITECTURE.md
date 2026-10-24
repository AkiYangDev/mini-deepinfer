# 架构设计

[返回首页](../README.md) · [实施路线](ROADMAP.md)

第一课保留同步 FIFO 单请求 Engine，第三课增加按轮调度的 BatchEngine。Runner 可以执行 Qwen2，或第四课的纯文本 Qwen3.5。批次通过 mask 隔开完整注意力，通过请求边界隔开 DeltaNet 递推，仍等待本轮计算结束再提交。

## 核心对象

| 对象 | 职责 | 主要状态 |
| --- | --- | --- |
| Request | 维护请求生命周期与输出 | prompt、已输出 token、已计算输入数量、停止原因、错误 |
| Step | 描述一次执行，不承载模型计算 | 输入 token、绝对位置、本轮写入槽位、有序上下文槽位 |
| SlotPool | 管理逻辑槽位归属，不存储张量 | 空闲槽位、请求独占槽位 |
| ModelRunner | 准备输入、执行 forward、选择 token | 模型权重、设备输入、各请求的已计算前缀 |
| TensorKVCache | 存放真实 K/V 张量 | 按 layer / slot / KV head / head dimension 分配 |
| HybridCache | 管理 Qwen3.5 两类设备缓存 | 完整注意力的 KV；每个请求、每个线性层的卷积窗口与 FP32 递推矩阵 |
| Engine | FIFO 调度、准备元数据、提交结果与终止 | 等待队列、一个运行请求、该请求的预留槽位 |
| BatchEngine | 按轮准入与整批结果提交 | 等待队列、运行请求及各自槽位、请求数与 token 预算 |

调度在 Engine / BatchEngine 内。Runner 将输入准备、主模型与 greedy 选择分开；批次计算后按照同样的请求顺序返回 token，Engine 才提交请求状态。批次版本维护每个请求独立的已计算前缀，并在请求结束后删除对应记录。

一个最小执行循环：

```text
接收请求 → 调度 → 预留缓存 → 准备元数据 → 模型计算
       → 采样 → 提交进度 → 判断停止条件 → 回收资源
```

prefill 计算一段输入；decode 通常消费上一轮选出的 token 并预测下一个。**已输出 token 数与已写入 KV 的 token 数并不总是相同**，应分别记录。chunked prefill、投机解码和异步重叠会进一步要求精确描述这些进度。

## 状态和资源约定

请求状态从 waiting 进入 running，再转为 finished、cancelled 或 failed。状态转换由 Engine 统一提交，模型层不直接修改请求队列。

当前准入规则：请求最多需要 `len(prompt) + max_new_tokens - 1` 个槽位；超过总容量时在提交阶段拒绝。运行前一次性预留最大需求，排队请求尚不占用槽位。第一课逐个运行，BatchEngine 在空闲槽位、请求数和本轮 token 预算允许时接纳多个请求，详见[第三课](BATCHING.md)。抢占、重计算及共享前缀的引用计数后续再加入。

Engine 接口只在一个线程中顺序调用。`Runner.run()` 必须在返回或抛错前完成本轮工作，Engine 才能立即释放槽位；NPU Runner 显式同步。`num_cached_tokens` 保留已成功提交的计算进度，请求终止后它是历史计数。当前没有异步执行或在途引用。

当前与后续缓存实现需要满足以下条件，其中在途计算、投机与共享规则仅适用于相应能力加入后：

1. 活跃请求拥有的块与空闲块不重叠。
2. attention 只读取已写入且在当前因果范围内的缓存。
3. 已取消请求的在途计算结束后，相关块才能再次分配。
4. 投机 token 的暂存、接受与回退不会破坏已提交前缀。
5. 若未来支持前缀共享，共享块须有独立引用计数与写入规则。

分页 KV 管理与融合 paged attention 是两个可分步实现的功能。先通过页表 gather 得到参考计算输入，再接入可直接读取分页布局的优化算子。

## Ascend 执行后端

底层接口集中管理设备选择、内存、stream/event、同步、图执行和优化算子。模型与调度代码使用项目自己的明确接口。

参考路径用基础运算表达数学含义；优化路径逐项替换归一化、attention、专家计算或缓存写入。每个算子记录输入形状、布局、dtype、版本和数值对照。

当前真实示例使用 FP32 与 eager，BF16 的已验证范围见[对照说明](BF16_REFERENCE.md)。[第五课](STEP_STATE.md)先用独立 CPU 实验解释在途步骤状态、批次映射与延迟释放；NPU 事件、真实执行重叠和图执行分别验证后再接入。需要特别检查上一轮输出尚未被主机读取时，下一轮批次如何更新，以及取消请求如何安全退出。

## 模型层

入门模型实现 Qwen 架构；V2-Lite 单独实现对应的 MLA、MoE 与权重映射。模型接口尽量共享执行元数据和输出形式，不以过度统一来掩盖缓存布局差异。

MLA 教学分为展开计算、压缩缓存和优化计算三个步骤。MoE 从路由结果与逐专家参考计算开始，再引入 token 分组、融合计算及通信。

## 当前目录

第一课执行 `python -m minideepinfer`，真实生成执行 `python -m minideepinfer.generate`：

```text
minideepinfer/
  __init__.py
  __main__.py
  engine.py       # Request / Step / Runner / Engine
  batch_engine.py # 按轮调度 / 预算 / 整批提交
  kv_cache.py     # 逻辑槽位分配与回收
  demo.py         # 整数 Runner 与可观察的执行轨迹
  batch_demo.py   # 批次从 [A, B] 变成 [B, C]
  step_state.py   # 独立实验：按请求映射结果 / 等使用者完成后释放
  step_demo.py    # 手动事件：提前读取与提前回收为什么不行
  model_runner.py # 输入准备 / forward / greedy
  tensor_cache.py # 真实 K/V 张量与槽位读写
  hybrid_cache.py # 完整注意力 KV 与请求独占的线性状态
  models/qwen2.py # Qwen2 模型与 safetensors 加载
  models/qwen3_5.py # 纯文本混合注意力与 checkpoint 映射
  generate.py     # 离线真实生成入口
  verify.py       # 真实权重 logits 对照
  verify_batch.py # 批次与独立请求的数值对照
tests/
  test_engine.py
  test_model.py
  test_batch_engine.py
  test_batch_model.py
```

逐步阅读见[第一课](FIRST_REQUEST.md)和[第二课](REAL_GENERATION.md)。目前通过基础 PyTorch/torch_npu 运算执行，融合算子后端在有具体需求时加入。

## 与 DeepInfer 的对应关系

固定参考提交、具体文件与许可证记录见[参考与来源](REFERENCES.md)。调度器初期用 Python 表达；调度、执行、设备缓存及步骤元数据的边界借鉴原版设计。

独立实验与主引擎通过同一套数值和状态约定连接。只有机制验证与模型兼容性均完成后，才将实验能力标记为引擎功能。
