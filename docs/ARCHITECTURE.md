# 架构设计

[返回首页](../README.md) · [实施路线](ROADMAP.md)

当前有三种执行方式，共用模型计算与 Step 构造。Runner 可以执行 Qwen2 或纯文本 Qwen3.5；批次通过 mask 隔开完整注意力，通过请求边界隔开 DeltaNet 递推。

| 入口 | 当前行为 | 结果与资源 |
| --- | --- | --- |
| 默认 `generate` | Engine 单请求或 BatchEngine 同步合批 | 本轮完成并提交后，再安排下一轮 |
| `generate --deferred` | BatchEngine 分开发起与收取，最多一批在途 | 可在中间提交或取消请求，等待完成后提交结果 |
| `generate --pipeline` | StoppingDecodePipeline 跨轮接续，最多持有两轮状态 | 下一轮 forward 先提交，再处理上一轮结果；缓存等最后使用完成后归还 |

三种方式都保留清楚的参考实现。下文带课号的段落解释能力怎样逐步加入，不能把某一课的限制当成当前所有入口的限制。第一次阅读可以先跑[A/B/C 演示](WALKTHROUGH.md)。

跨轮循环另有默认关闭的 `early_first_result` Python 实验参数，只把第 0 轮的收取移到第 1 轮发起之前，见[对照与边界](FIRST_RESULT.md)。生成 CLI 仍使用上表的默认顺序。

`collect_before_admit` 则只在队列非空、并发位置已满且有请求已发出长度上限时，先收取上一轮再调度。仍由完成事件保护缓存归还，实验与实际计划见[准入对照](PIPELINE_BENCHMARK.md#等待请求的后续对照)。它也默认关闭，未读到的 EOS 不作为提前收取依据。

## 核心对象

v0.3 的[投机教学实验](SPECULATIVE.md)是独立同步单请求路径：`verify_greedy` 决定输出，`CachedTarget` 管理有效 KV 前缀，以及 Qwen3.5 的线性状态快照与有效输入重算，`generate` 负责提交和回退。新增 `LookupDraft` 只根据已提交 token 查找续文，可返回少于上限甚至零个候选；独立 greedy 对照在其生成结束后运行。它不复用或改变下表的跨轮调度；主文本生成路径仍按前面的三种入口运行。

v0.4 的独立教学入口另有[双模型参考](MODEL_DRAFT.md)：`ModelDraft` 复用 `CachedTarget` 保存独立缓存，下一次候选回调先对齐 target 已提交前缀。target 的接受、提交和回退继续由原 `generate` 完成；新增 `on_commit` 只观察 CPU 可见的输出组。数值检查与计时分别运行，未接入跨轮引擎。

| 对象 | 职责 | 主要状态 |
| --- | --- | --- |
| Request | 维护请求生命周期与输出 | prompt、已输出 token、已计算输入数量、停止原因、错误 |
| Step | 描述一次执行，不承载模型计算 | 输入 token、绝对位置、本轮写入槽位、有序上下文槽位 |
| SlotPool | 管理逻辑槽位归属，不存储张量 | 空闲槽位、请求独占槽位 |
| ModelRunner | 准备输入、执行 forward、选择 token | 模型权重、设备输入、各请求的已计算前缀 |
| TensorKVCache | 存放真实 K/V 张量 | 按 layer / slot / KV head / head dimension 分配 |
| HybridCache | 管理 Qwen3.5 两类设备缓存 | 完整注意力的 KV；每个请求、每个线性层的卷积窗口与 FP32 递推矩阵 |
| Engine | FIFO 调度、准备元数据、提交结果与终止 | 等待队列、一个运行请求、该请求的预留槽位 |
| BatchEngine | 按轮准入与整批结果提交；可选分步执行 | 等待队列、运行请求及各自槽位、请求数与 token 预算、本批计划与结果句柄 |
| DeferredRunner | 提交真实模型计算，读取后再更新主机前缀 | 独占的 ModelRunner、专用流、最多一批在途结果、取消记录 |
| PendingBatch | 持有本批张量与结果拷贝事件，控制读取、取消和放弃 | Step 元组、CPU token 缓冲区、logits、完成与清理状态 |
| StoppingDecodePipeline | 跨轮输入预算、动态准入与退出 | 两轮状态、设备 token、每请求已发出进度和最后使用事件 |

同步与 deferred 调度在 Engine / BatchEngine 内，跨轮调度在 StoppingDecodePipeline 内。Runner 将输入准备、主模型与 greedy 选择分开；执行循环负责提交请求状态。各请求持有独立前缀，确认最后使用结束后再删除相应记录。

一个最小执行循环：

```text
接收请求 → 调度 → 预留缓存 → 准备元数据 → 模型计算
       → 采样 → 提交进度 → 判断停止条件 → 回收资源
```

prefill 计算一段输入；decode 通常消费上一轮选出的 token 并预测下一个。**已输出 token 数与已写入 KV 的 token 数并不总是相同**，应分别记录。chunked prefill、投机解码和异步重叠会进一步要求精确描述这些进度。

## 状态和资源约定

请求状态从 waiting 进入 running，再转为 finished、cancelled 或 failed。状态转换由相应执行循环提交，模型层不直接修改请求队列。

当前准入规则：请求最多需要 `len(prompt) + max_new_tokens - 1` 个槽位；超过总容量时在提交阶段拒绝。运行前一次性预留最大需求，排队请求尚不占用槽位。第一课逐个运行，BatchEngine 在空闲槽位、请求数和本轮 token 预算允许时接纳多个请求，详见[第三课](BATCHING.md)。抢占、重计算及共享前缀的引用计数后续再加入。

Engine 接口只在一个线程中顺序调用。同步路径要求 Runner 在返回或抛错前完成本轮工作；NPU Runner 显式同步。`num_cached_tokens` 保留已成功提交的计算进度，请求终止后它是历史计数。

BatchEngine 的可选分步路径用 `launch_step()` 提交，`finish_step()` 等待、校验并提交。两者之间可以接收请求或取消请求；取消中的运行请求保留槽位，确认本批结束后才清理。等待失败时，即使请求已经标成 failed，也继续持有资源并拒绝下一批；`abort_step()` 只有在排空设备工作成功后才能清理。详见[第八课](DEFERRED_ENGINE.md)。

第九课的 `text_loop` 每轮只读取并准备一条新文本，可在本批结果收取前完成分词和入队；请求仍由下一轮的同一调度器准入。它不提前构造依赖当前结果的 decode，也不增加线程。若新文本准备失败，先收取已经发出的本批结果，再把前端错误交还调用方；设备等待失败则继续保留资源。实际重叠和计时见[第九课](PREPARE_DURING_STEP.md)。

第十课进一步把不依赖 logits 的设备采样索引放到主模型执行前准备，继续由 PendingBatch 持有到使用结束。中间 prompt 块不产生索引；索引准备失败时还没有进入模型缓存写入阶段。旧、新实现在同一模型上数值与状态一致，采样步骤的主机等待边界随位置改变；整轮性能结论见[第十课](SAMPLER_INPUTS.md)。

当前与后续缓存实现需要满足以下条件，其中在途计算、投机与共享规则仅适用于相应能力加入后：

1. 活跃请求拥有的块与空闲块不重叠。
2. attention 只读取已写入且在当前因果范围内的缓存。
3. 已取消请求的在途计算结束后，相关块才能再次分配。
4. 投机 token 的暂存、接受与回退不会破坏已提交前缀。
5. 若未来支持前缀共享，共享块须有独立引用计数与写入规则。

分页 KV 管理与融合 paged attention 是两个可分步实现的功能。先通过页表 gather 得到参考计算输入，再接入可直接读取分页布局的优化算子。

## Ascend 执行后端

当前后端使用基础 PyTorch/torch_npu 运算，设备选择与同步在 ModelRunner，stream/event 在 deferred 和跨轮执行路径。图执行和融合算子仍是候选，尚无统一的优化后端。

参考路径用基础运算表达数学含义；优化路径逐项替换归一化、attention、专家计算或缓存写入。每个算子记录输入形状、布局、dtype、版本和数值对照。

当前跨轮文本入口使用 FP32 与 eager，BF16 的已有范围见[对照说明](BF16_REFERENCE.md)。第五、六课解释状态与事件；[第七课](DEFERRED_RUNNER.md)把它们接到真实模型，[第八课](DEFERRED_ENGINE.md)接入 BatchEngine，单批限制属于这条 deferred 路径。第十一至十四课另行实现跨轮循环，v0.2 已通过 `--pipeline` 接入文本生成；后续[短负载对照](PIPELINE_BENCHMARK.md)未显示整批提速，不能由执行顺序推定设备重叠收益。

[第十一课](DECODE_PIPELINE.md)单独实现固定批次的设备 token 接续：下一轮 forward 在上一轮 CPU 输出处理前提交，最多保留两轮状态，整批排空才回收。它独占空闲 Runner，不接入默认生成路径；动态成员、EOS 和取消仍由后续实验处理，没有性能结论。

[第十二课](STOPPING_PIPELINE.md)在同一循环上增加成员退出：EOS/取消阻止后续选择，已经提交的那一轮输出丢弃，最后使用事件完成后回收请求缓存。本课只检查结果回调内的单线程取消和退出，主引擎的单批限制保留。

[第十三课](ARRIVALS.md)继续扩展这个类，支持结果回调中的新请求提交、FIFO 准入和混合输入。旧请求读设备 token，新请求上传 prompt，完成事件确认后才复用槽位；按请求记录已提交计算次数，允许新请求延长整体运行。暂不做 token 预算或分块 prefill，尚未并入主 BatchEngine。

## 模型层

当前模型层实现 Qwen2 和纯文本 Qwen3.5。DeepSeek-V2-Lite 的 MLA、MoE 与权重映射尚未实现；若后续加入，会单独处理其数学与缓存布局。模型接口共享适用的执行元数据，不以过度统一掩盖模型差异。

候选的 MLA 教学可分为展开计算、压缩缓存和优化计算；MoE 可从路由结果与逐专家参考计算开始。是否开展及实施顺序见[路线](ROADMAP.md)。

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
  npu_events.py   # 未记录事件保护 / 等待结果拷贝 / 持有缓冲区
  npu_step_demo.py # 微型张量的实际 NPU 跨流与回收实验
  deferred_runner.py # 单批在途模型执行 / 显式读取与提交
  verify_deferred.py # 同步与延后提交的固定轨迹对照
  verify_deferred_engine.py # 同步/分步引擎的请求生命周期对照
  decode_pipeline.py # 固定成员与长度 / 设备token接续 / 延后一轮处理输出
  verify_decode_pipeline.py # 同步greedy与固定跨轮执行的数值对照
  stopping_pipeline.py # EOS/回调取消 / 缩批 / 等最后使用事件再回收
  verify_stopping_pipeline.py # 成员退出与已提交后继轮次的数值/资源检查
  verify_arrivals.py # 新prompt/旧设备token合批、等待与槽位复用检查
  planning.py # 共用Step构造 / 不可变的设备已发出进度
  verify_chunk_pipeline.py # 跨轮输入预算、分块与整段prompt参考对照
  text_loop.py    # 逐条准备文本 / 在收取前或收取后入队
  benchmark_prepare.py # 相同负载下比较文本准备的位置
  analyze_prepare_trace.py # CPU分词与设备kernel区间求交
  model_runner.py # 输入准备 / forward / greedy
  tensor_cache.py # 真实 K/V 张量与槽位读写
  hybrid_cache.py # 完整注意力 KV 与请求独占的线性状态
  models/qwen2.py # Qwen2 模型与 safetensors 加载
  models/qwen3_5.py # 纯文本混合注意力与 checkpoint 映射
  generate.py     # 离线真实生成入口 / 同步、跨轮或双模型方式选择
  speculative_text.py # 用户文本 / EOS / 独立 greedy 对照，复用投机参考
  pipeline_text.py # 文本编码 / 单线程回调到达 / 提交前观察
  verify_pipeline_text.py # 共享生成前端的同步、跨轮与EOS对照
  verify.py       # 真实权重 logits 对照
  verify_batch.py # 批次与独立请求的数值对照
tests/
  test_engine.py
  test_model.py
  test_batch_engine.py
  test_batch_model.py
```

逐步阅读见[第一课](FIRST_REQUEST.md)和[第二课](REAL_GENERATION.md)。目前通过基础 PyTorch/torch_npu 运算执行，融合算子后端在有具体需求时加入。

[第十四课](CHUNK_PIPELINE.md)让同步 BatchEngine 和跨轮实验共用 `plan_step()`；调用者明确提供 CPU 已提交或设备已发出的起点。跨轮循环的 `IssuedProgress` 规划输入、`Request` 保存 CPU 提交历史、每请求最后使用事件决定回收。中间 prompt 块不采样，输入预算先服务 decode，再服务已有 prompt 和 FIFO 新请求；两套调度及主引擎单批限制仍分别保留。

## 与 DeepInfer 的对应关系

固定参考提交、具体文件与许可证记录见[参考与来源](REFERENCES.md)。调度器初期用 Python 表达；调度、执行、设备缓存及步骤元数据的边界借鉴原版设计。

独立实验与主引擎通过同一套数值和状态约定连接。只有机制验证与模型兼容性均完成后，才将实验能力标记为引擎功能。

main 的[双模型文本入口](SPECULATIVE_TEXT.md)复用 `ModelDraft` 与 `speculative.generate()`，不经过同步/跨轮批调度器。文本编码、流式解码与可选的独立 greedy 对照放在接入层；候选验证、状态回退和槽位归还仍由原有执行代码负责。取消在 prefill 前和提交回调后检查，不打断已开始的一轮；已提交的组完整交付后再清理，EOS 与长度结束优先。
