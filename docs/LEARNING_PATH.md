# 十四课：沿着一个请求读代码

下面十四课记录主执行路径的演进。[v0.3 的三步投机路线](RELEASE_v0.3.md)在主线之外继续研究候选验证、Qwen3.5 状态恢复，以及不读取答案的 lookup draft。

[返回首页](../README.md) · [先运行](START_HERE.md) · [源码映射](SOURCE_MAP.md)

先跑[安装指南](START_HERE.md)里的例子，再按感兴趣的问题读。前三课从单请求走到批次调度；第五课开始区分函数返回与设备完成，后面再把这些依赖接到真实模型。

想先看一次完整运行，可以从[A、B、C 的调度演示](WALKTHROUGH.md)开始；想换自己的文本，接着看[生成入口与逐轮观察](PIPELINE_TEXT.md)。它们复用第十二至十四课的循环；十四课仍按原来的实现阶段展开。

读完执行顺序后，可以运行[同步与跨轮对照](PIPELINE_BENCHMARK.md)，看看为什么“下一轮提前发出”不一定更快。这是现有路径的测量补充，不另开一套执行器。

接着看[首轮结果实验](FIRST_RESULT.md)：只改第 0 轮的收取位置，在相同批次计划下观察首 token 与整批耗时怎样变化。

再看同一份基准中的[等待请求对照](PIPELINE_BENCHMARK.md#等待请求的后续对照)：等待结果会耽误准入，但先等待也有成本。沿着两份实际计划，检查少了一次 forward 是否就一定更快。

继续研究投机时，v0.4 的[双模型参考](MODEL_DRAFT.md)让小模型真正产生候选，复用前一阶段的验证与回退；读者可以比较高接受率与实际计算、复制和延迟。先从[逐轮演示](MODEL_DRAFT_WALKTHROUGH.md)观察拒绝和恢复；它是独立教学入口，十四课的编号与历史阶段不变。

## 从同步请求到真实模型

| 章节 | 要回答的问题 | 可运行入口 |
| --- | --- | --- |
| [一：跟踪一个请求](FIRST_REQUEST.md) | 调度、prefill、decode、停止和回收怎样连起来？ | `python -m minideepinfer`，无依赖 |
| [二：真实权重与 KV Cache](REAL_GENERATION.md) | 整数 Runner 换成真实模型后，多了哪些状态？ | `python -m minideepinfer.generate --help`，运行需权重 |
| [三：同步批处理](BATCHING.md) | 多请求怎样共享一轮计算，又互不串缓存？ | `python -m minideepinfer.batch_demo`，无依赖 |
| [四：Qwen3.5 的两种缓存](QWEN35.md) | 完整注意力与 DeltaNet 分别记住什么？ | Qwen3.5-2B 生成；0.8B 开发对照 |

## 计算发出后，谁持有结果

| 章节 | 要回答的问题 | 可运行入口 |
| --- | --- | --- |
| [五：请求结束了，缓存为什么还不能还？](STEP_STATE.md) | 行号改变和取消怎样影响在途使用者？ | `python -m minideepinfer.step_demo`，手动事件 |
| [六：NPU 算完了，CPU 就能读吗？](NPU_EVENTS.md) | 流、拷贝与完成事件怎样形成读取边界？ | `python -m minideepinfer.npu_step_demo`，NPU 小张量 |
| [七：结果回来后，再提交进度](DEFERRED_RUNNER.md) | 如何把执行与主机提交分开？ | `python -m minideepinfer.verify_deferred --device cpu` |
| [八：BatchEngine 分步执行](DEFERRED_ENGINE.md) | 计算途中取消，怎样接回原引擎？ | `python -m minideepinfer.verify_deferred_engine --device cpu` |

## 顺序允许重叠，是否真的更快

| 章节 | 要回答的问题 | 证据 |
| --- | --- | --- |
| [九：等待期间准备新文本](PREPARE_DURING_STEP.md) | 分词移到等待前，设备有没有同时工作？ | 相同负载计时与三步 trace；短负载未显示提速 |
| [十：先准备采样索引，再计算](SAMPLER_INPUTS.md) | 哪些准备依赖 logits，哪些不依赖？ | 旧新数值对照与时间线；没有稳定整体提速结论 |

## 从固定批次走到跨轮分块

| 章节 | 要回答的问题 | CPU 小模型入口 |
| --- | --- | --- |
| [十一：下一轮先发出去](DECODE_PIPELINE.md) | CPU 未读取 token，设备怎样继续计算？ | `python -m minideepinfer.verify_decode_pipeline --device cpu` |
| [十二：请求结束，下一轮已经发出去了](STOPPING_PIPELINE.md) | EOS 或取消后，多算的一轮怎样收尾？ | `python -m minideepinfer.verify_stopping_pipeline --device cpu` |
| [十三：新请求来了，放进哪一轮？](ARRIVALS.md) | 新 prompt 怎样与旧设备 token 合批并复用槽位？ | `python -m minideepinfer.verify_arrivals --device cpu` |
| [十四：发出去了，不等于已经提交了](CHUNK_PIPELINE.md) | token 预算怎样切 prompt，位置该按哪份进度规划？ | `python -m minideepinfer.verify_chunk_pipeline --device cpu` |

这些 `verify_*` 入口默认使用随机小模型，不下载权重；真实 Ascend 对照增加 `--model weights/Qwen3.5-2B --device npu:0`。模型检查需要 PyTorch，真实权重还需 tokenizer 与 safetensors。计时、profiling 和固定数值检查的用途不同，不必为了读教程把所有实机实验重跑一遍。

读到后几课时，可以回看两个公共入口：[`plan_step()`](../minideepinfer/planning.py)负责构造同步和跨轮使用的 Step；[`_launch()` / `_sample()`](../minideepinfer/decode_pipeline.py)负责固定批次与动态实验的设备输入和采样。两种调度循环仍分别保留，因为跨轮执行必须额外记住已发出的工作和最后使用事件，不能只看 CPU 已提交的进度。

每课记录自己的实现阶段与实测条件；正式 v0.4 的入口见[运行指南](START_HERE.md)与[版本说明](RELEASE_v0.4.md)，main 新增的用户文本投机入口见[双模型文本生成](SPECULATIVE_TEXT.md)。DeepInfer 固定参考及教学简化见[源码映射](SOURCE_MAP.md)，后续方向见[路线](ROADMAP.md)。

理解 greedy 投机后，可以接着运行 [CPU 随机采样](SAMPLING.md)：用小词表看接受和补采样为什么需要概率，再用两个随机模型检查同一套缓存回退。无需权重，不计入前十四课。
