<div align="center">

# mini-DeepInfer

**Understand DeepSeek inference. Build it on Ascend 910C.**

A readable inference engine, built step by step for learning and experimentation.

[![Platform: Ascend 910C target](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/ROADMAP.md) [![Status: Runnable skeleton](https://img.shields.io/badge/Status-Runnable%20skeleton-f59e0b)](docs/FIRST_REQUEST.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

[简体中文](README.md) · **English**

[Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md) · [Validation](docs/VALIDATION.md) · [Contributing](CONTRIBUTING.md)

</div>

mini-DeepInfer is an educational inference engine under development for **Ascend 910C**, inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer). The goal is to make inference understandable through code, diagrams, and experiments: from the first generated token to KV caches, continuous batching, MLA, and MoE.

Following the learning approach of [mini-sglang](https://github.com/sgl-project/mini-sglang), we aim to make the core execution path small enough to follow from request to output, then extend it one optimization at a time and measure the effect on latency, throughput, and memory.

**DeepInfer is the primary engine design reference.** Engine responsibilities, execution metadata, result handling, and future in-flight state mechanisms are studied against its source. mini-sglang informs the compact teaching format; Ascend SGLang informs NPU API and compatibility work. The [source map](docs/SOURCE_MAP.md#deepinfer-主线与当前骨架) distinguishes conceptual mappings, teaching simplifications, and mechanisms not yet implemented.

> **Status: single-request model implementation.** An independent DeepSeek-R1-Distill-Qwen-1.5B forward pass, tensor KV cache, and offline generation are implemented, with real-weight CPU numerical checks passing. Full-model validation on Ascend 910C is pending. The integer runner remains the first lesson; the [second lesson](docs/REAL_GENERATION.md) covers real weights.

## Run your first request

Python 3.10+. The demo and tests use only the standard library; no weights to download:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer
python -m unittest discover -s tests -v
```

Watch two requests execute in FIFO order:

```text
req-0 prefill input=(10, 11) positions=(0, 1) write_slots=(0, 1)
  -> token=12, cached_inputs=2, stop=-
req-0 decode  input=(12,) positions=(2,) write_slots=(2,)
  -> token=13, cached_inputs=3, stop=-
req-0 decode  input=(13,) positions=(3,) write_slots=(3,)
  -> token=14, cached_inputs=4, stop=length
...
Free slots: 8/8
```

Why is token `14` generated but absent from KV? Why can the next request reuse slot `0`? Follow [`demo.py`](minideepinfer/demo.py) → [`Engine.step()`](minideepinfer/engine.py) → [`SlotPool`](minideepinfer/kv_cache.py). The [first lesson](docs/FIRST_REQUEST.md) walks through the same trace in Chinese; code comments are in English.

## Run a real model

Install in an environment with matching PyTorch / `torch_npu` versions and prepare a local checkpoint:

```bash
python -m pip install -e ".[model]"
python -m minideepinfer.generate \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device npu:0 --dtype bfloat16 --max-new-tokens 32
```

[`qwen2.py`](minideepinfer/models/qwen2.py) implements RMSNorm, RoPE, GQA attention, and the MLP with basic operations. [`ModelRunner`](minideepinfer/model_runner.py) follows DeepInfer's preparation, forward, and sampling responsibilities; the engine commits results. Transformers provides the tokenizer and numerical reference.

See the [second lesson](docs/REAL_GENERATION.md) for scope, dependencies, and logits checks. This is a short-context teaching implementation; dynamic batching and scheduling overlap are still planned.

## From a forward pass to an inference engine

Generating text is the starting point. Serving requests of different lengths adds a new set of decisions: what to compute next, where to keep cached states, and when to return results while the device continues working.

mini-DeepInfer aims to make those decisions visible in the implementation:

- **Follow the request.** Trace scheduling, cache management, model execution, and output handling through a clear execution loop.
- **Explore DeepSeek architectures.** Progress from basic generation to MLA and MoE, then study selected DeepInfer mechanisms.
- **Learn Ascend execution.** Understand how `torch_npu`, fused operators, streams, events, and graph execution affect a real workload.

Every step follows the same sequence: **understand the computation → verify correctness → introduce an optimization → measure the change.**

## Planned mechanisms

| Topic | From concept to implementation |
| --- | --- |
| **Autoregressive generation** | Weight loading, prefill, token-by-token decode, sampling, and stopping conditions |
| **KV caches** | Contiguous and paged storage, position mapping, allocation, reclamation, and request isolation |
| **Batching and scheduling** | Continuous batching, chunked prefill, token budgets, and request lifecycles |
| **DeepSeek architectures** | MLA compressed caches, MoE routing, expert execution, and output combination |
| **Execution and advanced optimization** | Graph execution, scheduling overlap, single-node parallelism, and focused speculative decoding and bounded replay labs |

The design keeps reference and optimized paths available side by side, so readers can understand the math and compare the effect of a fused operator or execution strategy. Scope and completion criteria are in the [roadmap](docs/ROADMAP.md).

## One request through the system

This is the target execution structure. The scheduler selects the tokens for a step, the runner executes the model, and committed results advance the next step.

```mermaid
flowchart LR
    A[Request] --> B[Scheduler]
    B --> C[Model Runner]
    C --> D[Sample and Commit]
    D --> E[Streaming Output]
    D --> B
    B <--> K[KV Cache]
    C <--> K
    C --> N[Ascend Backend]
```

Start with synchronous execution, then add paged caches, changing batch membership, and scheduling overlap. Each step retains a reference for explaining and checking its behavior. See the [architecture](docs/ARCHITECTURE.md) for the design boundaries.

## From small models to DeepSeek

The learning path has three stages, each with a specific role:

1. **Generate the first token.** Use DeepSeek-R1-Distill-Qwen-1.5B and its Qwen architecture to implement real weight loading, a KV cache, and the generation loop.
2. **Explore MLA and MoE.** Use DeepSeek-V2-Lite / V2-Lite-Chat to study compressed caches and expert routing, with numerical checks against the corresponding model.
3. **Study DeepInfer mechanisms.** Build focused experiments for scheduling overlap, speculative state commit/rollback, and SWA bounded replay, tied to a pinned upstream revision.

Each lesson will identify its model and architectural assumptions. Full V4.1 deployment is a later research milestone. Model choices and deliverables are described in the [roadmap](docs/ROADMAP.md).

## Start here

If you know basic PyTorch and want to understand inference systems, start by following a single execution step:

| Read | What you will find |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | How requests, the scheduler, the runner, and caches fit together |
| [Validation](docs/VALIDATION.md) | How to compare logits, catch cache errors, and measure TTFT, TPOT, and throughput |
| [Roadmap](docs/ROADMAP.md) | Deliverables from single-request generation to DeepSeek architectures |
| [References](docs/REFERENCES.md) | Connections to DeepInfer and mini-sglang, including simplifications and attribution |

Detailed design documents are currently in Chinese. Next comes synchronous dynamic batching, including request admission, completion, and cache reuse, followed by a focused study of DeepInfer's in-flight states and scheduling overlap. Each stage retains a runnable example and numerical checks.

## Build and learn with us

Contribute a clearer diagram, a more readable implementation, or a reproducible Ascend experiment. Design discussions and documentation improvements are welcome too. See the [contribution guide](CONTRIBUTING.md).

Interested in following the build? **Star the repository to bookmark it and follow future code and lessons.**

## Acknowledgments and license

Thanks to [DeepInfer](https://github.com/deepseek-ai/DeepInfer), [mini-sglang](https://github.com/sgl-project/mini-sglang), and [SGLang](https://github.com/sgl-project/sglang) for their open work.

This is an independent community project, unaffiliated with DeepSeek, SGLang, or Huawei. Original code and documentation are licensed under [Apache-2.0](LICENSE). Models and third-party components retain their own licenses; see [References](docs/REFERENCES.md) for attribution.
