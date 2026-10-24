<div align="center">

# mini-DeepInfer

**Follow a real request through a small inference engine inspired by DeepInfer.**

Built for learning, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

[简体中文](README.md) · **English**

[Read the code](#follow-a-request-through-the-code) · [Run a model](#run-a-real-model) · [Validation](docs/VALIDATION.md) · [Roadmap](docs/ROADMAP.md)

</div>

mini-DeepInfer studies the design of [DeepInfer](https://github.com/deepseek-ai/DeepInfer) through a small teaching engine: how inputs become tokens, how prefill leads into decode, where the KV cache lives, and who releases it when a request ends.

The project now runs **text-only Qwen3.5-2B on Ascend 910C** using its own forward pass and cache management. Qwen3.5-0.8B is the smaller development model; DeepSeek-R1-Distill-Qwen-1.5B remains a regression baseline. Full attention stores token KV, while DeltaNet keeps convolution and recurrent state for each request.

The main example uses FP32. BF16 also passes fixed short trajectories against an independent recurrent reference with FP32 state storage; native chunk-path and cross-schedule generation differences are documented in the [reference notes](docs/BF16_REFERENCE.md). The engine supports synchronous dynamic batching, chunked prefill, cancellation, and cache reclamation.

## What runs today

| Check | Result |
| --- | --- |
| Qwen3.5-0.8B / Ascend 910C FP32 | Chunked prefill + 4 fixed-token decode steps; maximum absolute difference about `2.00e-5` |
| Qwen3.5-2B / Ascend 910C FP32 | Single-request and two-request numerical checks pass; maximum absolute difference about `2.29e-5`, with top-1 agreement at every checked position |
| Qwen3.5 BF16 recurrent checks | Fixed short trajectories for 0.8B, 2B, and a two-request 2B batch have maximum difference **0**, with zero tolerances and matched scan/state semantics |
| 2B, two concurrent requests | Chinese explanation and English answer both stop at EOS; 512/512 logical slots reclaimed |
| Ascend 910C event experiment | Cross-stream reads, host result copies, and deferred release pass on tiny tensors; [lesson six](docs/NPU_EVENTS.md) preserves the actual output |
| 2B deferred-result experiment | Six real-weight FP32 batches match synchronous Runner logits with zero difference, covering chunks, cancellation, and host-error cleanup; see [lesson seven](docs/DEFERRED_RUNNER.md) |

The English answer from the real 2B run:

```text
1 plus 1 equals 2.

[eos; 9 output tokens; free KV slots 512/512]
```

The [complete bilingual output](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt), inputs, environment, errors, and failed BF16 checks are preserved in the [validation record](docs/VALIDATION.md). These are short-context correctness checks, without throughput or latency claims.

## Start with a small example

No dependencies, weights, or NPU are needed for this CPU simulation:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

Follow requests A, B, and C: B changes batch rows but keeps its own result; A is cancelled but its slots stay protected until all users finish. [Lesson five](docs/STEP_STATE.md) explains the code in Chinese. The demo prints Chinese explanations and advances events by hand; the real-model engine remains synchronous.

With an Ascend environment, [lesson six](docs/NPU_EVENTS.md) checks those boundaries using real NPU events and tiny tensors, without loading weights.

[Lesson seven](docs/DEFERRED_RUNNER.md) applies the same ordering to model execution: launch computation, then commit progress after reading the result. It also includes a tiny PyTorch model requiring no weights. Only one batch may be in flight.

## Run a real model

Install the model dependencies from the repository directory:

```bash
python -m pip install -e ".[qwen35]"
```

Download the pinned checkpoints using [lesson four](docs/QWEN35.md). In an Ascend environment with compatible PyTorch, torch_npu, and CANN:

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

Without an NPU, use `--device cpu --dtype float32` and start with 0.8B. This implementation targets short teaching examples; CPU execution still loads the full text model.

## Follow a request through the code

If you know basic Python and PyTorch, follow these entry points. The project implements the model computation; Transformers supplies the tokenizer and numerical reference.

| Question | Read |
| --- | --- |
| What does each prefill or decode step consume? | [`Engine.step()`](minideepinfer/engine.py) |
| How do tokens, positions, and KV slots line up? | [`ModelRunner.prepare_input()`](minideepinfer/model_runner.py) |
| How does attention write and read past K/V? | [`qwen2.py`](minideepinfer/models/qwen2.py) · [`TensorKVCache`](minideepinfer/tensor_cache.py) |
| How does linear attention remember history without mixing requests? | [`qwen3_5.py`](minideepinfer/models/qwen3_5.py) · [`HybridCache`](minideepinfer/hybrid_cache.py) |
| How do logits become the next token? | [`ModelRunner.execute_sample()`](minideepinfer/model_runner.py) |
| Who releases slots on completion or cancellation? | [`Engine`](minideepinfer/engine.py) · [`SlotPool`](minideepinfer/kv_cache.py) |

New to inference engines? [Lesson one](docs/FIRST_REQUEST.md) uses an integer runner so every input, position, and slot can be printed. It needs only Python 3.10+, without model dependencies or weights:

```bash
python -m minideepinfer
python -m unittest discover -s tests -v
```

Without model dependencies, 30 engine and state tests run and model comparisons are explicitly skipped. [Lesson two](docs/REAL_GENERATION.md) replaces the runner with a real model. Detailed lessons and design documents are currently in Chinese; code comments are in English.

## How this relates to DeepInfer

DeepInfer is the primary technical reference. The current implementation separates input preparation, main-model execution, sampling, and result handling. Lessons five through seven move from integer events to real NPU operations and model execution, explaining state ownership, batch remapping, and result commits. Scheduling overlap still needs separate validation. The [source map](docs/SOURCE_MAP.md) identifies the pinned upstream revision, corresponding entry points, and teaching simplifications.

**Small Qwen models carry the teaching examples; DeepInfer remains the engine-design reference.** Single-request and synchronous batch paths remain available for comparison. [Lesson four](docs/QWEN35.md) explains hybrid model state management. MLA, MoE, speculative decoding, and bounded replay belong to later, separate lessons; see the [roadmap](docs/ROADMAP.md).

[mini-sglang](https://github.com/sgl-project/mini-sglang) informs the compact teaching format; Ascend SGLang informs NPU API and compatibility work. See the [architecture](docs/ARCHITECTURE.md) and [references](docs/REFERENCES.md) for further details.

## Contribute

Bring a concrete question: a step that is hard to follow, a command that fails, or a cache behavior that needs an example. [Open an issue](https://github.com/AkiYangDev/mini-deepinfer/issues), or improve the code and lessons through the [contribution guide](CONTRIBUTING.md).

If this learning path is useful to you, star the project to follow the next implementation and experiment.

This is an independent community project, with no official affiliation with DeepSeek, SGLang, or Huawei. Original code and documentation use [Apache-2.0](LICENSE); models and third-party code retain their respective licenses.
