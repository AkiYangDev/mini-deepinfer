<div align="center">

# mini-DeepInfer

**Follow a real request through a small inference engine inspired by DeepInfer.**

Built for learning, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

[简体中文](README.md) · **English**

[Setup](docs/START_HERE.md) · [Fourteen lessons](docs/LEARNING_PATH.md) · [v0.1 notes](docs/RELEASE_v0.1.md) · [Validation](docs/VALIDATION.md)

</div>

mini-DeepInfer studies the design of [DeepInfer](https://github.com/deepseek-ai/DeepInfer) through a small teaching engine: how prompts are split into chunks, how decode consumes the previous device token, who owns the KV and linear-attention state, and why a finished request cannot always release its cache immediately.

The project runs **text-only Qwen3.5-2B on Ascend 910C** using its own forward pass and cache management. Fourteen lessons follow the same request path, from dependency-free integer examples to tiny PyTorch models and real checkpoints. Qwen3.5-0.8B is the development model; DeepSeek-R1-Distill-Qwen-1.5B remains a regression baseline.

## Start small

With Python 3.10+, run an example that needs no model dependencies, weights, or NPU:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m pip install -e .
python -m minideepinfer.step_demo
```

Why are A's cache slots still held after cancellation? How does B find its previous result after changing batch rows? The demo prints state transitions and advances events by hand. It does not execute a model; its explanations are in Chinese.

With PyTorch installed, run a tiny hybrid-attention model without downloading weights:

```bash
python -m minideepinfer.verify_chunk_pipeline --device cpu
```

It mixes decode with prompt chunks, admits a later request, and compares the results against synchronous whole-prompt execution. The report contains batch plans, errors, and cache reclamation. See the [setup guide](docs/START_HERE.md) for dependency choices.

## Run a real model on Ascend 910C

Use an environment with matching PyTorch, torch_npu, and CANN. Install the Python dependencies and download the pinned Qwen3.5-2B checkpoint as described in [setup](docs/START_HERE.md). From the repository directory:

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

The English answer from the real 2B run:

```text
1 plus 1 equals 2.

[eos; 9 output tokens; free KV slots 512/512]
```

The [complete bilingual output](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt) is preserved. The main path uses FP32, eager execution, and greedy sampling for short teaching examples. Without an NPU, use `--device cpu` and start with the 0.8B checkpoint; CPU execution still loads the full text model.

## Follow the code

| Question | Read |
| --- | --- |
| What does each prefill or decode step consume? | [`Engine.step()`](minideepinfer/engine.py) · [lesson one](docs/FIRST_REQUEST.md) |
| How are requests batched while keeping positions and cache separate? | [`BatchEngine`](minideepinfer/batch_engine.py) · [lesson three](docs/BATCHING.md) |
| Why does Qwen3.5 need two kinds of cache? | [`HybridCache`](minideepinfer/hybrid_cache.py) · [lesson four](docs/QWEN35.md) |
| How is an in-flight cancellation drained? | [`DeferredRunner`](minideepinfer/deferred_runner.py) · [lessons seven and eight](docs/DEFERRED_ENGINE.md) |
| How can the next forward use a token before the CPU reads it? | [`decode_pipeline.py`](minideepinfer/decode_pipeline.py) · [lesson eleven](docs/DECODE_PIPELINE.md) |
| How do stops, arrivals, and chunks share one loop? | [`StoppingDecodePipeline`](minideepinfer/stopping_pipeline.py) · [lessons twelve through fourteen](docs/CHUNK_PIPELINE.md) |
| How do issued progress, CPU commits, and final device use differ? | [`IssuedProgress` / `plan_step()`](minideepinfer/planning.py) · [lesson fourteen](docs/CHUNK_PIPELINE.md) |

Default BatchEngine supports synchronous dynamic batching, input budgets, prompt chunks, cancellation, and reclamation. `--deferred` separates launch from commit while keeping one batch in flight. Cross-round device-token handoff and callback arrivals remain a separate teaching experiment through `verify_chunk_pipeline`; they are not integrated into `generate`. Detailed lessons are currently in Chinese, with English code comments. The [learning path](docs/LEARNING_PATH.md) lists all fourteen.

## What is verified

| Check | Evidence |
| --- | --- |
| Real 2B / Ascend 910C FP32 | Single-request and two-request comparisons against independent references pass, with top-1 agreement at checked positions |
| Cross-round chunks, arrivals, and reuse | Nine rounds and fourteen full-logit comparisons pass at existing tolerances; maximum difference about `2.67e-5`. Four intermediate chunks produce no sample, and 24/24 slots are reclaimed |
| Cancellation and host errors | Release waits for final device use, and the Runner can be reused. Failed-drain ownership is separately checked through CPU fault injection |
| BF16 | Fixed short trajectories pass a matched recurrent reference with FP32 state. Native chunk-path and longer cross-schedule generation still differ; see the [notes](docs/BF16_REFERENCE.md) |

These model checks cover short-context correctness. Existing small-workload timing and trace experiments do not establish a general speedup, and this project makes no production-throughput claim. The [validation record](docs/VALIDATION.md) preserves environments, raw results, and source hashes. Run `python scripts/check_evidence.py` to compare the v0.1 evidence with a clean checkout without launching an NPU workload.

## Why DeepInfer

DeepInfer is the engine design reference: input preparation, model execution, sampling, result handling, and the lifetime of cross-round state become runnable code. The [source map](docs/SOURCE_MAP.md) identifies pinned upstream files and teaching simplifications. Qwen supplies the model example; DeltaNet is not presented as a DeepInfer-specific mechanism.

[mini-sglang](https://github.com/sgl-project/mini-sglang) informs the compact teaching format, while Ascend SGLang informs NPU API and compatibility work. MLA, MoE, speculative decoding, bounded replay, graph execution, and distribution remain on the [roadmap](docs/ROADMAP.md). See [references](docs/REFERENCES.md) for sources and licenses.

[Open an issue](https://github.com/AkiYangDev/mini-deepinfer/issues) with a concrete question: a confusing step, a command that fails, or a cache behavior needing an example. Code and tutorial improvements are welcome through the [contribution guide](CONTRIBUTING.md). If this learning path is useful, star the project to follow its development.

This is an independent community project with no official affiliation with DeepSeek, SGLang, or Huawei. Original code and documentation use [Apache-2.0](LICENSE); models and third-party code retain their respective licenses.
