<div align="center">

# mini-DeepInfer

**Follow a real request through a small inference engine inspired by DeepInfer.**

Built for learning, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

[简体中文](README.md) · **English**

[Setup](docs/START_HERE.en.md) · [A/B/C walkthrough](docs/WALKTHROUGH.en.md) · [Fourteen lessons](docs/LEARNING_PATH.md) · [v0.2 notes](docs/RELEASE_v0.2.md)

</div>

mini-DeepInfer is a teaching inference engine for Ascend 910C, based on ideas from [DeepInfer](https://github.com/deepseek-ai/DeepInfer). Follow a request through input preparation, model execution, sampling, and result handling, then see when its cache can be safely released.

The code runs **text-only Qwen3.5-2B on Ascend 910C** with its own forward pass and cache management. Fourteen lessons start with integer examples whose state you can print, then move to tiny PyTorch models and real checkpoints. Qwen3.5-0.8B is also used during development, and the earlier DeepSeek-R1-Distill-Qwen-1.5B example remains in the regression checks.

v0.2 connects cross-round execution to text generation and adds a complete scheduling walkthrough without a checkpoint. See the [release notes](docs/RELEASE_v0.2.md) for the new entry points and validation scope.

The v0.3 development version on main adds a [controlled greedy speculation lesson](docs/SPECULATIVE.md#english-summary): accept or correct proposed tokens, then roll back the rejected KV prefix. Start with a tiny model without weights, then inspect Qwen3.5 state snapshots and replay. A real 2B Ascend 910C run matched greedy output and cold-cache state checks. This synchronous experiment is separate from the text frontend and records snapshot and replay costs.

## Start small

With Python 3.10+, run an example that needs no model dependencies, weights, or NPU:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m pip install -e .
python -m minideepinfer.step_demo
```

In this example, A is cancelled while the device still needs its cache. B changes batch rows but must keep its own previous result. The demo prints both cases and advances completion events by hand, without executing a model. Its explanations are in Chinese.

With PyTorch installed, run a tiny hybrid-attention model without downloading weights:

```bash
python -m minideepinfer.pipeline_demo
```

In this example, A has a chunked prompt, B finishes first, and C waits before taking B's slots. Actual output includes:

```text
  B: finished; cached CPU/issued=3/3; slots=0
  C: running; cached CPU/issued=0/3; slots=6
  observed released: B
  reused B -> C: slots [14, 15, 16]
  free slots: 0/20
```

B has released its slots, yet none are free because C already took them. [Follow this run through the code](docs/WALKTHROUGH.en.md) to understand CPU progress, issued work, and ownership. For dependencies, see [setup](docs/START_HERE.en.md).

## Run a real model on Ascend 910C

Use an environment with matching PyTorch, torch_npu, and CANN. Install the Python dependencies and download the pinned Qwen3.5-2B checkpoint as described in [setup](docs/START_HERE.en.md). From the repository directory:

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
| What can be committed when a draft token is wrong? | [`verify_greedy()` / `CachedTarget`](minideepinfer/speculative.py) · [speculation lesson](docs/SPECULATIVE.md#english-summary) |

Default BatchEngine supports synchronous dynamic batching, input budgets, prompt chunks, cancellation, and reclamation. `--deferred` separates launch from commit while keeping one batch in flight. You can now choose cross-round execution with `--pipeline`, then add `--stagger-prompts --trace-rounds` to observe arrivals and cache ownership; see [setup](docs/START_HERE.en.md). Detailed lessons are currently in Chinese, with English code comments. The [learning path](docs/LEARNING_PATH.md) lists all fourteen.

## What is verified

| Check | Evidence |
| --- | --- |
| Real 2B / Ascend 910C FP32 | Single-request and two-request comparisons against independent references pass, with top-1 agreement at checked positions |
| Cross-round chunks, arrivals, and reuse | Nine rounds and fourteen full-logit comparisons pass at existing tolerances; maximum difference about `2.67e-5`. Four intermediate chunks produce no sample, and 24/24 slots are reclaimed |
| Text generation frontend | Sync, pipeline, and staggered arrivals produce matching output. Real chat reaches EOS; 43 full-logit comparisons include the issued EOS lookahead, with maximum difference about `4.94e-5`. See the [guide](docs/PIPELINE_TEXT.md) |
| Cancellation and host errors | Release waits for final device use, and the Runner can be reused. Failed-drain ownership is separately checked through CPU fault injection |
| BF16 | Fixed short trajectories pass a matched recurrent reference with FP32 state. Native chunk-path and longer cross-schedule generation still differ; see the [notes](docs/BF16_REFERENCE.md) |

These checks cover numerical correctness and state handling on short prompts. A [paired sync/pipeline benchmark](docs/PIPELINE_BENCHMARK.md#english-summary), added to main after v0.2.0, found longer cross-round completion times in all three short workloads. A subsequent [first-result experiment](docs/FIRST_RESULT.md#english-summary) delivers the first CPU-visible token earlier with identical batch plans, without establishing a consistent throughput gain. Both preserve every sample. The [validation record](docs/VALIDATION.md) keeps reports separately; [evidence checks](docs/EVIDENCE.md) now compare historical results with their recorded immutable commits. Run `python scripts/check_evidence.py --index docs/results/v0.2/evidence.json` from a Git checkout with the required history; it does not launch NPU work or certify changed source.

A later admission experiment collects a known length completion before scheduling queued work. The two four-request workloads use one fewer forward with matching outputs and input work. Timing varies substantially, so no stable speedup is established. This optional Python parameter remains off by default and reuses the existing benchmark; see the [report](docs/PIPELINE_BENCHMARK.md#english-summary).

## Why DeepInfer

In DeepInfer, input preparation, device execution, and result handling span several components and depend on work from earlier rounds. This project turns those steps into small experiments you can run and compare. The [source map](docs/SOURCE_MAP.md) links to the pinned upstream files and explains what the teaching version changes or leaves out. Qwen3.5 supplies a real model for studying the computation and its two kinds of cache.

[mini-sglang](https://github.com/sgl-project/mini-sglang) informs the compact teaching format, while Ascend SGLang informs NPU API and compatibility work. MLA, MoE, learned-draft execution, bounded replay, graph execution, and distribution remain on the [roadmap](docs/ROADMAP.md). See [references](docs/REFERENCES.md) for sources and licenses.

[Open an issue](https://github.com/AkiYangDev/mini-deepinfer/issues) with a concrete question: a confusing step, a command that fails, or a cache behavior needing an example. Code and tutorial improvements are welcome through the [contribution guide](CONTRIBUTING.md). If this learning path is useful, star the project to follow its development.

This is an independent community project with no official affiliation with DeepSeek, SGLang, or Huawei. Original code and documentation use [Apache-2.0](LICENSE); models and third-party code retain their respective licenses.
