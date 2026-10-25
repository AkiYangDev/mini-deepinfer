<div align="center">

# mini-DeepInfer

**Learn DeepInfer's engine design on CPU and Ascend 910C.**

A readable inference engine, from request scheduling to cache recovery.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

[简体中文](README.md) · **English**

[Quick start](#quick-start) · [Run on Ascend 910C](#generate-text-on-ascend-910c) · [Learning path](docs/LEARNING_PATH.md) · [Source map](docs/SOURCE_MAP.md)

</div>

mini-DeepInfer turns ideas from [DeepInfer](https://github.com/deepseek-ai/DeepInfer) into a teaching inference engine in Python and PyTorch. Scheduling, model forward passes and cache access live in this repository. Start with a CPU state walkthrough, then run real Qwen3.5 checkpoints on Ascend 910C and follow how a request computes, commits output, stops and releases its cache.

## Highlights

- **Real execution on Ascend 910C.** Qwen3.5-2B runs with single requests, batching and cross-round text generation. The repository keeps [actual output](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt), numerical comparisons and the corresponding source revisions.
- **CPU model examples without checkpoints.** Integer examples need only Python. With PyTorch, run random tiny models through actual forward passes, hybrid caches, cross-round scheduling and rejected proposals, without downloading weights.
- **Fourteen lessons along a request's lifetime.** Follow prefill and decode through chunks, arrivals and cancellation. Separate CPU-committed progress, issued device work and final device use, with [commands and code locations](docs/LEARNING_PATH.md) for each lesson.
- **A draft model with cache recovery.** Qwen3.5-0.8B proposes and 2B verifies. Main accepts your own text, streams committed output and supports cancellation. The [round walkthrough](docs/MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough) follows both caches back to the committed prefix after a rejection.

## Quick start

Python 3.10+, from the source checkout. This example needs no PyTorch, checkpoint or NPU:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

A is cancelled while the device still needs its cache. B changes batch rows but must retain its own previous result. The demo advances completion events by hand and prints the order of commits and releases. Terminal explanations are in Chinese.

With PyTorch installed, run a tiny Qwen3.5 model without downloading weights:

```bash
python -m minideepinfer.pipeline_demo
```

A has a chunked prompt, B finishes first, and C waits before taking B's slots. Actual output includes:

```text
  B: finished; cached CPU/issued=3/3; slots=0
  C: running; cached CPU/issued=0/3; slots=6
  observed released: B
  reused B -> C: slots [14, 15, 16]
  free slots: 0/20
```

B has released its slots, yet none are free because C already took them. [Follow A, B and C through the code](docs/WALKTHROUGH.en.md) to see how progress and ownership change. See [setup](docs/START_HERE.en.md) for dependencies. Random models illustrate execution state; their text is not a measure of model quality.

<details>
<summary>Follow a rejected proposal and cache recovery on CPU</summary>

```bash
python -m minideepinfer.model_draft_demo --cpu-example rejection --trace-rounds --width 3 --output outputs/draft-rejection.json
```

The draft moves from six computed inputs back to four valid ones, then consumes the correction to reach five. The [round walkthrough](docs/MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough) explains candidates, actual model inputs and committed output. No checkpoint or NPU is needed. Terminal labels are in Chinese; JSON fields are in English.

</details>

## Generate text on Ascend 910C

Prepare matching PyTorch, torch_npu and CANN versions, plus a local Qwen3.5-2B checkpoint, following [setup](docs/START_HERE.en.md). Run from the repository directory; multiline commands below use Bash:

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

The [complete bilingual output](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt) is preserved. Add `--pipeline` to use cross-round execution with your own text, then `--stagger-prompts --trace-rounds` to observe arrivals and cache release. See the [text frontend](docs/PIPELINE_TEXT.md).

### Two models, streaming and cancellation

On main, `generate --speculative` reuses the existing proposal verification and rollback loop, with chat templates, raw input, EOS, streamed output and cooperative cancellation. The [usage guide](docs/SPECULATIVE_TEXT.md#english-usage) covers arguments, independent greedy checks and saved results.

Commands on this page target **main (0.5.0.dev0)**. For a fixed learning version, use [v0.4.0](docs/RELEASE_v0.4.md#english-release-notes), which includes the dual-model round experiments. Custom speculative text, streaming and cancellation require main.

<details>
<summary>Run streamed generation and cancel at a commit boundary</summary>

With local Qwen3.5-0.8B and 2B checkpoints:

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --draft-model weights/Qwen3.5-0.8B \
  --speculative --device npu:0 --dtype float32 \
  --prompt "The capital of France is" --raw \
  --max-new-tokens 10 --cache-capacity 128 --draft-width 3 \
  --stream --cancel-after-tokens 2 --check-greedy \
  --output outputs/stream-cancel.json
```

The Ascend 910C check stopped after delivering a whole group, with four committed tokens and all 14 slots returned in each pool. Cancellation does not truncate a delivered group. The greedy check covers only that committed prefix; it does not certify the remaining output or EOS.

</details>

## Read along the code

| Mechanism | Code and guide |
| --- | --- |
| One request: prefill, decode and output commits | [`Engine.step()`](minideepinfer/engine.py) · [lesson one](docs/FIRST_REQUEST.md) |
| Dynamic batching, token budgets and chunked prefill | [`BatchEngine`](minideepinfer/batch_engine.py) · [lesson three](docs/BATCHING.md) |
| Qwen3.5 KV, convolution windows and recurrent state | [`HybridCache`](minideepinfer/hybrid_cache.py) · [lesson four](docs/QWEN35.md) |
| Reading, cancelling and releasing issued work | [`DeferredRunner`](minideepinfer/deferred_runner.py) · [lessons seven and eight](docs/DEFERRED_ENGINE.md) |
| Continuing before the CPU reads the previous token | [`StoppingDecodePipeline`](minideepinfer/stopping_pipeline.py) · [lessons eleven to fourteen](docs/CHUNK_PIPELINE.md) |
| Verification, acceptance, correction and two-cache rollback | [`verify_greedy()`](minideepinfer/speculative.py) / [`ModelDraft`](minideepinfer/model_draft.py) · [round walkthrough](docs/MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough) |

The [full fourteen-lesson index](docs/LEARNING_PATH.md) is in Chinese; English setup and the A/B/C walkthrough are available. The [source map](docs/SOURCE_MAP.md) links to a pinned DeepInfer revision and explains teaching simplifications. Synchronous, cross-round and speculative references retain separate execution paths.

## Validation and experiments

| Scenario | Saved evidence |
| --- | --- |
| Real weights and model execution | Qwen3.5-2B / Ascend 910C FP32 single-request and two-request comparisons against independent references; see [validation](docs/VALIDATION.md) |
| Cross-round text generation | Matching sync, pipeline and staggered-arrival outputs; 43 full-logit comparisons in an EOS trajectory pass, with maximum difference about `4.94e-5`; see [text checks](docs/PIPELINE_TEXT.md) |
| Dual-model state alignment | Widths 1/3 match independent greedy on the NPU; a random CPU rejection example shows draft truncation and recovery; see [implementation and experiments](docs/MODEL_DRAFT.md#english-summary) |
| Streamed cancellation | Four committed tokens match independent greedy's prefix, with all 14 slots returned in each pool; see the [cancellation record](docs/SPECULATIVE_TEXT.md#english-usage) |
| Probability sampling (CPU) | Toy distribution experiments, dual-model rejection sampling and cold-cache comparisons without checkpoints; see [sampling](docs/SAMPLING.md#english-usage) |

Reports preserve inputs, configuration, raw results and source hashes. Historical results are checked against immutable commits; see [evidence checks](docs/EVIDENCE.md). Performance experiments also retain slower short-workload results: [sync versus pipeline](docs/PIPELINE_BENCHMARK.md#english-summary), [first commit](docs/FIRST_RESULT.md#english-summary), and [dual-model costs](docs/MODEL_DRAFT.md#english-summary). Acceptance and fewer forward calls do not by themselves establish higher throughput.

## Scope and sources

The main examples use Qwen3.5, FP32, eager execution and greedy decoding on short contexts. See [BF16 notes](docs/BF16_REFERENCE.md) for reference checks and chunk-path differences. The earlier DeepSeek-R1-Distill-Qwen-1.5B example remains in regression checks. DeepSeek-V4.1 model architecture, MLA/MoE, MTP, bounded replay, graph execution and distribution remain on the [roadmap](docs/ROADMAP.md).

A separate [CPU temperature-sampling reference](docs/SAMPLING.md#english-usage) reuses speculative cache rollback. This new path has not been validated on Ascend 910C.

Engine design primarily follows [DeepInfer](https://github.com/deepseek-ai/DeepInfer). [mini-sglang](https://github.com/sgl-project/mini-sglang) informs the compact teaching format; Ascend SGLang informs NPU API and compatibility work. See [references](docs/REFERENCES.md) for sources and licenses.

Star the repository to keep it handy, or [open an issue](https://github.com/AkiYangDev/mini-deepinfer/issues/new/choose) with a command that fails or a state change that needs a clearer explanation. Code and tutorial contributions are welcome through the [contribution guide](CONTRIBUTING.md).

This is an independent community project with no official affiliation with DeepSeek, SGLang or Huawei. Original code and documentation use [Apache-2.0](LICENSE); models and third-party code retain their respective licenses.
