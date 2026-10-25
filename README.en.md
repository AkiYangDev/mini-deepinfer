<div align="center">

# ![mini-DeepInfer](docs/assets/readme-hero-whale.png)

**Explore DeepInfer's engine design. Run real inference on Ascend 910C.**

From three CPU commands to dual-model speculative decoding with Qwen3.5.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Ascend 910C](https://img.shields.io/badge/Ascend%20910C-FP32%20checked-0f766e)](docs/VALIDATION.md) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-2563eb)](pyproject.toml) [![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-475569)](LICENSE)

[简体中文](README.md) · **English**

[Quick start](#quick-start) · [Ascend 910C](#generate-text-on-ascend-910c) · [Speculative decoding](#speculative-decoding-and-sampling) · [Fourteen lessons](docs/LEARNING_PATH.md) · [DeepInfer source map](docs/SOURCE_MAP.md)

</div>

mini-DeepInfer is a compact inference engine for **Ascend 910C**, inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer)'s execution design. Request scheduling, model execution, output commits and cache management are implemented in Python and PyTorch, with real Qwen3.5 checkpoints running on the NPU.

Follow a request through the engine, change a scheduling decision, or cancel generation and inspect how execution continues.

## What you can build on

<img align="right" src="docs/assets/deepseek-girl-full.png" width="126" alt="Blue-haired, whale-themed DeepSeek girl illustration" />

- **Real models on Ascend 910C.** Run Qwen3.5-2B text generation, dynamic batching and cross-round execution. [Commands and saved output](docs/VALIDATION.md) accompany the NPU checks.
- **Scheduling you can follow.** Trace chunked prompts, new arrivals and the next round starting before the CPU reads the previous token. Watch [requests A, B and C](docs/WALKTHROUGH.en.md) take ownership of cache slots and reuse them.
- **Dual-model speculation, including recovery.** Qwen3.5-0.8B proposes; 2B verifies. Follow rejection, recurrent-state recovery, probability sampling, streamed groups and cancellation. [Start with a wrong proposal](docs/MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough).
- **A runnable path into engine internals.** Start the CPU scheduling demo in three commands, with no dependencies or checkpoints. [Fourteen lessons](docs/LEARNING_PATH.md) introduce batching, chunked prefill and cross-round execution; the [DeepInfer source map](docs/SOURCE_MAP.md) connects the implementation to its design references.

<br clear="both" />

## Quick start

Run the engine first. This scheduling demo needs only the Python 3.10+ standard library, with no dependencies or checkpoints to install:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

A is cancelled while the device still needs its cache. B changes batch rows but must retain its previous result. Manual completion events separate issued computation, committed output and cache release. Terminal explanations are in Chinese.

Use **[v0.5.0](docs/RELEASE_v0.5.md#english-release-notes)** as the fixed learning version. It includes custom speculative text, streaming cancellation, temperature and top-k / top-p sampling. The [setup guide](docs/START_HERE.en.md) uses the same tag; older releases and reports remain available.

**Ready for real forward passes?** With PyTorch installed, run a random tiny Qwen3.5 model without downloading weights:

```bash
python -m minideepinfer.pipeline_demo
```

![Actual CPU output: B finishes and C reuses its cache slots](docs/assets/cache-reuse.svg)

B released its slots, yet none are free because C already took them. A's long prompt is still being processed in chunks. [Follow A, B and C through the code](docs/WALKTHROUGH.en.md), or read the [full terminal output](docs/assets/pipeline-demo.txt). Random models illustrate execution state; their text does not demonstrate language quality.

## Pick a starting point

| What you want to try | Start here |
| --- | --- |
| Explore scheduling and cache ownership on CPU | [Setup](docs/START_HERE.en.md) → [A/B/C walkthrough](docs/WALKTHROUGH.en.md) |
| Run your own prompt on Ascend 910C | [Environment and weights](docs/START_HERE.en.md) → [Text generation](#generate-text-on-ascend-910c) |
| Understand recovery after a wrong proposal | [Round walkthrough](docs/MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough) → [Probability sampling](docs/SAMPLING.md#english-usage) |

Each route has runnable examples. The [lesson index](docs/LEARNING_PATH.md) provides commands and code locations in Chinese; English setup and the A/B/C walkthrough are available.

<details>
<summary>Follow a rejected proposal and cache recovery on CPU</summary>

```bash
python -m minideepinfer.model_draft_demo --cpu-example rejection --trace-rounds --width 3 --output outputs/draft-rejection.json
```

The draft moves from six computed inputs back to four valid ones, then consumes the correction to reach five. The [round walkthrough](docs/MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough) follows candidates, actual model inputs and committed output. No checkpoint or NPU is needed. Terminal labels are in Chinese; JSON fields are in English.

</details>

## Generate text on Ascend 910C

Real Qwen3.5-2B checkpoints have passed single-request, batch and cross-round generation checks. Prepare matching PyTorch, torch_npu and CANN versions and local weights using [setup](docs/START_HERE.en.md), then run from the repository directory. Multiline commands use Bash:

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

The English answer from the real run:

```text
1 plus 1 equals 2.

[eos; 9 output tokens; free KV slots 512/512]
```

The [complete bilingual output](docs/results/2026-10-24/qwen35-2b-generation-npu-fp32.txt) is preserved. Add `--pipeline` for cross-round execution and `--stagger-prompts --trace-rounds` to follow arrivals and cache release. See the [text frontend](docs/PIPELINE_TEXT.md).

## Speculative decoding and sampling

Let Qwen3.5-0.8B propose and 2B verify. Accept a group and deliver several tokens together; reject it, discard temporary results, restore both caches and continue from the valid prefix. KV, convolution windows and recurrent state must all agree before the next forward pass.

Use your own prompts, watch committed groups stream out, or cancel after a commit. The [text guide](docs/SPECULATIVE_TEXT.md#english-usage) has complete commands. Positive-temperature sampling uses the same loop, including probabilistic acceptance and correction sampling after rejection. The [sampling guide](docs/SAMPLING.md#english-usage) takes you from small-vocabulary experiments to probability and state-recovery checks on Ascend 910C. Model execution uses NPU FP32 while probabilities stay in CPU FP64.

<details>
<summary>Stream output and cancel at a commit boundary</summary>

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

The saved run stopped after delivering a whole group, with four committed tokens and all 14 slots returned in each pool. Cancellation does not truncate an accepted group. The greedy check covers only the delivered prefix, not the remaining output or EOS. Random sampling cannot use `--check-greedy`.

<p align="center"><img src="docs/assets/deepseek-girl-chibi.jpg" width="96" alt="A small glowing-eyed chibi whale girl Easter egg" /></p>

</details>

## Find the code behind a question

| Mechanism | Code and guide |
| --- | --- |
| One request: prefill, decode and output commits | [`Engine.step()`](minideepinfer/engine.py) · [lesson one](docs/FIRST_REQUEST.md) |
| Batching, token budgets and chunked prefill | [`BatchEngine`](minideepinfer/batch_engine.py) · [lesson three](docs/BATCHING.md) |
| KV, convolution windows and recurrent state | [`HybridCache`](minideepinfer/hybrid_cache.py) · [Qwen3.5](docs/QWEN35.md) |
| Continuing before the CPU reads the previous token | [`StoppingDecodePipeline`](minideepinfer/stopping_pipeline.py) · [cross-round chunks](docs/CHUNK_PIPELINE.md) |
| Recovering both caches after rejection | [`ModelDraft`](minideepinfer/model_draft.py) · [round walkthrough](docs/MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough) |
| Acceptance and correction sampling | [`verify_sampled()`](minideepinfer/sampling.py) · [sampling experiments](docs/SAMPLING.md#english-usage) |

The [fourteen-lesson index](docs/LEARNING_PATH.md) provides the full sequence. The [DeepInfer source map](docs/SOURCE_MAP.md) records corresponding upstream code, retained mechanisms and teaching simplifications.

## Already running on Ascend 910C

| Saved record | What it checks |
| --- | --- |
| [Qwen3.5-2B on the NPU](docs/VALIDATION.md) | Ascend 910C FP32 model execution, single requests and batching |
| [Cross-round text](docs/PIPELINE_TEXT.md) | Matching sync, pipeline and staggered-arrival outputs; 43 full-logit comparisons pass |
| [Speculative sampling](docs/SAMPLING.md#english-usage) | 28 probability and six cold-state comparisons pass, including actual draft recovery after rejection |
| [Streamed cancellation](docs/SPECULATIVE_TEXT.md#english-usage) | Four delivered tokens match an independent greedy prefix; both pools release all slots |

Every result includes inputs, configuration, raw records and source hashes for [verification at an immutable commit](docs/EVIDENCE.md). To explore performance, start with [sync versus pipeline](docs/PIPELINE_BENCHMARK.md#english-summary), [first commit](docs/FIRST_RESULT.md#english-summary) and [dual-model costs](docs/MODEL_DRAFT.md#english-summary). These include short workloads that got slower, so you can study when an optimization pays off. Assess acceptance and forward-call counts alongside measured runtime.

## Scope and sources

The project stays within single-device, FP32, eager execution on short contexts. Greedy and positive-temperature speculative sampling are implemented. See [BF16 notes](docs/BF16_REFERENCE.md) for reference checks and chunk-path differences. The earlier DeepSeek-R1-Distill-Qwen-1.5B example remains in regression checks. DeepSeek-V4.1 architecture, MLA/MoE, MTP, bounded replay, graphs and distributed execution remain on the [roadmap](docs/ROADMAP.md).

Engine design primarily follows [DeepInfer](https://github.com/deepseek-ai/DeepInfer). [mini-sglang](https://github.com/sgl-project/mini-sglang) informs the compact teaching format; Ascend SGLang informs NPU API work. [References and licenses](docs/REFERENCES.md) record the sources.

**Start with three commands and explore inference engines on Ascend 910C.** Star the repository to keep it in your reading list. Bring a failing command or a mechanism you want explained to [Issues](https://github.com/AkiYangDev/mini-deepinfer/issues/new/choose), or help implement a mechanism and improve a lesson through the [contribution guide](CONTRIBUTING.md).

This independent community project has no official affiliation with DeepSeek, SGLang or Huawei. Original code and documentation use [Apache-2.0](LICENSE); models and third-party code retain their licenses. Character artwork is covered separately in the [artwork notes](docs/assets/ARTWORK.md).
