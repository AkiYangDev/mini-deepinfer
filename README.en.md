<div align="center">

# ![mini-DeepInfer](docs/assets/readme-hero.svg)

**Learn DeepInfer's engine design by running a small inference engine.**

Start with one request on CPU. Run real models on Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Ascend 910C](https://img.shields.io/badge/Ascend%20910C-FP32%20checked-0f766e)](docs/VALIDATION.md) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-2563eb)](pyproject.toml) [![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-475569)](LICENSE)

[简体中文](README.md) · **English**

[Quick start](#quick-start) · [Ascend 910C](#generate-text-on-ascend-910c) · [Speculative decoding](#speculative-decoding-and-sampling) · [Fourteen lessons](docs/LEARNING_PATH.md) · [DeepInfer source map](docs/SOURCE_MAP.md)

</div>

Reading an inference engine means following more than model forward passes: when a request joins a batch, when a computed token reaches the user, and when its cache can belong to another request. mini-DeepInfer makes those steps visible in runnable Python and PyTorch code, drawing on [DeepInfer](https://github.com/deepseek-ai/DeepInfer)'s execution design.

Qwen3.5 is the current model example. Scheduling, model forward passes, KV and recurrent state, verification and rollback all live in this repository. Follow actual output back to the code that produced it.

## Quick start

Commands follow main (0.5.0.dev0) and need Python 3.10+. Three commands to follow a request, with no dependencies or checkpoints to install:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m minideepinfer.step_demo
```

A is cancelled while the device still needs its cache. B changes batch rows but must retain its previous result. Manual completion events separate issued computation, committed output and cache release. Terminal explanations are in Chinese.

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

Each route leads to runnable code in this repository. Fourteen lessons progress from one request to dynamic batching, chunked prefill, cross-round execution, arrivals and cancellation. The [lesson index](docs/LEARNING_PATH.md) provides commands and code locations in Chinese; English setup and the A/B/C walkthrough are available.

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

Qwen3.5-0.8B proposes and 2B verifies. Look beyond the accepted-token count: which temporary results does the target discard, how does the draft restore recurrent state, and which prefix does each model consume next?

Main accepts your own text, streams committed groups and supports cancellation; the [text guide](docs/SPECULATIVE_TEXT.md#english-usage) has complete commands. Positive-temperature sampling reuses the same loop, including probabilistic acceptance and correction sampling after rejection. [Toy experiments](docs/SAMPLING.md#english-usage) explain the rule; Ascend 910C records check probabilities and state recovery. Model execution uses NPU FP32 while probabilities stay in CPU FP64.

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

</details>

Commands target **main (0.5.0.dev0)**. For a fixed learning version, see [v0.4.0](docs/RELEASE_v0.4.md#english-release-notes). Custom speculative text, streaming cancellation and probability sampling require main.

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

## Results you can check

| Saved record | What it checks |
| --- | --- |
| [Qwen3.5-2B on the NPU](docs/VALIDATION.md) | Ascend 910C FP32 model execution, single requests and batching |
| [Cross-round text](docs/PIPELINE_TEXT.md) | Matching sync, pipeline and staggered-arrival outputs; 43 full-logit comparisons pass |
| [Speculative sampling](docs/SAMPLING.md#english-usage) | 28 probability and six cold-state comparisons pass, including actual draft recovery after rejection |
| [Streamed cancellation](docs/SPECULATIVE_TEXT.md#english-usage) | Four delivered tokens match an independent greedy prefix; both pools release all slots |

Reports preserve inputs, configuration, raw results and source hashes; [historical records use immutable commits](docs/EVIDENCE.md). [Sync versus pipeline](docs/PIPELINE_BENCHMARK.md#english-summary), [first commit](docs/FIRST_RESULT.md#english-summary) and [dual-model costs](docs/MODEL_DRAFT.md#english-summary) also retain slower short-workload results. Acceptance and fewer forward calls alone do not establish higher throughput.

## Scope and sources

The project stays within single-device, FP32, eager execution on short contexts. Greedy and positive-temperature speculative sampling are implemented. See [BF16 notes](docs/BF16_REFERENCE.md) for reference checks and chunk-path differences. The earlier DeepSeek-R1-Distill-Qwen-1.5B example remains in regression checks. DeepSeek-V4.1 architecture, MLA/MoE, MTP, bounded replay, graphs and distributed execution remain on the [roadmap](docs/ROADMAP.md).

Engine design primarily follows [DeepInfer](https://github.com/deepseek-ai/DeepInfer). [mini-sglang](https://github.com/sgl-project/mini-sglang) informs the compact teaching format; Ascend SGLang informs NPU API work. [References and licenses](docs/REFERENCES.md) record the sources.

Learning inference engines too? Star the repository as a code-reading reference. [Open an issue](https://github.com/AkiYangDev/mini-deepinfer/issues/new/choose) with a failing command or a state change that needs explaining, or start with the [contribution guide](CONTRIBUTING.md) to improve code and tutorials.

This independent community project has no official affiliation with DeepSeek, SGLang or Huawei. Original code and documentation use [Apache-2.0](LICENSE); models and third-party code retain their licenses.
