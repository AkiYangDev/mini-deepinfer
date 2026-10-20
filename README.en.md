<div align="center">

# mini-DeepInfer

**Follow a real request through a small inference engine inspired by DeepInfer.**

Built for learning, targeting Ascend 910C.

[![CPU checks](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml/badge.svg)](https://github.com/AkiYangDev/mini-deepinfer/actions/workflows/cpu.yml) [![Target: Ascend 910C](https://img.shields.io/badge/Target-Ascend%20910C-2563eb)](docs/VALIDATION.md) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-16a34a)](LICENSE)

[简体中文](README.md) · **English**

[Read the code](#follow-a-request-through-the-code) · [Run a model](#run-a-real-model) · [Validation](docs/VALIDATION.md) · [Roadmap](docs/ROADMAP.md)

</div>

mini-DeepInfer studies the design of [DeepInfer](https://github.com/deepseek-ai/DeepInfer) through a small teaching engine: how inputs become tokens, how prefill leads into decode, where the KV cache lives, and who releases it when a request ends.

The project can now run **DeepSeek-R1-Distill-Qwen-1.5B** using its own Qwen2 forward pass and tensor KV cache. You can read and run the CPU path without an NPU. **Ascend 910C is the target platform; full-model NPU validation is still pending.**

## What runs today

| Check | Result |
| --- | --- |
| Real weights, CPU FP32 | 5-token prefill + 4 fixed-token decode steps; maximum absolute logits difference from the Transformers cached path: **0** |
| Cache vs. recomputation | Maximum absolute difference about `7.56e-5`, also present in the reference model's own cache/recompute comparison; top-1 agrees at every checked position |
| Text generation | 96 generated tokens, length stop, then **512/512** KV slots reclaimed |
| Automated checks | 16 tests pass, covering model comparisons, causal masking, noncontiguous slots, reuse, and request lifecycles |

These results apply to the recorded model, inputs, and dependency versions. See the [validation record](docs/VALIDATION.md) for the environment and raw JSON. Execution is synchronous, with one active request; dynamic batching and scheduling overlap are not implemented yet.

A real CPU run with `What is 1 plus 1? Answer in one short sentence.` produced this excerpt:

```text
Okay, so I need to figure out what 1 plus 1 is. Hmm, I remember learning this in school,
but I'm not entirely sure. Let me think. ...

[length; 96 output tokens; free KV slots 512/512]
```

This is an unfinished R1 reasoning passage. It demonstrates generation and reclamation, rather than answer quality or performance. The [complete output](docs/results/2026-10-20/generation-cpu.txt) is preserved.

## Run a real model

Clone the project and install the model dependencies:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m pip install -e ".[model]"
```

Prepare local weights using the [pinned download command in lesson two](docs/REAL_GENERATION.md), then start on CPU:

```bash
python -m minideepinfer.generate \
  --model weights/DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu --dtype float32 \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 96
```

NPU execution requires compatible PyTorch, `torch_npu`, and CANN versions. [Lesson two](docs/REAL_GENERATION.md) includes the NPU and numerical-check commands. Start with short contexts; the CPU path also loads the full model weights.

## Follow a request through the code

If you know basic Python and PyTorch, follow these entry points. The project implements the model computation; Transformers supplies the tokenizer and numerical reference.

| Question | Read |
| --- | --- |
| What does each prefill or decode step consume? | [`Engine.step()`](minideepinfer/engine.py) |
| How do tokens, positions, and KV slots line up? | [`ModelRunner.prepare_input()`](minideepinfer/model_runner.py) |
| How does attention write and read past K/V? | [`qwen2.py`](minideepinfer/models/qwen2.py) · [`TensorKVCache`](minideepinfer/tensor_cache.py) |
| How do logits become the next token? | [`ModelRunner.execute_sample()`](minideepinfer/model_runner.py) |
| Who releases slots on completion or cancellation? | [`Engine`](minideepinfer/engine.py) · [`SlotPool`](minideepinfer/kv_cache.py) |

New to inference engines? [Lesson one](docs/FIRST_REQUEST.md) uses an integer runner so every input, position, and slot can be printed. It needs only Python 3.10+, without model dependencies or weights:

```bash
python -m minideepinfer
python -m unittest discover -s tests -v
```

Without model dependencies, 8 engine tests run and 8 model tests are explicitly skipped. [Lesson two](docs/REAL_GENERATION.md) replaces the runner with a real model. Detailed lessons and design documents are currently in Chinese; code comments are in English.

## How this relates to DeepInfer

DeepInfer is the primary technical reference. The current implementation separates input preparation, main-model execution, sampling, and result handling. Future lessons will study in-flight states and scheduling overlap. The [source map](docs/SOURCE_MAP.md) identifies the pinned upstream revision, corresponding entry points, and teaching simplifications.

**Qwen2 is the introductory model, not a reproduction of DeepInfer's advanced mechanisms.** Next comes full-model Ascend 910C validation, followed by synchronous dynamic batching. MLA, MoE, speculative decoding, and bounded replay belong to later, separate lessons; see the [roadmap](docs/ROADMAP.md).

[mini-sglang](https://github.com/sgl-project/mini-sglang) informs the compact teaching format; Ascend SGLang informs NPU API and compatibility work. See the [architecture](docs/ARCHITECTURE.md) and [references](docs/REFERENCES.md) for further details.

## Contribute

Bring a concrete question: a step that is hard to follow, a command that fails, or a cache behavior that needs an example. [Open an issue](https://github.com/AkiYangDev/mini-deepinfer/issues), or improve the code and lessons through the [contribution guide](CONTRIBUTING.md).

If this learning path is useful to you, star the project to follow the next implementation and experiment.

This is an independent community project, with no official affiliation with DeepSeek, SGLang, or Huawei. Original code and documentation use [Apache-2.0](LICENSE); models and third-party code retain their respective licenses.
