# mini-DeepInfer

**Understand inference. Build on Ascend.**

A readable inference engine for DeepSeek models on Ascend 910C.

[简体中文](README.md) · [Roadmap](docs/ROADMAP.md) · [Architecture](docs/ARCHITECTURE.md) · [Validation](docs/VALIDATION.md)

mini-DeepInfer is an educational inference project targeting **Ascend 910C**. It brings together readable Python implementations, explanations, and reproducible hardware experiments to show how requests become batches, how KV caches are managed, and how prefill and decode execute.

The project draws on [DeepInfer](https://github.com/deepseek-ai/DeepInfer) for engine design and [mini-sglang](https://github.com/sgl-project/mini-sglang) for its compact, accessible approach to learning serving systems.

> **Status: design and documentation.** The inference engine is not implemented yet. Ascend 910C is the target platform; model and operator support will be documented as it is verified.

## What we are building

| Area | Learning goal |
| --- | --- |
| Ascend execution | Understand `torch_npu`, device memory, streams, events, fused operators, and graph execution |
| Autoregressive generation | Trace weight loading, prefill, KV updates, decode, and sampling |
| Serving | Implement continuous batching, paged KV management, chunked prefill, and streaming responses |
| DeepSeek architecture | Explore MLA, compressed caches, MoE routing, and expert execution |
| DeepInfer mechanism labs | Study scheduling overlap, speculative commit/rollback, and SWA bounded replay in focused experiments |
| Reproducible optimization | Compare correctness, TTFT, TPOT, throughput, and memory under matched workloads |

Each optimization follows the same path: **read the reference implementation, verify correctness, introduce an optimization, and measure the tradeoff.** All capabilities above are planned; completion criteria are in the [roadmap](docs/ROADMAP.md).

## Model progression

- **DeepSeek-R1-Distill-Qwen-1.5B:** establish a real generation loop using its Qwen architecture.
- **DeepSeek-V2-Lite / V2-Lite-Chat:** teach MLA and MoE with real checkpoints and numerical checks.
- **Focused DeepInfer labs:** reproduce selected mechanisms from a pinned upstream revision. Full V4.1 deployment is a later research milestone.

These models have different architectures. Each lesson will state its applicable model, assumptions, and simplifications. Tiny random-weight configurations validate mechanisms, not language capability.

## Start reading

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
```

Start with the [roadmap](docs/ROADMAP.md), [architecture](docs/ARCHITECTURE.md), [validation plan](docs/VALIDATION.md), and [source references](docs/REFERENCES.md). Detailed design documents are currently in Chinese. Installation and inference commands will be added alongside verified implementations.

## Contributing

Reproducible bug reports, operator checks, teaching examples, implementation work, and documentation are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments and license

Thanks to [DeepInfer](https://github.com/deepseek-ai/DeepInfer), [mini-sglang](https://github.com/sgl-project/mini-sglang), and [SGLang](https://github.com/sgl-project/sglang) for their open work.

This is an independent community project, unaffiliated with DeepSeek, SGLang, or Huawei. Original code and documentation are licensed under [Apache-2.0](LICENSE). Model weights and third-party components retain their own licenses; attribution is tracked in [References](docs/REFERENCES.md).
