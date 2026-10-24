# From installation to the first run

[Home](../README.en.md) · [简体中文](START_HERE.md) · [Learning path](LEARNING_PATH.md)

The base package has no runtime dependencies. PyTorch is only needed for model execution. Multiline commands below use Bash; in PowerShell, put each command on one line. Use the same Python environment for installation and execution.

## No weights or hardware

With Python 3.10+:

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. From the repository directory:

```bash
python -m pip install -e .
python -m minideepinfer
python -m minideepinfer.batch_demo
python -m minideepinfer.step_demo
```

The first two examples trace integer requests and batches. The third advances events by hand to explain row remapping and protected cache ownership. These are simulations, not model execution; their explanations are in Chinese.

## Tiny CPU model, no checkpoint

In a regular CPU environment:

```bash
python -m pip install -e ".[qwen35]"
python -m minideepinfer.verify_chunk_pipeline --device cpu \
  --output outputs/chunk-pipeline-cpu.json
```

The checker itself only needs PyTorch. The extra also installs the tokenizer and safetensors dependencies used by real checkpoints. If compatible PyTorch is already installed, the base package is sufficient for this random-model check.

It compares whole-prompt execution with budgeted cross-round chunks in a tiny hybrid-attention model, including arrivals, host-error cleanup, and reuse. Expect `passed: true`, `free_slots: 24`, and no remaining prefixes or linear states. Floating-point error values need not be identical across environments.

## Real Qwen3.5-2B on Ascend 910C

Start with an existing environment that has matching PyTorch, torch_npu, CANN, and drivers. The tested combination is Python 3.12.13, PyTorch 2.10.0+cpu, torch_npu 2.10.0.post6, and CANN 9.1.0 on aarch64. The `+cpu` suffix is the actual version string in this NPU environment; torch_npu registers the NPU backend. Keep these device dependencies in place.

Install the model interfaces and project:

```bash
python -m pip install transformers==5.12.1 safetensors==0.8.0
python -m pip install -e . --no-deps
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

Confirm NPU availability. Download the pinned checkpoint, or reuse an existing copy of the same revision. Model execution reads local files without downloading:

```bash
hf download Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --local-dir weights/Qwen3.5-2B
```

Check that your selected logical device is available and has room for FP32 weights, cache, and workspace. Device IDs depend on your mapping; the checker does not reserve or isolate hardware. From the repository directory:

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

This uses synchronous BatchEngine. Each output ends with its stopping reason, generated length, and free-slot count. The preserved [bilingual output](results/2026-10-24/qwen35-2b-generation-npu-fp32.txt) stops both requests at EOS. Each request reserves `prompt length + max_new_tokens - 1` logical slots; cache capacity does not automatically grow with free device memory.

For the independent cross-round teaching experiment:

```bash
python -m minideepinfer.verify_chunk_pipeline \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/chunk-pipeline-npu.json
```

This runs reference and experimental trajectories. To inspect existing evidence instead of repeating device computation:

```bash
python scripts/check_evidence.py
```

It compares the current checkout, normalizing CRLF/LF, with source hashes in the preserved NPU report. A mismatch means the historical run does not directly certify your modified source; it is not a newly failed model test. Earlier lesson reports retain historical source versions.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing torch | Install model dependencies in a CPU environment, or use your configured NPU environment |
| torch_npu import failure / NPU unavailable | Check matching device dependencies, drivers, and container device mappings |
| Whole prompt exceeds budget | Enable `--chunked-prefill`; each request must still fit in cache |
| Request exceeds capacity | Check prompt length plus output limit; logical slots are not device MiB |
| Old Transformers cannot load Qwen3.5 tokenizer | Use 5.12.1 for this example; 4.46.3 covers the older Qwen2 regression |
| BF16 generation changes across chunks | Read the [scan/state precision notes](BF16_REFERENCE.md); v0.1 defaults to FP32 |

Run `python -m unittest discover -s tests -v` from source. Model tests explicitly skip without their dependencies. The installable package contains runtime modules; lessons, tests, scripts, and reports live in the source repository. v0.1 is not published on PyPI.
