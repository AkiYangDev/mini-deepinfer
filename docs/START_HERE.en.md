# From installation to the first run

[Home](../README.en.md) · [简体中文](START_HERE.md) · [Learning path](LEARNING_PATH.md)

Start with the first section even if you have no NPU or checkpoint. The integer examples need only Python; the later sections add PyTorch and real weights.

Multiline commands use Bash. In PowerShell, put each command on one line. Install and run everything in the same Python environment.

The steps below use the fixed v0.4.0 tag. To upgrade an existing v0.3 setup, use the [model-draft steps](#v04-let-a-small-model-propose) later in this guide. Integer, cross-round and lookup examples remain available.

## No weights or hardware

The integer examples support Python 3.10+. Use Python 3.12 below if you also want to run the CPU model examples, matching the version used for those checks:

For integer examples only, an existing Python 3.10+ can run `python -m venv .venv`. The platform commands below create a Python 3.12 environment.

```bash
git clone https://github.com/AkiYangDev/mini-deepinfer.git
cd mini-deepinfer
git checkout v0.4.0
```

Create and activate the environment on Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

In Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

From the repository directory:

```bash
python -m pip install -e .
python -m minideepinfer
python -m minideepinfer.batch_demo
python -m minideepinfer.step_demo
```

The first two examples use integers in place of model output to show the request loop and batch scheduling. The third advances completion events by hand so you can inspect row remapping and cache ownership after cancellation. The explanations are in Chinese.

## Tiny CPU model, no checkpoint

In the Python 3.12 environment above, install CPU PyTorch. On Linux/Windows:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

On macOS:

```bash
python -m pip install torch==2.5.1
```

These download sources follow the [official PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/#v251). Project CPU checks cover Windows/Linux; this is not a new macOS validation. Skip installation if compatible PyTorch is already present. The random examples use this project's forward pass and need no Transformers, tokenizer, safetensors or checkpoint:

```bash
python -m minideepinfer.pipeline_demo
```

Real-checkpoint interface dependencies are introduced in the Ascend section below.

The tiny hybrid-attention model processes A's long prompt in chunks. B finishes first, and C waits before taking B's slots. Expect `whole-prompt reference outputs match: True` and `free slots: 20/20; clean: True`. The [A/B/C walkthrough](WALKTHROUGH.en.md) explains the observations.

Then compare full logits using the text frontend:

```bash
python -m minideepinfer.verify_pipeline_text --device cpu \
  --output outputs/pipeline-text-cpu.json
```

Random-model mode does not use natural EOS. With a real checkpoint, the verifier also checks chat-template generation and natural EOS. Floating-point error values need not be identical across environments.

## v0.3: continue after rejecting a draft

With the CPU model dependencies ready, run these in order:

```bash
python -m minideepinfer.spec_demo --output outputs/spec-controlled.json
python -m minideepinfer.spec_demo --hybrid --output outputs/spec-hybrid.json
python -m minideepinfer.spec_demo --draft lookup --hybrid --output outputs/spec-lookup.json
```

Inspect verification, hybrid-state recovery, then candidates from committed history only. Lookup runs its independent greedy baseline afterwards. Each JSON should contain `passed: true`, output and cleanup checks, and separate work counters. See [v0.3 notes](RELEASE_v0.3.md#english-release-notes) for expected results and hardware scope. These examples make no timing or speedup claim.

For `outputs/spec-lookup.json`, start with these fields:

| Field | Expected in this CPU example |
| --- | --- |
| `passed` | `true`: the overall check passed |
| `cases.lookup.matches_greedy` | `true`: output matches the subsequent independent baseline |
| `cases.lookup.free_slots` | `12`: all slots were returned |
| `cases.lookup.target_forward_calls` | `6`: generation calls, including replay |
| `baseline.forward_calls` | `10`: separate baseline calls |
| `reference_forward_calls` | `5`: cold-cache checks after commits |
| `total_forward_calls` | `21`: the sum of those three parts |

The total includes extra work used to verify the example. No time was measured, so `10/6` is not a speedup. This random model repeats one token at the fixed seed, making lookup matches easy. The real checkpoint accepted 0/5; see the [independent-draft results](LOOKUP_DRAFT.md#english-summary).

## v0.4: let a small model propose

Use the CPU PyTorch environment from above. From the repository directory, switch to v0.4.0 and reinstall the project so the package metadata follows the source:

```bash
git checkout v0.4.0
python -m pip install -e . --no-deps
python -c "from importlib.metadata import version; print(version('mini-deepinfer'))"
python -m minideepinfer.model_draft_demo --output outputs/model-draft-cpu.json
```

The current version should be `0.4.0`. Two randomly initialized Qwen3.5 models propose and verify candidates without weights or downloads. The default run checks correctness without timing. Expect `greedy outputs and both caches match: True`, followed by results for widths 1 and 3.

Start with the width-1 row in the generated JSON:

| Field | Expected in the current CPU example |
| --- | --- |
| `passed` | `true`: output, cache and cleanup checks passed |
| `checked_runs[0].target_forward_calls` | `6`: generation calls, counting replay when needed |
| `checked_runs[0].draft_work.forward_calls` | `4`: draft computation also contributes to cost |
| `checked_runs[0].draft_work.snapshot_count` | `0`: one candidate needs no tentative draft input to roll back |
| `checked_runs[0].draft_work.closed` | `true`: the draft was closed and cleaned up |
| `checked_runs[0].draft_work.free_slots` | `12`: all draft slots were returned |

These random models repeat one token at the fixed seed; their acceptance does not predict real-model performance. See the [dual-model explanation](MODEL_DRAFT.md#english-summary) for state transitions and real 0.8B-to-2B costs. To compare the previous and current draft snapshot policies, run:

```bash
python -m minideepinfer.model_draft_demo --compare-draft-snapshots --output outputs/draft-snapshot-cpu.json
```

This writes a separate report, also without timing. The comparison must preserve model work in each stage before the checks can pass.

For a rejected-candidate trajectory, follow the [round walkthrough](MODEL_DRAFT_WALKTHROUGH.md#english-walkthrough) with `--cpu-example rejection --trace-rounds --width 3`. It uses untrained random models with untied output heads and needs no checkpoint.

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

Check that your selected logical device is available and has room for FP32 weights, cache, and workspace. The example uses `npu:0`; choose the ID mapped into your environment. From the repository directory:

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "用两句话解释 KV Cache 的作用。" \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --max-new-tokens 128 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill
```

This uses synchronous BatchEngine. Each output ends with its stopping reason, generated length, and free-slot count. The preserved [bilingual output](results/2026-10-24/qwen35-2b-generation-npu-fp32.txt) stops both requests at EOS. Each request reserves `prompt length + max_new_tokens - 1` logical slots; cache capacity does not automatically grow with free device memory.

To use cross-round execution with your own text, add `--pipeline` to the generation command above. It currently requires FP32 and is separate from `--deferred` and `--prepare-during-step`. Add `--stagger-prompts` to submit one more text per collected result, and `--trace-rounds` to print rounds and observed cache ownership. These observations happen before CPU commit; release messages report when ownership changes are noticed, without device timestamps. Trace output uses English labels.

You can try that frontend without weights:

```bash
python -m minideepinfer.verify_pipeline_text --device cpu --trace-rounds \
  --output outputs/pipeline-text-cpu.json
```

The earlier fixed cross-round checker remains available:

```bash
python -m minideepinfer.verify_chunk_pipeline \
  --model weights/Qwen3.5-2B --device npu:0 \
  --output outputs/chunk-pipeline-npu.json
```

This runs reference and experimental trajectories. To inspect existing evidence instead of repeating device computation:

```bash
python scripts/check_evidence.py
```

The index selects either a recorded immutable Git revision or current files. A historical match does not establish a new hardware run. Use `--index docs/results/v0.3/evidence.json` for the speculation reports. See [evidence handling](EVIDENCE.md) for the rules and Git-history requirements.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing torch | Install model dependencies in a CPU environment, or use your configured NPU environment |
| Missing `minideepinfer.model_draft_demo` | Use v0.4.0 or newer source with the steps above, reinstall, and check the package version printed by the same Python |
| torch_npu import failure / NPU unavailable | Check matching device dependencies, drivers, and container device mappings |
| Whole prompt exceeds budget | Enable `--chunked-prefill`; each request must still fit in cache |
| Request exceeds capacity | Check prompt length plus output limit; logical slots are not device MiB |
| Old Transformers cannot load Qwen3.5 tokenizer | Use 5.12.1 for this example; 4.46.3 covers the older Qwen2 regression |
| BF16 generation changes across chunks | Read the [scan/state precision notes](BF16_REFERENCE.md); v0.1 defaults to FP32 |

Run `python -m unittest discover -s tests -v` from source. Model tests explicitly skip without their dependencies. The installable package contains runtime modules; lessons, tests, scripts, and reports live in the source repository. Install from the repository or [GitHub Releases](https://github.com/AkiYangDev/mini-deepinfer/releases); the project is not published on PyPI.
