"""Compare local checkpoint logits with Transformers and full recomputation."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=4, help="fixed-token decode steps")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.prefill_chunk_size is not None and args.prefill_chunk_size < 1:
        parser.error("prefill-chunk-size must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import Step
    from .model_runner import ModelRunner

    dtype = getattr(torch, args.dtype)
    atol = args.atol if args.atol is not None else (0.125 if dtype == torch.bfloat16 else 1e-4)
    rtol = args.rtol if args.rtol is not None else (0.01 if dtype == torch.bfloat16 else 2e-4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = tokenizer.encode("The capital of France is")
    forced = tokenizer.encode(" Paris. The answer is clear.", add_special_tokens=False)
    if len(forced) < args.steps:
        parser.error("steps exceeds the fixed continuation length")
    capacity = len(prompt) + args.steps
    runner = ModelRunner.from_pretrained(args.model, args.device, dtype, capacity)
    reference = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager", local_files_only=True
    ).to(runner.device).eval()
    rows = []
    reference_past = None

    def compare(actual, expected):
        actual, expected = actual.float().cpu(), expected.float().cpu()
        delta = actual - expected
        return {"passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
                "max_abs": float(delta.abs().max()),
                "rms": float(delta.square().mean().sqrt()),
                "top1_agreement": float((actual.argmax(-1) == expected.argmax(-1)).float().mean())}

    size = args.prefill_chunk_size or len(prompt)
    chunks = [prompt[start:start + size] for start in range(0, len(prompt), size)]
    work = [("prefill", chunk, index == len(chunks) - 1) for index, chunk in enumerate(chunks)]
    work += [("decode", [token], True) for token in forced[:args.steps]]
    prefix = []
    with torch.inference_mode():
        for phase, input_ids, last_chunk in work:
            start = len(prefix)
            prefix.extend(input_ids)
            length = len(prefix)
            step = Step("fixed", phase, tuple(input_ids), tuple(range(start, length)),
                        tuple(range(start, length)), tuple(range(length)), last_chunk)
            actual = runner.forward_batch((step,))[0] if args.prefill_chunk_size else runner.forward(step)
            expected = reference(torch.tensor([prefix], device=runner.device), use_cache=False).logits[0]
            cached_reference = reference(torch.tensor([input_ids], device=runner.device),
                                         past_key_values=reference_past, use_cache=True)
            reference_past = cached_reference.past_key_values
            full = runner.model(runner.prepare_input(Step(
                "recompute", "prefill", tuple(prefix), tuple(range(length)),
                tuple(range(length)), tuple(range(length)))),
                type(runner.cache)(runner.model.config, capacity, runner.device, dtype))
            selection = slice(-len(input_ids), None)
            rows.append({"phase": step.phase, "context_tokens": length,
                         "input_ids": input_ids, "is_last_prompt_chunk": last_chunk,
                         "reference": compare(actual, expected[selection]),
                         "reference_cached": compare(actual, cached_reference.logits[0]),
                         "full_recompute": compare(actual, full[selection]),
                         "reference_cache_recompute": compare(cached_reference.logits[0], expected[selection])})
        runner.synchronize()
    runner.release("fixed")
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    passed = all(row[k]["passed"] for row in rows
                 for k in ("reference", "reference_cached", "full_recompute", "reference_cache_recompute"))
    root = Path(__file__).parent
    sources = {"minideepinfer/" + str(path.relative_to(root)).replace("\\", "/"):
               hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (root / "model_runner.py", root / "models/qwen2.py",
                            root / "tensor_cache.py", Path(__file__))}
    report = {"passed": passed, "device": str(runner.device), "dtype": args.dtype,
              "atol": atol, "rtol": rtol, "packages": packages, "steps": rows,
              "prompt_ids": prompt, "forced_ids": forced[:args.steps],
              "prefill_chunk_size": args.prefill_chunk_size, "source_sha256": sources}
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
