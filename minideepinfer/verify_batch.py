"""Compare packed real-weight execution with single-request and HF references."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def packed_reference_cache():
    """Adapt HF cache order for numerical checks, not engine execution.

    HF appends all current queries after the old keys. Our reference layout
    groups keys by request. Reorder storage only; leave HF model math intact.
    """
    from transformers.cache_utils import DynamicCache

    class PackedCache(DynamicCache):
        permutation = None

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
            if self.permutation is not None:
                keys = keys.index_select(-2, self.permutation)
                values = values.index_select(-2, self.permutation)
                # HF 4.46 and 5.12 store cache layers differently.
                if hasattr(self, "layers"):
                    self.layers[layer_idx].keys, self.layers[layer_idx].values = keys, values
                else:
                    self.key_cache[layer_idx], self.value_cache[layer_idx] = keys, values
            return keys, values

    return PackedCache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import Step
    from .model_runner import ModelRunner

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = ["The capital of France is", "A KV cache"]
    prefixes = [tokenizer.encode(prompt) for prompt in prompts]
    original_ids = [p[:] for p in prefixes]
    forced = tokenizer.encode(" Paris. The answer is clear.", add_special_tokens=False)[:4]
    if len(forced) != 4:
        raise ValueError("tokenizer does not supply the four fixed continuation tokens")
    stride = len(prompts)
    capacity = stride * (max(map(len, prefixes)) + len(forced))
    slots = [tuple(range(i, capacity, stride)) for i in range(stride)]
    dtype = getattr(torch, args.dtype)
    runner = ModelRunner.from_pretrained(args.model, args.device, dtype, capacity)
    singles = [ModelRunner(runner.model, capacity) for _ in prompts]
    hybrid = hasattr(runner.model.config, "layer_types")
    reference_class = AutoModelForCausalLM
    if hybrid:
        from transformers import Qwen3_5ForCausalLM
        reference_class = Qwen3_5ForCausalLM
    reference = reference_class.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager", local_files_only=True
    ).to(runner.device).eval()
    past = [None] * stride
    packed_past = None if hybrid else packed_reference_cache()
    atol, rtol = (0.125, 0.01) if dtype == torch.bfloat16 else (2e-4, 2e-4)
    packed_atol = 0.125 if dtype == torch.bfloat16 else 1e-4

    def compare(actual, expected, tolerance=atol):
        actual, expected = actual.float().cpu(), expected.float().cpu()
        delta = actual - expected
        return {"passed": bool(torch.allclose(actual, expected, atol=tolerance, rtol=rtol)),
                "max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()),
                "top1_agreement": float((actual.argmax(-1) == expected.argmax(-1)).float().mean())}

    rows = []
    with torch.inference_mode():
        for iteration in range(5):
            plans = []
            for i, prefix in enumerate(prefixes):
                end = len(prefix)
                start = 0 if iteration == 0 else end - 1
                plans.append(Step(str(i), "prefill" if iteration == 0 else "decode",
                                  tuple(prefix[start:]), tuple(range(start, end)),
                                  slots[i][start:end], slots[i][:end]))
            batched = runner.forward_batch(tuple(plans))
            if iteration and not hybrid:
                old_lengths = [len(p) - 1 for p in prefixes]
                order, offset = [], 0
                for i, length in enumerate(old_lengths):
                    order.extend(range(offset, offset + length))
                    order.append(sum(old_lengths) + i)
                    offset += length
                packed_past.permutation = torch.tensor(order, device=runner.device)
            if not hybrid:
                # Independent reference mask: no reuse of the mini runner's mask.
                positions = [p for plan in plans for p in plan.positions]
                owners = [i for i, plan in enumerate(plans) for _ in plan.input_ids]
                keys = [(i, p) for i, prefix in enumerate(prefixes) for p in range(len(prefix))]
                allowed = torch.tensor([[owner == i and p <= pos for i, p in keys]
                                        for owner, pos in zip(owners, positions)], device=runner.device)
                mask = torch.zeros(allowed.shape, dtype=dtype, device=runner.device).masked_fill(
                    ~allowed, torch.finfo(dtype).min)
                packed = reference(torch.tensor([[t for plan in plans for t in plan.input_ids]], device=runner.device),
                                   attention_mask=mask[None, None],
                                   position_ids=torch.tensor([positions], device=runner.device),
                                   past_key_values=packed_past, use_cache=True).logits[0]
                packed_rows = packed.split([len(plan.input_ids) for plan in plans])
            for i, (plan, actual) in enumerate(zip(plans, batched)):
                single = singles[i].forward(plan)
                cached = reference(torch.tensor([plan.input_ids], device=runner.device),
                                   past_key_values=past[i], use_cache=True)
                past[i] = cached.past_key_values
                full = reference(torch.tensor([prefixes[i]], device=runner.device), use_cache=False).logits[0]
                expected = full[-len(plan.input_ids):]
                row = {"iteration": iteration, "request_id": plan.request_id,
                             "phase": plan.phase, "context_tokens": len(plan.context_slots),
                             "single_vs_batch": compare(actual, single),
                             "reference_cached": compare(actual, cached.logits[0]),
                             "reference_full": compare(actual, expected),
                             "reference_cache_recompute": compare(cached.logits[0], expected)}
                if not hybrid:
                    row.update(reference_packed=compare(actual, packed_rows[i], packed_atol),
                               reference_packed_vs_single=compare(packed_rows[i], cached.logits[0]))
                rows.append(row)
            if iteration < 4:
                for prefix in prefixes:
                    prefix.append(forced[iteration])
        runner.synchronize()
    for i, plan in enumerate(plans):
        runner.release(plan.request_id)
        singles[i].release(plan.request_id)
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).parent
    sources = {"minideepinfer/" + str(path.relative_to(root)).replace("\\", "/"):
               hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (root / "model_runner.py", root / "models/qwen2.py",
                            root / "tensor_cache.py", Path(__file__))}
    checks = ("single_vs_batch", "reference_packed", "reference_packed_vs_single",
              "reference_cached", "reference_full", "reference_cache_recompute")
    if hybrid:
        # A 4D attention mask does not isolate DeltaNet's recurrent scan. Keep
        # HF requests independent rather than create a misleading packed oracle.
        checks = ("single_vs_batch", "reference_cached", "reference_full", "reference_cache_recompute")
        sources.update({"minideepinfer/" + str(path.relative_to(root)).replace("\\", "/"):
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (root / "models/qwen3_5.py", root / "hybrid_cache.py")})
    report = {"passed": all(row[key]["passed"] for row in rows for key in checks),
              "device": str(runner.device), "dtype": args.dtype,
              "atol": atol, "packed_atol": packed_atol, "rtol": rtol,
              "packages": packages, "prompts": prompts, "prompt_ids": original_ids,
              "forced_ids": forced, "source_sha256": sources, "steps": rows,
              "remaining_prefixes": len(runner._prefixes)}
    if hybrid:
        report["reference_layout"] = "independent requests; no packed HF DeltaNet oracle"
        report["remaining_linear_states"] = len(runner.cache.linear_states)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
