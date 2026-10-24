"""Synchronous counterpart of DeepInfer's prepare / forward / sample stages.

See docs/SOURCE_MAP.md for the upstream revision and the omitted mechanisms.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .engine import Step
from .models.qwen2 import Qwen2ForCausalLM
from .tensor_cache import TensorKVCache
from .hybrid_cache import HybridCache


def select_device(name):
    if name.startswith("npu"):
        import torch_npu  # Registers the device; use a version matching PyTorch/CANN.

        if not torch.npu.is_available():
            raise RuntimeError("the requested NPU is unavailable")
        torch.npu.set_device(name)
        # Reference execution: FP32 checks must not silently use HF32 kernels.
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    elif name != "cpu":
        raise ValueError("choose cpu or an explicit npu device, such as npu:0")
    return torch.device(name)


@dataclass(frozen=True)
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_slots: torch.Tensor
    attention_mask: torch.Tensor | None = None  # True blocks a query/key pair.
    steps: tuple[Step, ...] = ()  # Host metadata: boundaries and state owners.


@dataclass(frozen=True)
class PrefixState:
    slots: tuple[int, ...]
    prompt_complete: bool


class ModelRunner:
    def __init__(self, model, cache_capacity=512):
        self.model = model.eval()
        weight = next(model.parameters())
        self.device = weight.device
        cache_type = HybridCache if hasattr(model.config, "layer_types") else TensorKVCache
        self.cache = cache_type(model.config, cache_capacity, self.device, weight.dtype)
        self._request_id = None
        self._cached_slots = ()
        self._prefixes: dict[str, PrefixState] = {}

    @classmethod
    def from_pretrained(cls, directory, device="npu:0", dtype=torch.bfloat16, cache_capacity=512):
        target = select_device(device)
        data = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
        model_type = data.get("model_type", "qwen2")
        if model_type in ("qwen3_5", "qwen3_5_text"):
            from .models.qwen3_5 import Qwen3_5ForCausalLM
            implementation = Qwen3_5ForCausalLM
        else:
            implementation = Qwen2ForCausalLM
        model = implementation.from_pretrained(directory, dtype=dtype).to(target)
        return cls(model, cache_capacity)

    def prepare_input(self, step: Step) -> ModelInput:
        prefix = self._cached_slots if step.request_id == self._request_id else None
        self._validate_step(step, prefix)
        return ModelInput(*(torch.tensor(value, dtype=torch.long, device=self.device)
                            for value in (step.input_ids, step.positions, step.slot_mapping, step.context_slots)),
                          steps=(step,))

    def _validate_step(self, step, prefix, allow_chunks=False):
        count = len(step.input_ids)
        if not count or len(step.positions) != count or len(step.slot_mapping) != count:
            raise ValueError("inputs, positions, and write slots must have the same nonzero length")
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in step.input_ids):
            raise ValueError("input token is outside the model vocabulary")
        start = step.positions[0]
        if type(step.is_last_prompt_chunk) is not bool:
            raise ValueError("is_last_prompt_chunk must be a boolean")
        if step.positions != tuple(range(start, start + count)) or start < 0:
            raise ValueError("positions must be contiguous and nonnegative")
        if start + count > self.model.config.max_position_embeddings:
            raise ValueError("input exceeds the model position limit")
        if len(step.context_slots) != start + count or step.context_slots[start:] != step.slot_mapping:
            raise ValueError("context slots must cover the prefix followed by the current write slots")
        if (len(set(step.context_slots)) != len(step.context_slots)
                or any(type(s) is not int or not 0 <= s < self.cache.capacity for s in step.context_slots)):
            raise ValueError("context slots must be unique and within cache capacity")
        if step.phase == "prefill":
            if start and (not allow_chunks or prefix is None or step.context_slots[:start] != prefix):
                raise ValueError("prefill chunk must extend this request's computed prefix")
            if not allow_chunks and not step.is_last_prompt_chunk:
                raise ValueError("partial prefill requires the batch execution path")
        elif step.phase == "decode":
            if (count != 1 or start == 0 or prefix is None or step.context_slots[:start] != prefix
                    or not step.is_last_prompt_chunk):
                raise ValueError("decode must extend the current request's computed prefix")
        else:
            raise ValueError("unknown execution phase")

    def prepare_batch(self, steps: tuple[Step, ...]) -> ModelInput:
        ids = [step.request_id for step in steps]
        if not steps or len(set(ids)) != len(ids):
            raise ValueError("batch must contain distinct requests")
        for step in steps:
            state = self._prefixes.get(step.request_id)
            self._validate_step(step, state.slots if state else None, allow_chunks=True)
            if state and step.phase == "prefill" and (state.prompt_complete or step.positions[0] == 0):
                raise ValueError("release a live request before restarting its prefill")
            if state and step.phase == "decode" and not state.prompt_complete:
                raise ValueError("decode cannot precede the final prompt chunk")
        contexts = tuple(slot for step in steps for slot in step.context_slots)
        other_slots = {s for rid, state in self._prefixes.items() if rid not in ids for s in state.slots}
        if len(set(contexts)) != len(contexts) or other_slots.intersection(contexts):
            raise ValueError("batch requests must not alias another request's KV slots")
        values = (tuple(t for step in steps for t in step.input_ids),
                  tuple(p for step in steps for p in step.positions),
                  tuple(s for step in steps for s in step.slot_mapping), contexts)
        tensors = tuple(torch.tensor(v, dtype=torch.long, device=self.device) for v in values)
        query_ids = torch.tensor([i for i, step in enumerate(steps) for _ in step.input_ids], device=self.device)
        key_ids = torch.tensor([i for i, step in enumerate(steps) for _ in step.context_slots], device=self.device)
        key_positions = torch.tensor([p for step in steps for p in range(len(step.context_slots))],
                                     device=self.device)
        # Packed reference attention: isolate sequences, then apply each one's causal mask.
        mask = ((query_ids[:, None] != key_ids[None, :])
                | (tensors[1][:, None] < key_positions[None, :]))
        return ModelInput(*tensors, mask, steps)

    def execute_main_model(self, inputs):
        return self.model(inputs, self.cache)

    def execute_sample(self, logits):
        return int(logits[-1].argmax().item())

    def synchronize(self):
        if self.device.type == "npu":
            torch.npu.synchronize(self.device)

    @torch.inference_mode()
    def forward(self, step: Step):
        """Expose logits for fixed-token numerical comparisons."""
        inputs = self.prepare_input(step)
        if isinstance(self.cache, HybridCache) and step.phase == "prefill":
            if self._request_id is not None:
                self.cache.release(self._request_id)
            self.cache.release(step.request_id)
        logits = self.execute_main_model(inputs)
        self._request_id, self._cached_slots = step.request_id, step.context_slots
        return logits

    @torch.inference_mode()
    def run(self, step: Step) -> int:
        try:
            return self.execute_sample(self.forward(step))
        finally:
            # Engine may reclaim slots immediately, including after a model error.
            self.synchronize()

    @torch.inference_mode()
    def forward_batch(self, steps: tuple[Step, ...]):
        inputs = self.prepare_batch(steps)
        logits = self.execute_main_model(inputs)  # One model call for the whole batch.
        for step in steps:
            self._prefixes[step.request_id] = PrefixState(step.context_slots, step.is_last_prompt_chunk)
        return logits.split([len(step.input_ids) for step in steps])

    @torch.inference_mode()
    def run_batch(self, steps: tuple[Step, ...]) -> tuple[int | None, ...]:
        try:
            logits = self.forward_batch(steps)
            # One host transfer; row order is the scheduler's request order.
            rows = [row[-1] for step, row in zip(steps, logits) if step.is_last_prompt_chunk]
            tokens = iter(torch.stack(rows).argmax(-1).tolist() if rows else ())
            return tuple(next(tokens) if step.is_last_prompt_chunk else None for step in steps)
        finally:
            self.synchronize()

    def release(self, request_id: str) -> None:
        self._prefixes.pop(request_id, None)
        if isinstance(self.cache, HybridCache):
            self.cache.release(request_id)
        if request_id == self._request_id:
            self._request_id, self._cached_slots = None, ()
