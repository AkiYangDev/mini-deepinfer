"""Synchronous counterpart of DeepInfer's prepare / forward / sample stages.

See docs/SOURCE_MAP.md for the upstream revision and the omitted mechanisms.
"""

from dataclasses import dataclass

import torch

from .engine import Step
from .models.qwen2 import Qwen2ForCausalLM
from .tensor_cache import TensorKVCache


def select_device(name):
    if name.startswith("npu"):
        import torch_npu  # Registers the device; use a version matching PyTorch/CANN.

        if not torch.npu.is_available():
            raise RuntimeError("the requested NPU is unavailable")
        torch.npu.set_device(name)
    elif name != "cpu":
        raise ValueError("choose cpu or an explicit npu device, such as npu:0")
    return torch.device(name)


@dataclass(frozen=True)
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_slots: torch.Tensor


class ModelRunner:
    def __init__(self, model, cache_capacity=512):
        self.model = model.eval()
        weight = next(model.parameters())
        self.device = weight.device
        self.cache = TensorKVCache(model.config, cache_capacity, self.device, weight.dtype)
        self._request_id = None
        self._cached_slots = ()

    @classmethod
    def from_pretrained(cls, directory, device="npu:0", dtype=torch.bfloat16, cache_capacity=512):
        target = select_device(device)
        model = Qwen2ForCausalLM.from_pretrained(directory, dtype=dtype).to(target)
        return cls(model, cache_capacity)

    def prepare_input(self, step: Step) -> ModelInput:
        count = len(step.input_ids)
        if not count or len(step.positions) != count or len(step.slot_mapping) != count:
            raise ValueError("inputs, positions, and write slots must have the same nonzero length")
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in step.input_ids):
            raise ValueError("input token is outside the model vocabulary")
        start = step.positions[0]
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
            if start != 0:
                raise ValueError("prefill must start at position zero")
        elif step.phase == "decode":
            if (count != 1 or step.request_id != self._request_id
                    or step.context_slots[:start] != self._cached_slots):
                raise ValueError("decode must extend the current request's computed prefix")
        else:
            raise ValueError("unknown execution phase")
        return ModelInput(*(torch.tensor(value, dtype=torch.long, device=self.device)
                            for value in (step.input_ids, step.positions, step.slot_mapping, step.context_slots)))

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
