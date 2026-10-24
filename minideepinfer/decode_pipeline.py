"""Fixed membership and length: forward this round before reading the last one.

This experiment exclusively owns an idle ModelRunner. It has no EOS, arrivals,
cancellation, speculative tokens, or rollback. BatchEngine remains unchanged.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from .deferred_runner import _CPUEvent
from .engine import Step
from .kv_cache import SlotPool
from .model_runner import PrefixState
from .npu_events import RecordedEvent
from .step_state import batch_remap


@dataclass
class _Round:
    index: int
    steps: tuple
    ready: RecordedEvent
    buffers: list = field(default_factory=list)
    logits: tuple = ()
    tokens: torch.Tensor | None = None
    host_tokens: torch.Tensor | None = None

    @property
    def request_ids(self):
        return tuple(s.request_id for s in self.steps)


class FixedDecodePipeline:
    def __init__(self, runner):
        if runner.device.type not in ("cpu", "npu"):
            raise ValueError("choose CPU or NPU")
        self.runner = runner
        self.pool = SlotPool(runner.cache.capacity)
        self.is_npu = runner.device.type == "npu"
        self.stream = torch.npu.Stream(device=runner.device) if self.is_npu else None
        self._states = []
        self._slots = {}
        self._active = False

    def _event(self):
        return RecordedEvent(torch.npu.Event() if self.is_npu else _CPUEvent())

    @torch.inference_mode()
    def run(self, prompts, max_new_tokens, *, orders=None, on_result=None):
        """Generate exactly this many greedy tokens, even if a token equals EOS.

        prompts maps request IDs to token tuples. Each optional orders entry is
        a permutation of all IDs, one per round (prefill counts as round zero).
        on_result(index, steps, tokens, logits) runs after that round's D2H event;
        decode Steps contain placeholders, while the model consumes device tokens.
        """
        if self._active or self.runner._prefixes or self.runner._request_id is not None:
            raise RuntimeError("pipeline requires an idle, exclusively owned runner")
        prompts = {rid: tuple(tokens) for rid, tokens in prompts.items()}
        ids = tuple(prompts)
        if (not ids or any(not isinstance(rid, str) or not rid or not p for rid, p in prompts.items())
                or type(max_new_tokens) is not int or max_new_tokens < 1):
            raise ValueError("provide nonempty prompts and a positive output length")
        for p in prompts.values():
            if (any(type(t) is not int or not 0 <= t < self.runner.model.config.vocab_size for t in p)
                    or len(p) + max_new_tokens - 1 > self.runner.model.config.max_position_embeddings):
                raise ValueError("prompt token or requested position is outside the model limits")
        orders = tuple(tuple(order) for order in orders) if orders is not None else (ids,) * max_new_tokens
        if (len(orders) != max_new_tokens
                or any(len(order) != len(ids) or set(order) != set(ids) for order in orders)):
            raise ValueError("each round must contain every request exactly once")
        if sum(len(p) + max_new_tokens - 1 for p in prompts.values()) > self.pool.capacity:
            raise MemoryError("the complete fixed batch does not fit in cache")
        self._active = True
        outputs = {rid: [] for rid in ids}
        previous = None
        try:
            for rid, p in prompts.items():
                self._slots[rid] = self.pool.reserve(rid, len(p) + max_new_tokens - 1)
            if self.is_npu:
                self.stream.wait_stream(torch.npu.current_stream(self.runner.device))
            with torch.npu.stream(self.stream) if self.is_npu else nullcontext():
                for index, order in enumerate(orders):
                    steps = tuple(self._step(rid, prompts[rid], index) for rid in order)
                    current = _Round(index, steps, self._event())
                    self._states.append(current)  # Retain ownership before submission.
                    device_tokens = None
                    if previous is not None:
                        remap = batch_remap(previous.request_ids, current.request_ids)
                        rows = torch.tensor(remap, dtype=torch.long, device=self.runner.device)
                        current.buffers.extend((previous.tokens, rows))
                        device_tokens = previous.tokens.index_select(0, rows)
                        current.buffers.append(device_tokens)
                    inputs = self.runner.prepare_batch(steps, device_input_ids=device_tokens)
                    current.buffers.append(inputs)
                    packed = self.runner.execute_main_model(inputs)
                    current.buffers.append(packed)
                    current.logits = packed.split([len(s.input_ids) for s in steps])
                    # These prefixes describe submitted cache writes, not CPU outputs.
                    for step in steps:
                        self.runner._prefixes[step.request_id] = PrefixState(step.context_slots, True)
                    if previous is not None:
                        self._finish(previous, outputs, on_result)
                        self._states.pop(0)
                    self._sample(current)
                    previous = current
                self._finish(previous, outputs, on_result)  # No next forward: drain the tail.
            return outputs
        finally:
            self.close()

    def _step(self, rid, prompt, index):
        count = len(prompt) if index == 0 else 1
        start = 0 if index == 0 else len(prompt) + index - 1
        slots = self._slots[rid]
        return Step(rid, "prefill" if index == 0 else "decode", prompt if index == 0 else (0,),
                    tuple(range(start, start + count)), slots[start:start + count], slots[:start + count])

    def _sample(self, state):
        state.tokens = torch.stack([row[-1] for row in state.logits]).argmax(-1)
        state.host_tokens = torch.empty(len(state.steps), dtype=torch.long, pin_memory=self.is_npu)
        state.host_tokens.copy_(state.tokens, non_blocking=self.is_npu)
        state.ready.record(self.stream)

    def _finish(self, state, outputs, on_result):
        state.ready.synchronize()  # Covers this round, not the later forward.
        tokens = tuple(state.host_tokens.tolist())
        if any(not 0 <= t < self.runner.model.config.vocab_size for t in tokens):
            raise ValueError("invalid sampled token")
        if on_result is not None:
            on_result(state.index, state.steps, tokens, state.logits)
        for rid, token in zip(state.request_ids, tokens):
            outputs[rid].append(token)

    def close(self):
        """Drain before release. A failed drain retains buffers, slots, and ownership."""
        if not self._active:
            return
        if self.is_npu:
            self.stream.synchronize()
        for rid in self._slots:
            self.runner.release(rid)
            self.pool.release(rid)
        self._states.clear()
        self._slots.clear()
        self._active = False
