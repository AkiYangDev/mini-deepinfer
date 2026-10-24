"""One in-flight batch: launch now, read and commit after the result copy.

Wrap an exclusively owned ModelRunner. The BatchEngine owns scheduling and
request progress; this adapter owns device completion and host prefixes.
"""

from contextlib import nullcontext

import torch

from .model_runner import PrefixState
from .npu_events import RecordedEvent


class _CPUEvent:
    def record(self, stream):
        pass

    def query(self):
        return True  # CPU work has finished when record is reached.

    def synchronize(self):
        pass


class PendingBatch:
    def __init__(self, owner, steps):
        self.owner, self.steps = owner, steps
        self.request_ids = tuple(step.request_id for step in steps)
        raw = torch.npu.Event() if owner.is_npu else _CPUEvent()
        self.ready = RecordedEvent(raw)
        self.cancelled = set()
        self._closed = self._safe_aborted = self._unsafe = False
        self._writes_started = False
        self._buffers = [owner.runner]
        self.host_tokens = torch.empty(len(steps), dtype=torch.long, pin_memory=owner.is_npu)
        self.logits = ()  # Diagnostic access also requires wait().

    def wait(self):
        if self.closed:
            raise RuntimeError("batch is already closed")
        try:
            self.ready.synchronize()
        except Exception:
            self._unsafe = self.owner._poisoned = True
            raise
        return self

    def cancel(self, request_id):
        if self.closed:
            raise RuntimeError("batch is already closed")
        if request_id not in self.request_ids:
            raise KeyError(request_id)
        self.cancelled.add(request_id)

    def result(self):
        return self.owner._commit(self)

    def close(self):
        """Abandon the whole batch, draining device work before clearing its state."""
        if not self.closed:
            self.owner._abort(self)

    @property
    def closed(self):
        return self._closed

    @property
    def releasable(self):
        return not self._unsafe and self.closed and (self._safe_aborted or self.ready.query())


class DeferredRunner:
    def __init__(self, runner):
        self.runner = runner
        self.is_npu = runner.device.type == "npu"
        if runner.device.type not in ("cpu", "npu"):
            raise ValueError("choose CPU or NPU")
        self.stream = torch.npu.Stream(device=runner.device) if self.is_npu else None
        self._pending = None
        self._poisoned = False
        self._cancelled = set()

    @property
    def pending(self):
        """Keep the handle reachable if launch fails before device work can drain."""
        return self._pending

    def _synchronize(self):
        if self.is_npu:
            self.stream.synchronize()

    @torch.inference_mode()
    def launch(self, steps):
        steps = tuple(steps)
        if self._poisoned or self._pending is not None:
            raise RuntimeError("finish the previous batch before launching another")
        if not steps or self._cancelled.intersection(s.request_id for s in steps):
            raise ValueError("batch must be nonempty and cancelled requests must be released")
        pending = PendingBatch(self, steps)
        self._pending = pending  # Own buffers before any model work is submitted.
        try:
            if self.is_npu:
                self.stream.wait_stream(torch.npu.current_stream(self.runner.device))
            with torch.npu.stream(self.stream) if self.is_npu else nullcontext():
                inputs = self.runner.prepare_batch(steps)
                pending._buffers.append(inputs)
                pending._writes_started = True
                packed = self.runner.execute_main_model(inputs)
                pending.logits = packed.split([len(s.input_ids) for s in steps])
                rows = [i for i, step in enumerate(steps) if step.is_last_prompt_chunk]
                tokens = torch.full((len(steps),), -1, dtype=torch.long, device=self.runner.device)
                if rows:
                    selected = torch.stack([pending.logits[i][-1] for i in rows]).argmax(-1)
                    indices = torch.tensor(rows, dtype=torch.long, device=self.runner.device)
                    tokens.index_copy_(0, indices, selected)
                    pending._buffers.extend((selected, indices))
                pending._buffers.extend((packed, tokens))
                pending.host_tokens.copy_(tokens, non_blocking=self.is_npu)
                pending.ready.record(self.stream)
            return pending
        except Exception:
            self._abort(pending)
            raise

    def _commit(self, pending):
        if self._poisoned or pending is not self._pending or pending.closed:
            raise RuntimeError("batch is not available for commit")
        if not pending.ready.query():
            raise RuntimeError("result copy is not ready; wait before reading")
        try:
            values = pending.host_tokens.tolist()
            if len(values) != len(pending.steps) or any((token < 0 if step.is_last_prompt_chunk else token != -1)
                   for step, token in zip(pending.steps, values)):
                raise ValueError("invalid sampled token or partial-prefill sentinel")
            outputs = []
            for step, token in zip(pending.steps, values):
                cancelled = step.request_id in pending.cancelled
                if not cancelled:
                    self.runner._prefixes[step.request_id] = PrefixState(
                        step.context_slots, step.is_last_prompt_chunk)
                outputs.append(token if step.is_last_prompt_chunk and not cancelled else None)
            self._cancelled.update(pending.cancelled)
            pending._closed = True
            self._pending = None
            return tuple(outputs)
        except Exception:
            self._abort(pending)
            raise

    def _abort(self, pending):
        if pending is not self._pending:
            raise RuntimeError("cannot abandon another batch")
        pending._unsafe = self._poisoned = True
        self._synchronize()  # On failure, keep ownership and block reuse.
        if pending._writes_started:
            for request_id in pending.request_ids:
                self.runner.release(request_id)
        pending._closed = pending._safe_aborted = True
        pending._unsafe = self._poisoned = False
        self._pending = None

    def release(self, request_id):
        if self._poisoned or (self._pending is not None and request_id in self._pending.request_ids):
            raise RuntimeError("cannot release state while its batch is pending")
        self.runner.release(request_id)
        self._cancelled.discard(request_id)
