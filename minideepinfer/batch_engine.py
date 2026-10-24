"""Iteration-level scheduling: one batch execution, then one result commit.

See docs/SOURCE_MAP.md for the DeepInfer reference and teaching simplifications.
Request metadata and validation are shared with the first lesson's Engine.
"""

from collections import deque
from typing import Iterator, Protocol

from .engine import Request, Status, Step, StepOutput, validate_request
from .kv_cache import SlotPool


class BatchRunner(Protocol):
    def run_batch(self, steps: tuple[Step, ...]) -> tuple[int | None, ...]:
        """Return tokens (None for partial prefill); finish device work even on error."""
        ...

    def release(self, request_id: str) -> None:
        """Discard host prefix metadata after device work has completed."""
        ...


class BatchResult(Protocol):
    releasable: bool

    def wait(self): ...
    def result(self) -> tuple[int | None, ...]: ...
    def cancel(self, request_id: str): ...
    def close(self): ...


class DeferredBatchRunner(Protocol):
    pending: BatchResult | None

    def launch(self, steps: tuple[Step, ...]) -> BatchResult: ...
    def release(self, request_id: str): ...


class BatchEngine:
    """FIFO admission, optional prompt chunks, and decode-first token budgeting.

    Requests reserve their maximum KV footprint before admission. A waiting head
    that does not fit holds up later requests, but existing decodes keep running.
    All public methods run on one CPU thread. Deferred execution allows submit
    and cancel between launch_step and finish_step, with one batch in flight.
    """

    def __init__(self, runner: BatchRunner | DeferredBatchRunner, cache_capacity=128,
                 max_num_seqs=4, max_num_batched_tokens=None, chunked_prefill=False,
                 *, deferred=False):
        budget = cache_capacity if max_num_batched_tokens is None else max_num_batched_tokens
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError("max_num_seqs must be a positive integer")
        if type(budget) is not int or budget < max_num_seqs:
            raise ValueError("token budget must be an integer >= max_num_seqs")
        if type(chunked_prefill) is not bool:
            raise ValueError("chunked_prefill must be a boolean")
        if type(deferred) is not bool:
            raise ValueError("deferred must be a boolean")
        self.runner = runner
        self.cache = SlotPool(cache_capacity)
        self._requests: dict[str, Request] = {}
        self._waiting: deque[str] = deque()
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = budget
        self.chunked_prefill = chunked_prefill
        self._running: dict[str, tuple[int, ...]] = {}
        self.deferred = deferred
        self._plans: tuple[Step, ...] = ()
        self._pending: BatchResult | None = None

    def submit(self, prompt_ids, max_new_tokens=4, eos_token_id=None) -> Request:
        prompt = validate_request(prompt_ids, max_new_tokens, eos_token_id, self.cache.capacity)
        if not self.chunked_prefill and len(prompt) > self.max_num_batched_tokens:
            raise ValueError("whole prompt exceeds the batch token budget; enable chunked_prefill")
        request = Request(f"req-{len(self._requests)}", prompt, max_new_tokens, eos_token_id)
        self._requests[request.request_id] = request
        self._waiting.append(request.request_id)
        return request

    def cancel(self, request_id: str) -> bool:
        request = self._requests[request_id]
        if request.status not in (Status.WAITING, Status.RUNNING):
            return False
        request.status, request.finish_reason = Status.CANCELLED, "cancelled"
        if self._plans:
            if any(plan.request_id == request_id for plan in self._plans):
                self._pending.cancel(request_id)
            # Keep all retired running state until this batch is safely closed.
        elif request_id in self._running or not self.deferred:
            self._release(request_id)
        return True

    def _schedule(self) -> tuple[Step, ...]:
        plans = []
        for request_id, slots in self._running.items():
            request = self._requests[request_id]
            if request.num_cached_tokens >= len(request.prompt_ids):
                plans.append(self._plan(request, slots, 1))
        budget = self.max_num_batched_tokens - len(plans)
        # Existing decodes get a turn before any prompt consumes the remaining budget.
        for request_id, slots in self._running.items():
            request = self._requests[request_id]
            if request.num_cached_tokens < len(request.prompt_ids) and budget:
                plan = self._plan(request, slots, budget)
                plans.append(plan)
                budget -= len(plan.input_ids)
        while self._waiting and len(self._running) < self.max_num_seqs and budget:
            request = self._requests[self._waiting[0]]
            if request.status == Status.CANCELLED:
                self._waiting.popleft()
                continue
            required = len(request.prompt_ids) + request.max_new_tokens - 1
            if required > self.cache.free_count or (not self.chunked_prefill and len(request.prompt_ids) > budget):
                break
            self._waiting.popleft()
            self._running[request.request_id] = self.cache.reserve(request.request_id, required)
            request.status = Status.RUNNING
            plan = self._plan(request, self._running[request.request_id], budget)
            plans.append(plan)
            budget -= len(plan.input_ids)
        return tuple(plans)

    def _plan(self, request, slots, budget):
        start = request.num_cached_tokens
        prefill = start < len(request.prompt_ids)
        end = min(len(request.prompt_ids), start + budget) if prefill else start + 1
        inputs = request.prompt_ids[start:end] if prefill else (request.output_ids[-1],)
        return Step(request.request_id, "prefill" if prefill else "decode", inputs,
                    tuple(range(start, end)), slots[start:end], slots[:end],
                    end >= len(request.prompt_ids))

    def step(self) -> tuple[StepOutput, ...]:
        if self.deferred:
            return self.finish_step() if self.launch_step() else ()
        plans = self._schedule()
        if not plans:
            return ()
        try:
            tokens = tuple(self.runner.run_batch(plans))
            self._validate(plans, tokens)
        except Exception as error:
            # No partial output commit. The runner has already synchronized.
            self._fail(plans, error)
            for plan in plans:
                self._release(plan.request_id)
            raise
        return self._commit(plans, tokens)

    def launch_step(self) -> tuple[Step, ...]:
        """Choose and launch one batch; request progress still describes the last commit."""
        if not self.deferred:
            raise RuntimeError("launch_step requires a deferred runner")
        if self._plans:
            raise RuntimeError("finish or abort the pending step before launching another")
        self._plans = self._schedule()
        if not self._plans:
            return ()
        try:
            self._pending = self.runner.launch(self._plans)
        except Exception as error:
            # A failed drain leaves an owned handle; never return its slots early.
            self._pending = self.runner.pending
            self._fail(self._plans, error)
            if self._pending is None or self._pending.releasable:
                self._discard()
            raise
        return self._plans

    def finish_step(self) -> tuple[StepOutput, ...]:
        """Wait, validate the whole result, then commit request progress and cleanup."""
        if self._pending is None:
            raise RuntimeError("there is no pending step")
        try:
            self._pending.wait()
            tokens = tuple(self._pending.result())
            self._validate(self._plans, tokens)
        except Exception as error:
            self._fail(self._plans, error)
            if self._pending.releasable:
                self._discard()
            raise
        outputs = self._commit(self._plans, tokens)
        self._clear_pending()
        return outputs

    def abort_step(self) -> bool:
        """Abandon this batch, drain it, then fail its requests and reclaim state."""
        if not self._plans:
            return False
        error = RuntimeError("batch abandoned before result commit")
        try:
            self._pending.close()
            if not self._pending.releasable:
                raise RuntimeError("cannot confirm completion; keep pending resources")
        except Exception as failure:
            self._fail(self._plans, failure)
            raise
        self._fail(self._plans, error)
        self._discard()
        return True

    def _validate(self, plans, tokens):
        if len(tokens) != len(plans) or any(
            (t is not None if self._requests[p.request_id].status == Status.CANCELLED
             or not p.is_last_prompt_chunk else type(t) is not int or t < 0)
            for p, t in zip(plans, tokens)
        ):
            raise ValueError("runner must return tokens for sampling rows and None for partial/cancelled rows")

    def _commit(self, plans, tokens):
        outputs = []
        for plan, token in zip(plans, tokens):
            request = self._requests[plan.request_id]
            if request.status == Status.CANCELLED:
                outputs.append(StepOutput(plan.request_id, plan.phase, None,
                                          request.num_cached_tokens, "cancelled"))
                continue
            request.num_cached_tokens += len(plan.input_ids)
            if token is not None:
                request.output_ids.append(token)
                if token == request.eos_token_id:
                    request.finish_reason = "eos"
                elif len(request.output_ids) == request.max_new_tokens:
                    request.finish_reason = "length"
            if request.finish_reason is not None:
                request.status = Status.FINISHED
                self._release(plan.request_id)
            outputs.append(StepOutput(plan.request_id, plan.phase, token,
                                      request.num_cached_tokens, request.finish_reason))
        self._release_cancelled()
        return tuple(outputs)

    def _fail(self, plans, error):
        for plan in plans:
            request = self._requests[plan.request_id]
            if request.status != Status.CANCELLED:
                request.status = Status.FAILED
            request.error = f"{type(error).__name__}: {error}"

    def _discard(self):
        for plan in self._plans:
            self._release(plan.request_id)
        self._release_cancelled()
        self._clear_pending()

    def _release_cancelled(self):
        for request_id in tuple(self._running):
            if self._requests[request_id].status == Status.CANCELLED:
                self._release(request_id)

    def _clear_pending(self):
        self._plans, self._pending = (), None

    def run(self) -> Iterator[StepOutput]:
        while outputs := self.step():
            yield from outputs

    def _release(self, request_id):
        self.runner.release(request_id)
        self.cache.release(request_id)
        self._running.pop(request_id, None)
