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


class BatchEngine:
    """FIFO admission, optional prompt chunks, and decode-first token budgeting.

    Requests reserve their maximum KV footprint before admission. A waiting head
    that does not fit holds up later requests, but existing decodes keep running.
    All public methods are called between steps, on the same thread.
    """

    def __init__(self, runner: BatchRunner, cache_capacity=128,
                 max_num_seqs=4, max_num_batched_tokens=None, chunked_prefill=False):
        budget = cache_capacity if max_num_batched_tokens is None else max_num_batched_tokens
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError("max_num_seqs must be a positive integer")
        if type(budget) is not int or budget < max_num_seqs:
            raise ValueError("token budget must be an integer >= max_num_seqs")
        if type(chunked_prefill) is not bool:
            raise ValueError("chunked_prefill must be a boolean")
        self.runner = runner
        self.cache = SlotPool(cache_capacity)
        self._requests: dict[str, Request] = {}
        self._waiting: deque[str] = deque()
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = budget
        self.chunked_prefill = chunked_prefill
        self._running: dict[str, tuple[int, ...]] = {}

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
        plans = self._schedule()
        if not plans:
            return ()
        try:
            tokens = tuple(self.runner.run_batch(plans))
            if len(tokens) != len(plans) or any(
                (t is not None if not p.is_last_prompt_chunk else type(t) is not int or t < 0)
                for p, t in zip(plans, tokens)
            ):
                raise ValueError("runner must return a token for sampling rows and None for partial prefill")
        except Exception as error:
            # No partial output commit. The runner has already synchronized.
            for plan in plans:
                request = self._requests[plan.request_id]
                request.status = Status.FAILED
                request.error = f"{type(error).__name__}: {error}"
                self._release(plan.request_id)
            raise

        outputs = []
        for plan, token in zip(plans, tokens):
            request = self._requests[plan.request_id]
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
        return tuple(outputs)

    def run(self) -> Iterator[StepOutput]:
        while outputs := self.step():
            yield from outputs

    def _release(self, request_id):
        self.runner.release(request_id)
        self.cache.release(request_id)
        self._running.pop(request_id, None)
