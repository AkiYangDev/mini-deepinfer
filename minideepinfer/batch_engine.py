"""Iteration-level scheduling: one batch execution, then one result commit.

See docs/SOURCE_MAP.md for the DeepInfer reference and teaching simplifications.
Request metadata and validation are shared with the first lesson's Engine.
"""

from collections import deque
from typing import Iterator, Protocol

from .engine import Request, Status, Step, StepOutput, validate_request
from .kv_cache import SlotPool


class BatchRunner(Protocol):
    def run_batch(self, steps: tuple[Step, ...]) -> tuple[int, ...]:
        """Return tokens in step order; finish device work even when raising."""
        ...

    def release(self, request_id: str) -> None:
        """Discard host prefix metadata after device work has completed."""
        ...


class BatchEngine:
    """FIFO admission, whole prompts, and one decode token per running request.

    Requests reserve their maximum KV footprint before admission. A waiting head
    that does not fit holds up later requests, but existing decodes keep running.
    All public methods are called between steps, on the same thread.
    """

    def __init__(self, runner: BatchRunner, cache_capacity=128,
                 max_num_seqs=4, max_num_batched_tokens=None):
        budget = cache_capacity if max_num_batched_tokens is None else max_num_batched_tokens
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError("max_num_seqs must be a positive integer")
        if type(budget) is not int or budget < max_num_seqs:
            raise ValueError("token budget must be an integer >= max_num_seqs")
        self.runner = runner
        self.cache = SlotPool(cache_capacity)
        self._requests: dict[str, Request] = {}
        self._waiting: deque[str] = deque()
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = budget
        self._running: dict[str, tuple[int, ...]] = {}

    def submit(self, prompt_ids, max_new_tokens=4, eos_token_id=None) -> Request:
        prompt = validate_request(prompt_ids, max_new_tokens, eos_token_id, self.cache.capacity)
        if len(prompt) > self.max_num_batched_tokens:
            raise ValueError("whole prompt exceeds the batch token budget; chunking is not implemented")
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
        budget = self.max_num_batched_tokens - len(self._running)
        while self._waiting and len(self._running) < self.max_num_seqs:
            request = self._requests[self._waiting[0]]
            if request.status == Status.CANCELLED:
                self._waiting.popleft()
                continue
            required = len(request.prompt_ids) + request.max_new_tokens - 1
            if required > self.cache.free_count or len(request.prompt_ids) > budget:
                break
            self._waiting.popleft()
            self._running[request.request_id] = self.cache.reserve(request.request_id, required)
            request.status = Status.RUNNING
            budget -= len(request.prompt_ids)

        plans = []
        for request_id, slots in self._running.items():
            request = self._requests[request_id]
            start = request.num_cached_tokens
            inputs = request.prompt_ids if start == 0 else (request.output_ids[-1],)
            end = start + len(inputs)
            plans.append(Step(request_id, "prefill" if start == 0 else "decode", inputs,
                              tuple(range(start, end)), slots[start:end], slots[:end]))
        return tuple(plans)

    def step(self) -> tuple[StepOutput, ...]:
        plans = self._schedule()
        if not plans:
            return ()
        try:
            tokens = tuple(self.runner.run_batch(plans))
            if len(tokens) != len(plans) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("runner must return one nonnegative integer token per request")
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
