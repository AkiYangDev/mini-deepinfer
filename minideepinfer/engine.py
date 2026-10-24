"""Synchronous FIFO: schedule -> prepare -> run -> commit -> release.

One request runs at a time. This module deliberately has no tensor dependency.
Conceptual DeepInfer source mapping and simplifications: docs/SOURCE_MAP.md.
FIFO and slot ownership are the baseline; ZOS/speculative state is not implemented.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Literal, Protocol, Sequence

from .kv_cache import SlotPool


class Status(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Request:
    request_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None = None
    output_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    num_cached_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None


def validate_request(prompt_ids, max_new_tokens, eos_token_id, capacity):
    prompt = tuple(prompt_ids)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_ids must contain nonnegative integer token IDs")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if eos_token_id is not None and (type(eos_token_id) is not int or eos_token_id < 0):
        raise ValueError("eos_token_id must be a nonnegative integer or None")
    # The final predicted token is returned, but never consumed by decode.
    required = len(prompt) + max_new_tokens - 1
    if required > capacity:
        raise ValueError(f"request needs {required} KV slots, capacity is {capacity}")
    return prompt


@dataclass(frozen=True)
class Step:
    request_id: str
    phase: Literal["prefill", "decode"]
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    slot_mapping: tuple[int, ...]  # Write locations for this step's inputs.
    context_slots: tuple[int, ...]  # Past KV plus this step, in sequence order.
    is_last_prompt_chunk: bool = True


class Runner(Protocol):
    def run(self, step: Step) -> int:
        """Consume inputs, write their KV, and return one selected next token.

        Synchronous contract: device work must be complete on return or error,
        so the engine can immediately reuse slots after finishing or failing.
        """
        ...


@dataclass(frozen=True)
class StepOutput:
    request_id: str
    phase: Literal["prefill", "decode"]
    token_id: int | None  # None means an unfinished prompt chunk, not an output token.
    num_cached_tokens: int
    finish_reason: str | None


class Engine:
    """Single-threaded engine; mutate request state only through this object."""

    def __init__(self, runner: Runner, cache_capacity: int = 128):
        self.runner = runner
        self.cache = SlotPool(cache_capacity)
        self._requests: dict[str, Request] = {}
        self._waiting: deque[str] = deque()
        self._active: Request | None = None
        self._slots: tuple[int, ...] = ()

    def submit(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 4,
        eos_token_id: int | None = None,
    ) -> Request:
        prompt = validate_request(prompt_ids, max_new_tokens, eos_token_id, self.cache.capacity)
        request = Request(f"req-{len(self._requests)}", prompt, max_new_tokens, eos_token_id)
        self._requests[request.request_id] = request
        self._waiting.append(request.request_id)
        return request

    def cancel(self, request_id: str) -> bool:
        request = self._requests[request_id]
        if request.status not in (Status.WAITING, Status.RUNNING):
            return False
        request.status = Status.CANCELLED
        request.finish_reason = "cancelled"
        if request is self._active:
            self._release_active()
        return True

    def step(self) -> StepOutput | None:
        if self._active is None:
            while self._waiting:
                request = self._requests[self._waiting.popleft()]
                if request.status == Status.CANCELLED:
                    continue
                self._slots = self.cache.reserve(
                    request.request_id, len(request.prompt_ids) + request.max_new_tokens - 1
                )
                request.status = Status.RUNNING
                self._active = request
                break
        if self._active is None:
            return None

        request = self._active
        phase = "prefill" if request.num_cached_tokens == 0 else "decode"
        inputs = request.prompt_ids if phase == "prefill" else (request.output_ids[-1],)
        start = request.num_cached_tokens
        end = start + len(inputs)
        plan = Step(
            request.request_id, phase, inputs, tuple(range(start, end)),
            self._slots[start:end], self._slots[:end],
        )
        try:
            token = self.runner.run(plan)
            if type(token) is not int or token < 0:
                raise ValueError("runner must return a nonnegative integer token ID")
        except Exception as error:
            request.status = Status.FAILED
            request.error = f"{type(error).__name__}: {error}"
            self._release_active()
            raise

        request.num_cached_tokens = end
        request.output_ids.append(token)
        if token == request.eos_token_id:
            request.finish_reason = "eos"
        elif len(request.output_ids) == request.max_new_tokens:
            request.finish_reason = "length"
        if request.finish_reason is not None:
            request.status = Status.FINISHED
            self._release_active()
        return StepOutput(request.request_id, phase, token, end, request.finish_reason)

    def run(self) -> Iterator[StepOutput]:
        while (output := self.step()) is not None:
            yield output

    def _release_active(self) -> None:
        assert self._active is not None
        release = getattr(self.runner, "release", None)
        if release is not None:
            release(self._active.request_id)
        self.cache.release(self._active.request_id)
        self._active = None
        self._slots = ()
