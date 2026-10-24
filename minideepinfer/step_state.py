"""Lesson five: batch remapping and resource lifetime, without device execution.

The caller supplies events covering ALL device uses of a state's resources.
See docs/STEP_STATE.md for the DeepInfer source and teaching simplifications.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .kv_cache import SlotPool


class Completion(Protocol):
    def query(self) -> bool:
        """True only after every device use covered by this event is finished."""
        ...


class RetainedState(Protocol):
    request_ids: tuple[str, ...]

    @property
    def closed(self) -> bool: ...

    @property
    def releasable(self) -> bool: ...


def batch_remap(previous: tuple[str, ...], current: tuple[str, ...]) -> tuple[int, ...]:
    """For each current row, find its previous row; -1 means a new request."""
    for ids in (previous, current):
        if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
            raise ValueError("batch IDs must be distinct nonempty strings")
    rows = {request_id: i for i, request_id in enumerate(previous)}
    return tuple(rows.get(request_id, -1) for request_id in current)


@dataclass(eq=False)
class StepState:
    request_ids: tuple[str, ...]
    sampled_tokens: tuple[int, ...]
    ready: Completion
    last_use: Completion
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.request_ids = tuple(self.request_ids)
        self.sampled_tokens = tuple(self.sampled_tokens)
        batch_remap((), self.request_ids)
        if not self.request_ids or len(self.sampled_tokens) != len(self.request_ids) or any(
            type(token) is not int or token < 0 for token in self.sampled_tokens
        ):
            raise ValueError("each request must have one nonnegative sampled token")

    def inputs_for(self, current: tuple[str, ...], fresh: dict[str, int]) -> tuple[int, ...]:
        """Read old samples by request ID and use supplied input tokens for new rows."""
        if self._closed:
            raise RuntimeError("this state's host reader is closed")
        if not self.ready.query():
            raise RuntimeError("sampled tokens are not ready")
        mapping = batch_remap(self.request_ids, current)
        tokens = tuple(self.sampled_tokens[row] if row >= 0 else fresh[rid]
                       for rid, row in zip(current, mapping))
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("fresh inputs must be nonnegative integer tokens")
        return tokens

    def close(self) -> None:
        """Host code is done reading, or has abandoned this result after cancellation."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def releasable(self) -> bool:
        return self._closed and self.ready.query() and self.last_use.query()


class DeferredRelease:
    """Retire requests now; return slots only after every retained state is done.

    All calls are single-threaded. Retain before launch, and never launch new work
    for a retired request. The callback must complete without raising.
    """

    def __init__(self, pool: SlotPool, release: Callable[[str], None]):
        self.pool = pool
        self.release = release
        self._states: list[RetainedState] = []
        self._retired: set[str] = set()
        self._released: set[str] = set()

    def retain(self, state: RetainedState) -> None:
        if state.closed or any(item is state for item in self._states):
            raise ValueError("retain an open state exactly once before launching work")
        if (self._retired | self._released).intersection(state.request_ids):
            raise ValueError("cannot launch work for a retired request")
        self._states.append(state)

    def retire(self, request_id: str) -> None:
        if request_id not in self._released:
            self._retired.add(request_id)

    def collect(self) -> tuple[str, ...]:
        self._states = [state for state in self._states if not state.releasable]
        busy = {rid for state in self._states for rid in state.request_ids}
        released = tuple(sorted(self._retired - busy))
        for request_id in released:
            self.release(request_id)
            self.pool.release(request_id)
            self._retired.remove(request_id)
            self._released.add(request_id)
        return released
