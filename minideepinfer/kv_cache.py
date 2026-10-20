"""Logical slot ownership only; tensor KV storage belongs to a future runner."""

import heapq


class SlotPool:
    """Reserve a request's maximum footprint before it starts computing."""

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._free = list(range(capacity))
        self._owned: dict[str, tuple[int, ...]] = {}

    @property
    def free_count(self) -> int:
        return len(self._free)

    def reserve(self, request_id: str, count: int) -> tuple[int, ...]:
        if request_id in self._owned:
            raise ValueError(f"{request_id} already owns slots")
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        if count > self.free_count:
            raise MemoryError(f"need {count} slots, only {self.free_count} free")
        slots = tuple(heapq.heappop(self._free) for _ in range(count))
        self._owned[request_id] = slots
        return slots

    def release(self, request_id: str) -> None:
        # Idempotent cleanup also covers cancellation before admission.
        for slot in self._owned.pop(request_id, ()):
            heapq.heappush(self._free, slot)
