"""Tensor-free step construction shared by committed and issued progress."""

from dataclasses import dataclass

from .engine import Step


def plan_step(request, slots, start, budget, decode_token=0):
    """Plan from an explicit cache position; the caller chooses its clock.

    The synchronous engine supplies a committed token. The device pipeline uses
    a zero placeholder and replaces it with its previous device sample.
    """
    if type(start) is not int or start < 0 or type(budget) is not int or budget < 1:
        raise ValueError("cache position and positive token budget must be integers")
    prefill = start < len(request.prompt_ids)
    end = min(len(request.prompt_ids), start + budget) if prefill else start + 1
    if end > len(slots):
        raise ValueError("step exceeds the request's reserved slots")
    inputs = request.prompt_ids[start:end] if prefill else (decode_token,)
    return Step(request.request_id, "prefill" if prefill else "decode", inputs,
                tuple(range(start, end)), slots[start:end], slots[:end], end >= len(request.prompt_ids))


@dataclass(frozen=True)
class IssuedProgress:
    cached_tokens: int = 0
    sampled_tokens: int = 0

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.cached_tokens, self.sampled_tokens)):
            raise ValueError("issued progress must contain nonnegative integers")

    def advance(self, step):
        count = len(step.input_ids)
        if (not count or step.positions != tuple(range(self.cached_tokens, self.cached_tokens + count))
                or len(step.context_slots) != self.cached_tokens + count):
            raise ValueError("issued step must extend this progress contiguously")
        return IssuedProgress(self.cached_tokens + count,
                              self.sampled_tokens + int(step.is_last_prompt_chunk))
