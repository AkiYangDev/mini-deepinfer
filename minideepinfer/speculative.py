"""Single-request CPU lesson: greedy verification, commit, and logical KV rollback.

Controlled drafts are test inputs, not MTP or a learned draft model. Only the
existing full-attention Qwen2 implementation is supported; see SPECULATIVE.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    tokens: tuple[int, ...]
    accepted_draft: int
    final_role: str | None
    finish_reason: str | None


def verify_greedy(draft, target, remaining, eos_token_id=None):
    """target[i] predicts draft[i]; the last target row supplies the bonus."""
    draft, target = tuple(draft), tuple(target)
    if (type(remaining) is not int or remaining < 1 or len(draft) >= remaining
            or len(target) != len(draft)+1):
        raise ValueError('verification needs draft+1 target rows and room for a final token')
    if any(type(t) is not int or t < 0 for t in draft+target):
        raise ValueError('verification tokens must be nonnegative integers')
    if eos_token_id is not None and (type(eos_token_id) is not int or eos_token_id < 0):
        raise ValueError('EOS must be a nonnegative integer or None')
    accepted = 0
    for candidate, expected in zip(draft, target):
        if candidate != expected:
            break
        accepted += 1
        if candidate == eos_token_id:
            return Decision(draft[:accepted], accepted, None, 'eos')
    token = target[accepted]
    tokens = draft[:accepted]+(token,)
    reason = 'eos' if token == eos_token_id else 'length' if len(tokens)==remaining else None
    role = 'target' if not draft else 'bonus' if accepted==len(draft) else 'correction'
    return Decision(tokens, accepted, role, reason)


class CachedTarget:
    """Exclusive token slots; only tokens[:valid] may be read as past context."""

    def __init__(self, model, slots, capacity):
        import torch
        from .models.qwen2 import Qwen2ForCausalLM
        from .tensor_cache import TensorKVCache

        if not isinstance(model, Qwen2ForCausalLM) or next(model.parameters()).device.type != 'cpu':
            raise ValueError('this lesson requires the CPU full-attention Qwen2 implementation')
        if (not slots or len(set(slots)) != len(slots) or any(
                type(s) is not int or not 0 <= s < capacity for s in slots)):
            raise ValueError('slots must be unique and inside cache capacity')
        self.model, self.slots, self.tokens = model.eval(), tuple(slots), []
        self.cache = TensorKVCache(model.config, capacity, 'cpu', next(model.parameters()).dtype)
        self.forward_calls = self.input_tokens = 0

    def forward(self, tokens):
        import torch
        from .model_runner import ModelInput

        tokens = tuple(tokens)
        start, end = len(self.tokens), len(self.tokens)+len(tokens)
        if not tokens or end > len(self.slots) or end > self.model.config.max_position_embeddings:
            raise ValueError('nonempty input must fit reserved slots and model positions')
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in tokens):
            raise ValueError('input token is outside the vocabulary')
        tensors = (tokens, tuple(range(start,end)), self.slots[start:end], self.slots[:end])
        inputs = ModelInput(*(torch.tensor(v,dtype=torch.long,device='cpu') for v in tensors))
        with torch.inference_mode():
            logits = self.model(inputs,self.cache)
        self.tokens.extend(tokens)
        self.forward_calls += 1
        self.input_tokens += len(tokens)
        return logits

    def rollback(self, retained):
        if type(retained) is not int or not 0 <= retained <= len(self.tokens):
            raise ValueError('rollback must retain a prefix of computed inputs')
        invalidated = self.slots[retained:len(self.tokens)]
        del self.tokens[retained:]  # Tensor bytes remain; future writes replace them before reading.
        return invalidated


def generate(model, prompt_ids, max_new_tokens, draft, *, width=3, eos_token_id=None,
             cache_capacity=None, on_round=None):
    """draft(committed_prefix, width) supplies candidates after the pending token.

    The last committed output is pending: it has not yet been consumed by target.
    Each verify forward consumes pending+draft, then keeps only the valid prefix.
    This CPU-only function owns its cache and slots for the duration of one call.
    """
    from .engine import validate_request
    from .kv_cache import SlotPool

    prompt = tuple(prompt_ids)
    if type(width) is not int or not 0 <= width <= 8 or not callable(draft):
        raise ValueError('provide a draft callback and an integer width between 0 and 8')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    capacity = len(prompt)+max_new_tokens-1 if cache_capacity is None else cache_capacity
    pool = SlotPool(capacity)
    prompt = validate_request(prompt,max_new_tokens,eos_token_id,capacity)
    if (any(t >= model.config.vocab_size for t in prompt)
            or eos_token_id is not None and eos_token_id >= model.config.vocab_size
            or len(prompt)+max_new_tokens-1 > model.config.max_position_embeddings):
        raise ValueError('request exceeds the vocabulary or position limit')
    slots = pool.reserve('spec',len(prompt)+max_new_tokens-1)
    target, output, rounds = None, [], []
    proposed = accepted = corrections = bonuses = 0
    direct = 1  # The first output comes from prompt prefill, not draft verification.
    try:
        target = CachedTarget(model,slots,capacity)
        output.append(int(target.forward(prompt)[-1].argmax()))
        reason = 'eos' if output[-1]==eos_token_id else 'length' if max_new_tokens==1 else None
        while reason is None:
            remaining = max_new_tokens-len(output)
            count = min(width,remaining-1)
            candidates = tuple(draft(prompt+tuple(output),count))
            if len(candidates) != count or any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in candidates):
                raise ValueError('draft must return exactly the requested number of vocabulary IDs')
            before = len(target.tokens)
            inputs = (output[-1],)+candidates
            logits = target.forward(inputs)
            predictions = tuple(logits.argmax(-1).tolist())
            tentative = len(target.tokens)
            decision = verify_greedy(candidates,predictions,remaining,eos_token_id)
            output.extend(decision.tokens)
            retained = len(prompt)+len(output)-1
            invalidated = target.rollback(retained)
            if tuple(target.tokens) != (prompt+tuple(output))[:-1]:
                raise RuntimeError('retained cache must match committed tokens except the pending last token')
            proposed += len(candidates)
            accepted += decision.accepted_draft
            corrections += decision.final_role=='correction'
            bonuses += decision.final_role=='bonus'
            direct += decision.final_role=='target'
            record = dict(input_ids=list(inputs), draft=list(candidates), target=list(predictions), committed=list(decision.tokens),
                          accepted_draft=decision.accepted_draft, final_role=decision.final_role,
                          cached_before=before, tentative_cached=tentative, retained_cached=retained,
                          invalidated_slots=list(invalidated), pending_token=output[-1],
                          finish_reason=decision.finish_reason)
            rounds.append(record)
            if on_round is not None:
                on_round(record)
            reason = decision.finish_reason
        result = dict(output_ids=output,finish_reason=reason,rounds=rounds,
                      cached_tokens=len(target.tokens), proposed_draft_tokens=proposed,
                      accepted_draft_tokens=accepted, correction_tokens=corrections, bonus_tokens=bonuses,
                      direct_target_tokens=direct,
                      target_forward_calls=target.forward_calls,target_input_tokens=target.input_tokens)
    finally:
        pool.release('spec')  # CPU operators finish synchronously; no device work survives this scope.
    result['free_slots'] = pool.free_count
    return result
