"""Synchronous single-request verification and cache rollback reference.

Controlled drafts are test inputs, not MTP or a learned draft model. Only the
existing Qwen2/Qwen3.5 implementations are supported; see SPECULATIVE.md.
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
        from .models.qwen3_5 import Qwen3_5ForCausalLM
        from .hybrid_cache import HybridCache
        from .tensor_cache import TensorKVCache

        if not isinstance(model, (Qwen2ForCausalLM,Qwen3_5ForCausalLM)):
            raise ValueError('this reference requires Qwen2 or Qwen3.5')
        weight=next(model.parameters())
        if weight.device.type not in ('cpu','npu') or weight.dtype != torch.float32:
            raise ValueError('choose CPU or NPU with FP32 reference weights')
        if (not slots or len(set(slots)) != len(slots) or any(
                type(s) is not int or not 0 <= s < capacity for s in slots)):
            raise ValueError('slots must be unique and inside cache capacity')
        self.model, self.slots, self.tokens = model.eval(), tuple(slots), []
        self.device=weight.device
        self.hybrid=isinstance(model,Qwen3_5ForCausalLM)
        cache_type=HybridCache if self.hybrid else TensorKVCache
        self.cache = cache_type(model.config, capacity, self.device, weight.dtype)
        self.forward_calls = self.input_tokens = 0
        self.replay_calls=self.replay_tokens=self.snapshots=self.snapshot_bytes=0
        self.peak_snapshot_bytes=0
        self._snapshot=None

    def synchronize(self):
        if self.device.type=='npu':
            import torch
            torch.npu.synchronize(self.device)

    def checkpoint(self):
        if self._snapshot is not None:
            raise RuntimeError('resolve the previous checkpoint before making another')
        if self.hybrid:
            states={key:tuple(t.clone() for t in values) for key,values in self.cache.linear_states.items()}
            size=sum(t.numel()*t.element_size() for values in states.values() for t in values)
            self._snapshot=(len(self.tokens),states,self.last_logits)
            self.snapshots+=1
            self.snapshot_bytes+=size
            self.peak_snapshot_bytes=max(self.peak_snapshot_bytes,size)

    def forward(self, tokens):
        import torch
        from .model_runner import ModelInput
        from .engine import Step

        tokens = tuple(tokens)
        start, end = len(self.tokens), len(self.tokens)+len(tokens)
        if not tokens or end > len(self.slots) or end > self.model.config.max_position_embeddings:
            raise ValueError('nonempty input must fit reserved slots and model positions')
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in tokens):
            raise ValueError('input token is outside the vocabulary')
        tensors = (tokens, tuple(range(start,end)), self.slots[start:end], self.slots[:end])
        step=Step('spec','prefill' if start==0 else 'decode',*tensors)
        inputs = ModelInput(*(torch.tensor(v,dtype=torch.long,device=self.device) for v in tensors),steps=(step,))
        try:
            with torch.inference_mode():
                logits = self.model(inputs,self.cache)
        finally:
            self.synchronize()  # Synchronous reference, including replay and model errors.
        self.tokens.extend(tokens)
        self.forward_calls += 1
        self.input_tokens += len(tokens)
        self.last_logits=logits[-1]
        return logits

    def rollback(self, retained):
        if type(retained) is not int or not 0 <= retained <= len(self.tokens):
            raise ValueError('rollback must retain a prefix of computed inputs')
        invalidated = self.slots[retained:len(self.tokens)]
        if self.hybrid and retained<len(self.tokens):
            if self._snapshot is None or retained<self._snapshot[0]:
                raise ValueError('hybrid rollback requires a checkpoint before the retained inputs')
            before,states,last_logits=self._snapshot
            kept=tuple(self.tokens[before:retained])
            self.cache.linear_states=states
            del self.tokens[before:]
            self.last_logits=last_logits
            if kept:
                self.forward(kept)  # Rebuild convolution and recurrence, as well as retained KV writes.
                self.replay_calls+=1
                self.replay_tokens+=len(kept)
        else:
            del self.tokens[retained:]  # Full-attention tensor tails remain outside valid context.
        self._snapshot=None
        return invalidated


def generate(model, prompt_ids, max_new_tokens, draft, *, width=3, eos_token_id=None,
             cache_capacity=None, on_round=None, on_cache=None, on_commit=None, should_cancel=None,
             sampler=None):
    """draft(committed_prefix, limit) supplies up to limit candidates after pending.

    The last committed output is pending: it has not yet been consumed by target.
    Each verify forward consumes pending+draft, then keeps only the valid prefix.
    This synchronous function owns its cache and slots for the duration of one call.
    on_commit receives each CPU-visible group of committed output tokens.
    should_cancel is polled before prefill and after each commit callback. A
    committed group is atomic; EOS and length completion take precedence.
    With sampler, draft returns SampledDraft with the actual proposal distributions.
    """
    from .engine import validate_request
    from .kv_cache import SlotPool
    from .sampling import SampledDraft, TemperatureSampler, verify_sampled

    prompt = tuple(prompt_ids)
    if type(width) is not int or not 0 <= width <= 8 or not callable(draft):
        raise ValueError('provide a draft callback and an integer width between 0 and 8')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    if should_cancel is not None and not callable(should_cancel):
        raise ValueError('should_cancel must be a callable or None')
    if sampler is not None and (not isinstance(sampler, TemperatureSampler)
                                or next(model.parameters()).device.type not in ('cpu','npu')):
        raise ValueError('sampled generation requires a CPU or NPU model and TemperatureSampler')
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
    direct = 0
    def stop_reason(natural):
        return natural or ('cancelled' if should_cancel is not None and should_cancel() else None)
    try:
        reason = stop_reason(None)
        if reason is None:
            target = CachedTarget(model,slots,capacity)
            first = target.forward(prompt)[-1]
            output.append(int(first.argmax()) if sampler is None else sampler.sample(first))
            direct = 1  # Prefill supplies the first output, without draft verification.
            reason = 'eos' if output[-1]==eos_token_id else 'length' if max_new_tokens==1 else None
            if on_commit is not None:
                on_commit(tuple(output))
            reason = stop_reason(reason)
        while reason is None:
            remaining = max_new_tokens-len(output)
            count = min(width,remaining-1)
            proposal = draft(prompt+tuple(output),count)
            if sampler is not None:
                if not isinstance(proposal, SampledDraft):
                    raise ValueError('sampled generation requires a SampledDraft, not greedy candidates')
                candidates = proposal.tokens
            else:
                if isinstance(proposal, SampledDraft):
                    raise ValueError('sampled candidates require a target sampler')
                candidates = tuple(proposal)
            if len(candidates) > count or any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in candidates):
                raise ValueError('draft must return at most the requested number of vocabulary IDs')
            before = len(target.tokens)
            inputs = (output[-1],)+candidates
            # With no candidates, only pending is consumed and nothing can be
            # rejected. The whole new cache prefix stays valid, even at EOS.
            if candidates:
                target.checkpoint()
            logits = target.forward(inputs)
            predictions = tuple(logits.argmax(-1).tolist())
            tentative = len(target.tokens)
            if sampler is None:
                decision = verify_greedy(candidates,predictions,remaining,eos_token_id)
            else:
                decision, checks = verify_sampled(proposal, tuple(sampler.probabilities(row) for row in logits),
                                                   remaining, sampler.rng, eos_token_id)
            output.extend(decision.tokens)
            retained = len(prompt)+len(output)-1
            replay_before=target.replay_calls
            replay_tokens_before=target.replay_tokens
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
                          replay_calls=target.replay_calls-replay_before,
                          replay_tokens=target.replay_tokens-replay_tokens_before,
                          finish_reason=decision.finish_reason)
            rounds.append(record)
            if sampler is not None:
                record['sampling_checks'] = checks
            if on_commit is not None:
                on_commit(decision.tokens)
            reason = stop_reason(decision.finish_reason)
            record['finish_reason'] = reason
            if on_cache is not None:
                on_cache(target)
            if on_round is not None:
                on_round(record)
        result = dict(output_ids=output,finish_reason=reason,rounds=rounds,
                      cached_tokens=len(target.tokens) if target is not None else 0, proposed_draft_tokens=proposed,
                      accepted_draft_tokens=accepted, correction_tokens=corrections, bonus_tokens=bonuses,
                      direct_target_tokens=direct,
                      target_forward_calls=target.forward_calls if target is not None else 0,
                      target_input_tokens=target.input_tokens if target is not None else 0,
                      replay_forward_calls=target.replay_calls if target is not None else 0,
                      replayed_input_tokens=target.replay_tokens if target is not None else 0,
                      snapshot_count=target.snapshots if target is not None else 0,
                      snapshot_bytes_total=target.snapshot_bytes if target is not None else 0,
                      peak_snapshot_bytes=target.peak_snapshot_bytes if target is not None else 0)
    finally:
        if target is not None:
            target.synchronize()  # If draining fails, do not return the slots to the pool.
            if target.hybrid:
                target.cache.release('spec')
            target._snapshot=None
        pool.release('spec')
    result['free_slots'] = pool.free_count
    if sampler is not None:
        result['sampling'] = dict(temperature=sampler.temperature, seed=sampler.seed,
                                  probability_device='cpu', probability_dtype='float64')
        if sampler.top_k or sampler.top_p != 1:
            result['sampling'].update(top_k=sampler.top_k, top_p=sampler.top_p,
                                      filter_order='temperature,top_k,top_p')
    return result
