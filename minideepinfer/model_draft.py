"""A draft model with its own cache, aligned to committed target tokens."""

import hashlib
import json
from pathlib import Path


def check_tokenizer_pair(draft_directory, target_directory):
    """A strict local-file contract; equal vocabulary sizes alone are insufficient."""
    roots = (Path(draft_directory), Path(target_directory))
    configs = [json.loads((root/'config.json').read_text(encoding='utf-8')) for root in roots]
    text = [config.get('text_config',config) for config in configs]
    fields = ('vocab_size','bos_token_id','eos_token_id','pad_token_id')
    ids = [{key:config.get(key) for key in fields} for config in text]
    if ids[0] != ids[1] or type(ids[0]['vocab_size']) is not int or ids[0]['vocab_size'] < 1:
        raise ValueError('draft and target vocabulary size or special token IDs differ')
    shared = {}
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','vocab.json'):
        paths = [root/name for root in roots]
        required = name in ('tokenizer.json','tokenizer_config.json')
        if not any(path.exists() for path in paths) and not required:
            continue
        if not all(path.is_file() for path in paths):
            raise ValueError('both checkpoints must contain matching '+name)
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        if digests[0] != digests[1]:
            raise ValueError('tokenizer files differ: '+name)
        shared[name] = digests[0]
    return dict(compatible=True,token_ids=ids[0],shared_file_sha256=shared,
                config_sha256=[hashlib.sha256((root/'config.json').read_bytes()).hexdigest() for root in roots])


class ModelDraft:
    """One request, separate slots. Resolve the previous branch on the next call."""

    def __init__(self, model, capacity, *, on_cache=None, sampler=None):
        from .kv_cache import SlotPool
        from .sampling import TemperatureSampler
        if sampler is not None and (not isinstance(sampler, TemperatureSampler)
                                    or next(model.parameters()).device.type != 'cpu'):
            raise ValueError('sampled draft requires a CPU model and TemperatureSampler')
        if type(capacity) is not int or capacity < 1 or capacity > model.config.max_position_embeddings:
            raise ValueError('draft capacity exceeds its position limit')
        self.model = model
        self.pool = SlotPool(capacity)
        self.slots = self.pool.reserve('draft',capacity)
        self.cache = None
        self.committed = ()
        self.closed = False
        self.failed = False
        self.on_cache = on_cache
        self.sampler = sampler
        self.calls = self.proposed = self.recoveries = 0
        self.sync_calls = self.sync_tokens = 0
        self.proposal_calls = self.proposal_tokens = 0

    def __call__(self, prefix, limit):
        prefix = tuple(prefix)
        if self.closed:
            raise ValueError('draft is closed')
        if self.failed:
            raise RuntimeError('draft failed; close it and create a new draft')
        if type(limit) is not int or not 0 <= limit <= 8:
            raise ValueError('draft limit must be an integer from zero to eight')
        if (not prefix or any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in prefix)
                or len(prefix)+max(0,limit-1) > len(self.slots)):
            raise ValueError('committed prefix and candidates must fit draft vocabulary and slots')
        if self.committed and (len(prefix) <= len(self.committed) or prefix[:len(self.committed)] != self.committed):
            raise ValueError('committed prefix must extend the previous committed prefix')
        self.calls += 1
        if limit == 0:
            if self.sampler is not None:
                from .sampling import SampledDraft
                return SampledDraft((), ())
            return ()  # No draft computation is needed for a direct final target output.
        try:
            return self._propose(prefix,limit)
        except BaseException:
            # A failed forward may already have changed KV or linear state.
            self.failed = True
            raise

    def _propose(self, prefix, limit):
        from .speculative import CachedTarget
        if self.cache is None:
            self.cache = CachedTarget(self.model,self.slots,self.pool.capacity)
        else:
            common = 0
            for cached,actual in zip(self.cache.tokens,prefix):
                if cached != actual:
                    break
                common += 1
            # A trimmed pure-attention cache has no saved logits at its new endpoint.
            if common == len(prefix) and common < len(self.cache.tokens):
                common -= 1
            self.recoveries += common < len(self.cache.tokens)
            self.cache.rollback(common)
        added = prefix[len(self.cache.tokens):]
        if added:
            self.cache.forward(added)
            self.sync_calls += 1
            self.sync_tokens += len(added)
        self.committed = prefix
        if self.on_cache is not None:
            self.on_cache(self.cache)  # State now represents exactly the committed prefix.
        # One candidate only reads the committed endpoint; no tentative state is written.
        if limit > 1:
            self.cache.checkpoint()
        candidates, probabilities = [], []
        for index in range(limit):
            if self.sampler is None:
                candidates.append(int(self.cache.last_logits.argmax()))
            else:
                from .sampling import draw
                q = self.sampler.probabilities(self.cache.last_logits)
                probabilities.append(q)
                candidates.append(draw(q, self.sampler.rng))
            if index+1 < limit:
                self.cache.forward((candidates[-1],))
                self.proposal_calls += 1
                self.proposal_tokens += 1
        self.proposed += len(candidates)
        if self.sampler is not None:
            from .sampling import SampledDraft
            return SampledDraft(tuple(candidates), tuple(probabilities))
        return tuple(candidates)

    def close(self):
        if self.closed:
            return
        if self.cache is not None:
            try:
                self.cache.synchronize()  # Failed drain keeps this request's slots owned.
            except BaseException:
                self.failed = True
                raise
            if self.cache.hybrid:
                self.cache.cache.release('spec')
            self.cache._snapshot = None
        self.pool.release('draft')
        self.closed = True

    def stats(self):
        cache = self.cache
        return dict(calls=self.calls,proposed_tokens=self.proposed,recovered_branches=self.recoveries,
                    forward_calls=cache.forward_calls if cache else 0,input_tokens=cache.input_tokens if cache else 0,
                    sync_forward_calls=self.sync_calls,sync_input_tokens=self.sync_tokens,
                    proposal_forward_calls=self.proposal_calls,proposal_input_tokens=self.proposal_tokens,
                    replay_forward_calls=cache.replay_calls if cache else 0,replayed_input_tokens=cache.replay_tokens if cache else 0,
                    snapshot_count=cache.snapshots if cache else 0,snapshot_bytes_total=cache.snapshot_bytes if cache else 0,
                    peak_snapshot_bytes=cache.peak_snapshot_bytes if cache else 0,
                    linear_states_remaining=len(cache.cache.linear_states) if cache and cache.hybrid else 0,
                    snapshot_retained=cache._snapshot is not None if cache else False,
                    failed=self.failed,closed=self.closed,free_slots=self.pool.free_count)
