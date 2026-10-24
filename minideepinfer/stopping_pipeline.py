"""Callback arrivals and stops in a small FIFO device-token pipeline.

Reuse the fixed pipeline's stream, device-token handoff, and two-round lifetime.
One later round may already have written cache when the CPU learns of a stop.
Discard its output and release the whole request after its last recorded use.
"""

from collections import deque
from contextlib import nullcontext

import torch

from .decode_pipeline import FixedDecodePipeline
from .engine import Request, Status, validate_request
from .planning import IssuedProgress, plan_step


class StoppingDecodePipeline(FixedDecodePipeline):
    def __init__(self, runner):
        super().__init__(runner)
        self.requests = {}
        self._retired = {}
        self._in_result = False
        self._waiting = deque()
        self._issued = {}
        self._last_use = {}

    @torch.inference_mode()
    def run(self, prompts, max_new_tokens, *, eos_token_ids=None, orders=None, on_result=None, max_num_seqs=None,
            max_num_batched_tokens=None, chunked_prefill=False, early_first_result=False):
        """Queue initial prompts; the result callback may submit or cancel requests.

        The callback has the fixed experiment's signature, with None for rows
        already stopped. It runs before commit and may call cancel(request_id).
        Calls are single-threaded. Orders permute initial IDs for the first rounds;
        new IDs follow them in FIFO order. Later rounds use insertion order.
        Returns request records, including committed tokens and stopping reasons.
        early_first_result collects round zero before launching round one. Later
        rounds keep the usual order; this also changes the first callback's timing.
        """
        if self._active or self.runner._prefixes or self.runner._request_id is not None:
            raise RuntimeError("pipeline requires an idle, exclusively owned runner")
        prompts = {rid: tuple(tokens) for rid, tokens in prompts.items()}
        if not prompts:
            raise ValueError("provide at least one initial request")
        if type(early_first_result) is not bool:
            raise ValueError("early_first_result must be boolean")
        limit = len(prompts) if max_num_seqs is None else max_num_seqs
        if type(limit) is not int or limit < 1:
            raise ValueError("max_num_seqs must be a positive integer")
        budget = self.pool.capacity if max_num_batched_tokens is None else max_num_batched_tokens
        if type(budget) is not int or budget < limit or type(chunked_prefill) is not bool:
            raise ValueError("token budget must be an integer >= max_num_seqs; chunked_prefill must be boolean")
        self._budget, self._chunked = budget, chunked_prefill
        eos = dict(eos_token_ids or {})
        if (set(eos) - set(prompts) or any(t is not None and
                (type(t) is not int or not 0 <= t < self.runner.model.config.vocab_size) for t in eos.values())):
            raise ValueError("EOS IDs must belong to known requests and be inside the vocabulary")
        initial = {rid: self._request(rid, p, max_new_tokens, eos.get(rid)) for rid, p in prompts.items()}
        if orders is not None:
            orders = tuple(tuple(order) for order in orders)
            if len(orders) != max_new_tokens or any(
                    len(order) != len(initial) or set(order) != set(initial) for order in orders):
                raise ValueError("each initial round order must contain every initial request once")
        else:
            orders = ()
        self.requests = initial
        self._retired = {}
        self._waiting = deque(initial)
        self._issued = {}
        self._last_use = {}
        self._default_max = max_new_tokens
        self._active = True
        previous, index = None, 0
        previous_collected = False
        try:
            if self.is_npu:
                self.stream.wait_stream(torch.npu.current_stream(self.runner.device))
            with torch.npu.stream(self.stream) if self.is_npu else nullcontext():
                while True:
                    steps = self._schedule(index, orders, limit)
                    if not steps:
                        if previous is None:
                            if any(r.status in (Status.WAITING, Status.RUNNING) for r in self.requests.values()):
                                raise RuntimeError("waiting work cannot be admitted after draining")
                            break
                        if not previous_collected:
                            self._finish(previous, None, on_result)
                        self._states.pop(0)
                        previous = None
                        continue  # Freed slots may admit a waiting prompt.
                    current = self._launch(index, steps, previous)
                    for step in steps:
                        rid = step.request_id
                        self._issued[rid] = self._issued[rid].advance(step)
                        self._last_use[rid] = current.ready
                    if previous is not None:
                        if not previous_collected:
                            self._finish(previous, None, on_result)
                        self._states.pop(0)
                    self._sample(current)
                    previous_collected = early_first_result and index == 0
                    if previous_collected:
                        self._finish(current, None, on_result)
                    previous, index = current, index + 1
            return dict(self.requests)
        except Exception as error:
            for request in self.requests.values():
                if request.status in (Status.WAITING, Status.RUNNING):
                    request.status = Status.FAILED
                    request.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.close()

    def _request(self, rid, tokens, count, eos):
        if not isinstance(rid, str) or not rid:
            raise ValueError("request ID must be a nonempty string")
        prompt = validate_request(tokens, count, eos, self.pool.capacity)
        config = self.runner.model.config
        if (any(t >= config.vocab_size for t in prompt) or eos is not None and eos >= config.vocab_size
                or len(prompt) + count - 1 > config.max_position_embeddings):
            raise ValueError("request exceeds the model vocabulary or position limit")
        if not self._chunked and len(prompt) > self._budget:
            raise ValueError("whole prompt exceeds token budget; enable chunked_prefill")
        return Request(rid, prompt, count, eos)

    def submit(self, request_id, prompt_ids, max_new_tokens=None, eos_token_id=None):
        """Queue from on_result; no slots or model state are assigned here."""
        if not self._active or not self._in_result:
            raise RuntimeError("submit only from this pipeline's result callback")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request ID must be a nonempty string")
        if request_id in self.requests:
            raise ValueError("request IDs cannot be reused during a run")
        request = self._request(request_id, prompt_ids,
                                self._default_max if max_new_tokens is None else max_new_tokens, eos_token_id)
        self.requests[request_id] = request
        self._waiting.append(request_id)
        return request

    def _schedule(self, index, orders, limit):
        order = orders[index] if index < len(orders) else ()
        order = order + tuple(rid for rid in self.requests if rid not in order)
        active = [rid for rid in order if self.requests[rid].status == Status.RUNNING
                  and self._issued[rid].sampled_tokens < self.requests[rid].max_new_tokens]
        plans, budget = [], self._budget
        # Like BatchEngine: decodes first, then existing prompt work, then FIFO admission.
        for prefill in (False, True):
            for rid in active:
                request, progress = self.requests[rid], self._issued[rid]
                if (progress.cached_tokens < len(request.prompt_ids)) == prefill and budget:
                    step = plan_step(request, self._slots[rid], progress.cached_tokens, budget)
                    plans.append(step)
                    budget -= len(step.input_ids)
        while self._waiting and budget:
            rid = self._waiting[0]
            request = self.requests[rid]
            if request.status != Status.WAITING:
                self._waiting.popleft()
                continue
            need = len(request.prompt_ids) + request.max_new_tokens - 1
            if (len(self._slots) >= limit or need > self.pool.free_count
                    or not self._chunked and len(request.prompt_ids) > budget):
                break
            self._slots[rid] = self.pool.reserve(rid, need)
            self._issued[rid] = IssuedProgress()
            request.status = Status.RUNNING
            self._waiting.popleft()
            step = plan_step(request, self._slots[rid], 0, budget)
            plans.append(step)
            budget -= len(step.input_ids)
        rank = {rid: i for i, rid in enumerate(order)}
        return tuple(sorted(plans, key=lambda step: (step.phase == "prefill", rank[step.request_id])))

    def cancel(self, request_id):
        """Cancel before this result commits; previously committed tokens remain."""
        if not self._active or not self._in_result:
            raise RuntimeError("cancel only from this pipeline's result callback")
        request = self.requests[request_id]
        if request.status not in (Status.WAITING, Status.RUNNING):
            return False
        was_running = request.status == Status.RUNNING
        request.status, request.finish_reason = Status.CANCELLED, "cancelled"
        if was_running:
            self._retire(request_id)
        return True

    def _retire(self, request_id):
        self._retired[request_id] = self._last_use[request_id]

    def _finish(self, state, outputs, on_result):
        state.ready.synchronize()
        tokens = tuple(state.host_tokens.tolist())
        if any((not 0 <= t < self.runner.model.config.vocab_size if step.is_last_prompt_chunk else t != -1)
               for step, t in zip(state.steps, tokens)):
            raise ValueError("invalid sampled token")
        visible = tuple(t if step.is_last_prompt_chunk and self.requests[step.request_id].status == Status.RUNNING else None
                        for step, t in zip(state.steps, tokens))
        if on_result is not None:
            self._in_result = True
            try:
                on_result(state.index, state.steps, visible, state.logits)
            finally:
                self._in_result = False
        for step, token in zip(state.steps, tokens):
            request = self.requests[step.request_id]
            if request.status != Status.RUNNING:
                continue  # Includes a cancellation made by the callback above.
            request.num_cached_tokens = len(step.context_slots)
            if not step.is_last_prompt_chunk:
                continue
            request.output_ids.append(token)
            if token == request.eos_token_id:
                request.finish_reason = "eos"
            elif len(request.output_ids) == request.max_new_tokens:
                request.finish_reason = "length"
            if request.finish_reason is not None:
                request.status = Status.FINISHED
                self._retire(request.request_id)
        self._collect()

    def _collect(self):
        for rid, last_use in tuple(self._retired.items()):
            if last_use.query():  # An unrecorded event is deliberately not ready.
                self.runner.release(rid)
                self.pool.release(rid)
                self._slots.pop(rid)
                self._last_use.pop(rid)
                del self._retired[rid]

    def close(self):
        if self._in_result:
            raise RuntimeError("use cancel in the result callback; close after run exits")
        super().close()  # If draining raises, retain retirement events as well.
        self._retired.clear()
        self._waiting.clear()
        self._issued.clear()
        self._last_use.clear()
