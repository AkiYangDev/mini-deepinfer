"""EOS and callback cancellation for a batch whose membership can only shrink.

Reuse the fixed pipeline's stream, device-token handoff, and two-round lifetime.
One later round may already have written cache when the CPU learns of a stop.
Discard its output and release the whole request after its last recorded use.
"""

from .decode_pipeline import FixedDecodePipeline
from .engine import Request, Status


class StoppingDecodePipeline(FixedDecodePipeline):
    def __init__(self, runner):
        super().__init__(runner)
        self.requests = {}
        self._retired = {}
        self._in_result = False

    def run(self, prompts, max_new_tokens, *, eos_token_ids=None, orders=None, on_result=None):
        """Admit once, then allow EOS, length stops, and cancellation in on_result.

        The callback has the fixed experiment's signature, with None for rows
        already stopped. It runs before commit and may call cancel(request_id).
        Calls are single-threaded; arrivals and concurrent cancellation are absent.
        Returns request records, including committed tokens and stopping reasons.
        """
        if self._active or self.runner._prefixes or self.runner._request_id is not None:
            raise RuntimeError("pipeline requires an idle, exclusively owned runner")
        prompts = {rid: tuple(tokens) for rid, tokens in prompts.items()}
        eos = dict(eos_token_ids or {})
        if (set(eos) - set(prompts) or any(t is not None and
                (type(t) is not int or not 0 <= t < self.runner.model.config.vocab_size) for t in eos.values())):
            raise ValueError("EOS IDs must belong to known requests and be inside the vocabulary")
        self.requests = {rid: Request(rid, p, max_new_tokens, eos.get(rid), status=Status.RUNNING)
                         for rid, p in prompts.items()}
        self._retired = {}
        try:
            super().run(prompts, max_new_tokens, orders=orders, on_result=on_result)
            return dict(self.requests)
        except Exception as error:
            for request in self.requests.values():
                if request.status == Status.RUNNING:
                    request.status = Status.FAILED
                    request.error = f"{type(error).__name__}: {error}"
            raise

    def _select(self, order):
        return tuple(rid for rid in order if self.requests[rid].status == Status.RUNNING)

    def cancel(self, request_id):
        """Cancel before this result commits; previously committed tokens remain."""
        if not self._active or not self._in_result:
            raise RuntimeError("cancel only from this pipeline's result callback")
        request = self.requests[request_id]
        if request.status != Status.RUNNING:
            return False
        request.status, request.finish_reason = Status.CANCELLED, "cancelled"
        self._retire(request_id)
        return True

    def _retire(self, request_id):
        # The later forward, if present, is the final cache user for this ID.
        last = next(s for s in reversed(self._states) if request_id in s.request_ids)
        self._retired[request_id] = last.ready

    def _finish(self, state, outputs, on_result):
        state.ready.synchronize()
        tokens = tuple(state.host_tokens.tolist())
        if any(not 0 <= t < self.runner.model.config.vocab_size for t in tokens):
            raise ValueError("invalid sampled token")
        visible = tuple(t if self.requests[rid].status == Status.RUNNING else None
                        for rid, t in zip(state.request_ids, tokens))
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
            request.output_ids.append(token)
            request.num_cached_tokens = len(step.context_slots)
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
                del self._retired[rid]

    def close(self):
        if self._in_result:
            raise RuntimeError("use cancel in the result callback; close after run exits")
        super().close()  # If draining raises, retain retirement events as well.
        self._retired.clear()
