"""Event and host-buffer ownership for the small NPU experiment.

No torch import: dependency-free tests can check the lifetime rules too.
"""

from .step_state import StepState, batch_remap


class RecordedEvent:
    """An unrecorded device event must not stand for completed work."""

    def __init__(self, event):
        self._event = event
        self._recorded = False

    def record(self, stream):
        if self._recorded:
            raise RuntimeError("use a new event for each step")
        self._event.record(stream)
        self._recorded = True

    def query(self):
        return self._recorded and self._event.query()

    def wait(self, stream):
        if not self._recorded:
            raise RuntimeError("cannot wait for an unrecorded event")
        stream.wait_event(self._event)

    def synchronize(self):
        if not self._recorded:
            raise RuntimeError("cannot synchronize an unrecorded event")
        self._event.synchronize()


class PendingTokens:
    """Keep buffers alive until CPU readers and every device use are finished.

    ready must cover the D2H copy, not just sampling. last_use must cover all
    later consumers. Callers retain this state before launch and keep each
    referenced buffer immutable until its device readers have finished.
    """

    def __init__(self, request_ids, host_tokens, ready, last_use, keepalive):
        self.request_ids = tuple(request_ids)
        batch_remap((), self.request_ids)
        if not self.request_ids or len(host_tokens) != len(self.request_ids):
            raise ValueError("host buffer needs one token per request")
        self._host_tokens = host_tokens
        self.ready, self.last_use = ready, last_use
        self._keepalive = tuple(keepalive)
        self._closed = False

    def inputs_for(self, current, fresh):
        if self.closed:
            raise RuntimeError("this state's host reader is closed")
        if not self.ready.query():
            raise RuntimeError("host token copy is not ready")
        # Materialize only after the copy. Reuse the CPU lesson's ID mapping
        # and token validation instead of maintaining a second mapping rule.
        result = StepState(self.request_ids, tuple(self._host_tokens.tolist()),
                           self.ready, self.last_use)
        return result.inputs_for(current, fresh)

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed

    @property
    def releasable(self):
        return self.closed and self.ready.query() and self.last_use.query()
