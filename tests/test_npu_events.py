"""Check asynchronous boundaries without NPU hardware or tensor dependencies."""

import unittest

from minideepinfer.kv_cache import SlotPool
from minideepinfer.npu_events import PendingTokens, RecordedEvent
from minideepinfer.step_state import DeferredRelease


class FakeEvent:
    def __init__(self):
        self.done = True  # Actual NPU query() also returns True before record.
        self.fail_record = False

    def record(self, stream):
        if self.fail_record:
            raise RuntimeError("record failed")
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


class HostBuffer:
    def __init__(self, values):
        self.values = values
        self.reads = 0

    def __len__(self):
        return len(self.values)

    def tolist(self):
        self.reads += 1
        return list(self.values)


class NpuEventTests(unittest.TestCase):
    def event(self):
        raw = FakeEvent()
        return raw, RecordedEvent(raw)

    def test_unrecorded_true_is_not_completion(self):
        raw, event = self.event()
        self.assertTrue(raw.query())
        self.assertFalse(event.query())
        with self.assertRaisesRegex(RuntimeError, "unrecorded"):
            event.synchronize()
        with self.assertRaisesRegex(RuntimeError, "unrecorded"):
            event.wait(None)
        event.record(None)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        with self.assertRaisesRegex(RuntimeError, "new event"):
            event.record(None)

    def test_record_failure_stays_unrecorded_and_wait_orders_the_stream(self):
        raw, event = self.event()
        raw.fail_record = True
        with self.assertRaisesRegex(RuntimeError, "record failed"):
            event.record(None)
        self.assertFalse(event.query())
        raw.fail_record = False
        event.record(None)
        class Stream:
            def wait_event(self, value):
                self.waited = value
        stream = Stream()
        event.wait(stream)
        self.assertIs(stream.waited, raw)
        self.assertFalse(event.query())  # A stream wait does not finish device work.

    def test_host_buffer_is_unreadable_until_d2h_and_closed_afterwards(self):
        raw, copied = self.event()
        _, used = self.event()
        buffer = HostBuffer([11, 21])
        state = PendingTokens(("A", "B"), buffer, copied, used, ())
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            state.inputs_for(("B", "C"), {"C": 30})
        self.assertEqual(buffer.reads, 0)
        copied.record(None)
        with self.assertRaises(RuntimeError):
            state.inputs_for(("B",), {})
        self.assertEqual(buffer.reads, 0)
        raw.done = True
        self.assertEqual(state.inputs_for(("B", "C"), {"C": 30}), (21, 30))
        state.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            state.inputs_for(("B",), {})
        self.assertEqual(buffer.reads, 1)

    def test_cancelled_result_waits_for_recorded_last_use(self):
        import weakref
        pool, calls = SlotPool(2), []
        slots = pool.reserve("A", 2)
        cleanup = DeferredRelease(pool, calls.append)
        _, copied = self.event()
        _, used = self.event()
        buffer = HostBuffer([11])
        kept = HostBuffer([99])
        reference = weakref.ref(kept)
        state = PendingTokens(("A",), buffer, copied, used, (kept,))
        cleanup.retain(state)
        cleanup.retire("A")
        state.close()
        del kept
        self.assertIsNotNone(reference())
        self.assertEqual(cleanup.collect(), ())
        copied.record(None)
        copied.synchronize()
        # Native query is True for the last-use event, but it has not been recorded.
        self.assertEqual(cleanup.collect(), ())
        with self.assertRaises(MemoryError):
            pool.reserve("C", 2)
        used.record(None)
        self.assertEqual(cleanup.collect(), ())
        used.synchronize()
        self.assertEqual(cleanup.collect(), ("A",))
        self.assertEqual(pool.reserve("C", 2), slots)
        del state
        self.assertIsNone(reference())
        self.assertEqual(calls, ["A"])

    def test_sync_error_propagates_without_reclaiming_and_bad_output_is_rejected(self):
        raw, event = self.event()
        buffer = HostBuffer([-1])
        state = PendingTokens(("A",), buffer, event, event, ())
        pool = SlotPool(1)
        pool.reserve("A", 1)
        cleanup = DeferredRelease(pool, lambda rid: None)
        cleanup.retain(state)
        cleanup.retire("A")
        event.record(None)
        def broken_sync():
            raise RuntimeError("device synchronization failed")
        raw.synchronize = broken_sync
        with self.assertRaisesRegex(RuntimeError, "device synchronization failed"):
            event.synchronize()
        self.assertEqual(cleanup.collect(), ())
        self.assertEqual(pool.free_count, 0)
        raw.done = True
        with self.assertRaises(ValueError):
            state.inputs_for(("A",), {})
        with self.assertRaises(ValueError):
            PendingTokens(("A", "A"), buffer, event, event, ())
        with self.assertRaises(ValueError):
            PendingTokens(("A", "B"), buffer, event, event, ())


if __name__ == "__main__":
    unittest.main()
