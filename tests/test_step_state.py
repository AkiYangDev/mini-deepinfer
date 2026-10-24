"""Catch premature reuse, request-row mixups, and abandoned result leaks."""

import unittest

from minideepinfer.kv_cache import SlotPool
from minideepinfer.step_demo import ManualEvent
from minideepinfer.step_state import DeferredRelease, StepState, batch_remap


class StepStateTests(unittest.TestCase):
    def make_state(self, ids=("A", "B"), tokens=(11, 21)):
        return StepState(ids, tokens, ManualEvent(), ManualEvent())

    def test_reorder_and_fresh_input_use_request_identity(self):
        state = self.make_state()
        state.ready.complete()
        self.assertEqual(batch_remap(state.request_ids, ("B", "C", "A")), (1, -1, 0))
        self.assertEqual(state.inputs_for(("B", "C", "A"), {"C": 30}), (21, 30, 11))
        self.assertEqual(state.inputs_for((), {}), ())
        with self.assertRaises(KeyError):
            state.inputs_for(("C",), {})
        with self.assertRaises(ValueError):
            state.inputs_for(("C",), {"C": True})

    def test_early_and_closed_reads_are_blocked(self):
        state = self.make_state()
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            state.inputs_for(("B",), {})
        state.ready.complete()
        self.assertEqual(state.inputs_for(("B",), {}), (21,))
        state.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            state.inputs_for(("B",), {})

    def test_every_device_and_host_completion_is_required_in_any_order(self):
        # Covers cancellation before sampling as well as normal result reading.
        import itertools
        for order in itertools.permutations(("sample", "device", "host")):
            with self.subTest(order=order):
                pool = SlotPool(2)
                slots = pool.reserve("A", 2)
                callbacks = []
                cleanup = DeferredRelease(pool, callbacks.append)
                state = self.make_state(("A",), (11,))
                cleanup.retain(state)
                cleanup.retire("A")
                actions = {"sample": state.ready.complete, "device": state.last_use.complete,
                           "host": state.close}
                for action in order[:2]:
                    actions[action]()
                    self.assertEqual(cleanup.collect(), ())
                    with self.assertRaises(MemoryError):
                        pool.reserve("C", 2)
                actions[order[2]]()
                self.assertEqual(cleanup.collect(), ("A",))
                self.assertEqual(pool.reserve("C", 2), slots)
                cleanup.retire("A")
                self.assertEqual(cleanup.collect(), ())
                self.assertEqual(callbacks, ["A"])

    def test_both_old_and_new_states_hold_a_continuing_request(self):
        pool = SlotPool(4)
        a, b = pool.reserve("A", 2), pool.reserve("B", 2)
        callbacks = []
        cleanup = DeferredRelease(pool, callbacks.append)
        previous = self.make_state()
        following = self.make_state(("B",), (22,))
        cleanup.retain(previous)
        cleanup.retain(following)
        cleanup.retire("A")
        cleanup.retire("B")
        previous.ready.complete()
        previous.last_use.complete()
        previous.close()
        self.assertEqual(cleanup.collect(), ("A",))
        self.assertEqual(pool.reserve("C", 2), a)
        self.assertEqual(pool.free_count, 0)
        following.ready.complete()
        following.last_use.complete()
        self.assertEqual(cleanup.collect(), ())
        following.close()
        self.assertEqual(cleanup.collect(), ("B",))
        self.assertEqual(pool.reserve("D", 2), b)
        self.assertEqual(callbacks, ["A", "B"])

    def test_exception_abandons_output_but_waits_for_device_cleanup(self):
        pool = SlotPool(2)
        pool.reserve("A", 2)
        callbacks = []
        cleanup = DeferredRelease(pool, callbacks.append)
        state = self.make_state(("A",), (11,))
        cleanup.retain(state)
        try:
            raise RuntimeError("simulated worker error after launch")
        except RuntimeError:
            cleanup.retire("A")
            state.close()
        self.assertEqual(cleanup.collect(), ())
        state.ready.complete()
        state.last_use.complete()
        self.assertEqual(cleanup.collect(), ("A",))
        self.assertEqual(pool.free_count, 2)
        self.assertEqual(callbacks, ["A"])

    def test_invalid_batches_and_retired_work_are_rejected(self):
        for ids in (("A", "A"), ("",), (1,)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                batch_remap((), ids)
        for ids, tokens in (((), ()), (("A",), ()), (("A",), (True,))):
            with self.subTest(ids=ids, tokens=tokens), self.assertRaises(ValueError):
                self.make_state(ids, tokens)
        cleanup = DeferredRelease(SlotPool(2), lambda rid: None)
        state = self.make_state()
        cleanup.retain(state)
        with self.assertRaises(ValueError):
            cleanup.retain(state)
        cleanup.retire("A")
        with self.assertRaises(ValueError):
            cleanup.retain(self.make_state())
        closed = self.make_state(("C",), (1,))
        closed.close()
        with self.assertRaises(ValueError):
            cleanup.retain(closed)


if __name__ == "__main__":
    unittest.main()
