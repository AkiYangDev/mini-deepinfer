import unittest

from minideepinfer import Engine, Status
from minideepinfer.kv_cache import SlotPool


class RecordingRunner:
    def __init__(self):
        self.steps = []

    def run(self, step):
        self.steps.append(step)
        return step.input_ids[-1] + 1


class EngineTests(unittest.TestCase):
    def test_prefill_then_decode_and_fifo(self):
        runner = RecordingRunner()
        engine = Engine(runner, cache_capacity=4)
        first = engine.submit([10, 11], max_new_tokens=3)
        second = engine.submit([20], max_new_tokens=1)
        outputs = list(engine.run())
        self.assertEqual(first.output_ids, [12, 13, 14])
        self.assertEqual(second.output_ids, [21])
        self.assertEqual([s.input_ids for s in runner.steps], [(10, 11), (12,), (13,), (20,)])
        self.assertEqual([s.positions for s in runner.steps], [(0, 1), (2,), (3,), (0,)])
        self.assertEqual([s.slot_mapping for s in runner.steps], [(0, 1), (2,), (3,), (0,)])
        self.assertEqual([len(s.context_slots) for s in runner.steps], [2, 3, 4, 1])
        self.assertEqual([o.phase for o in outputs], ["prefill", "decode", "decode", "prefill"])
        self.assertEqual(first.num_cached_tokens, 4)  # Final output 14 has no KV.
        self.assertEqual(first.finish_reason, "length")
        self.assertEqual(first.status, Status.FINISHED)
        self.assertEqual(engine.cache.free_count, 4)
        self.assertIsNone(engine.step())

    def test_eos_stops_at_prefill_or_decode(self):
        for eos, expected in ((11, [11]), (12, [11, 12])):
            with self.subTest(eos=eos):
                engine = Engine(RecordingRunner(), cache_capacity=4)
                request = engine.submit([10], max_new_tokens=4, eos_token_id=eos)
                list(engine.run())
                self.assertEqual(request.output_ids, expected)
                self.assertEqual(request.finish_reason, "eos")
                self.assertEqual(engine.cache.free_count, 4)

    def test_cancel_waiting_and_running_then_reuse(self):
        engine = Engine(RecordingRunner(), cache_capacity=4)
        active = engine.submit([10], max_new_tokens=4)
        waiting = engine.submit([20], max_new_tokens=1)
        engine.step()
        self.assertEqual(engine.cache.free_count, 0)
        self.assertTrue(engine.cancel(waiting.request_id))
        self.assertTrue(engine.cancel(active.request_id))
        self.assertFalse(engine.cancel(active.request_id))
        self.assertEqual(engine.cache.free_count, 4)
        self.assertIsNone(engine.step())
        later = engine.submit([30], max_new_tokens=4)
        list(engine.run())
        self.assertEqual(later.output_ids, [31, 32, 33, 34])
        self.assertEqual(active.status, Status.CANCELLED)
        self.assertEqual(waiting.output_ids, [])

    def test_runner_error_releases_slots_and_preserves_waiting_request(self):
        class FailsOnDecode(RecordingRunner):
            def run(self, step):
                if step.phase == "decode":
                    raise RuntimeError("injected failure")
                return super().run(step)

        engine = Engine(FailsOnDecode(), cache_capacity=3)
        failed = engine.submit([10], max_new_tokens=3)
        queued = engine.submit([20], max_new_tokens=1)
        engine.step()
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            engine.step()
        self.assertEqual(failed.status, Status.FAILED)
        self.assertEqual(failed.error, "RuntimeError: injected failure")
        self.assertEqual(failed.output_ids, [11])
        self.assertEqual(failed.num_cached_tokens, 1)
        self.assertEqual(engine.cache.free_count, 3)
        list(engine.run())
        self.assertEqual(queued.output_ids, [21])

    def test_invalid_runner_output_fails_without_committing(self):
        class BadRunner:
            def run(self, step):
                return -1

        engine = Engine(BadRunner())
        request = engine.submit([1])
        with self.assertRaises(ValueError):
            engine.step()
        self.assertEqual(request.status, Status.FAILED)
        self.assertEqual(request.output_ids, [])
        self.assertEqual(engine.cache.free_count, engine.cache.capacity)

    def test_invalid_requests_do_not_block_the_queue(self):
        engine = Engine(RecordingRunner(), cache_capacity=2)
        for prompt, limit, eos in (([], 1, None), ([1], 0, None), ([-1], 1, None),
                                   ([True], 1, None), ([1], True, None), ([1], 1, -1),
                                   ([1, 2], 2, None)):
            with self.subTest(prompt=prompt, limit=limit, eos=eos):
                with self.assertRaises(ValueError):
                    engine.submit(prompt, limit, eos)
        good = engine.submit([1, 2], max_new_tokens=1)
        list(engine.run())
        self.assertEqual(good.output_ids, [3])

    def test_cached_history_matches_full_recomputation(self):
        class PrefixSumRunner:
            # Tiny stateful oracle: each cached value is a prefix sum, not K/V.
            def __init__(self):
                self.cache = {}

            def run(self, step):
                start = step.positions[0]
                total = self.cache[step.context_slots[start - 1]] if start else 0
                for token, slot in zip(step.input_ids, step.slot_mapping):
                    total += token
                    self.cache[slot] = total
                return total % 17

        for prompt_length in (1, 2, 4):
            for limit in (1, 2, 4):
                with self.subTest(prompt_length=prompt_length, limit=limit):
                    engine = Engine(PrefixSumRunner(), cache_capacity=prompt_length + limit - 1)
                    # The second request reuses dirty slots from the first.
                    for base in (1, 7):
                        prompt = list(range(base, base + prompt_length))
                        expected = []
                        for _ in range(limit):
                            expected.append(sum(prompt + expected) % 17)
                        request = engine.submit(prompt, limit)
                        list(engine.run())
                        self.assertEqual(request.output_ids, expected)
                        self.assertEqual(engine.cache.free_count, engine.cache.capacity)


class SlotPoolTests(unittest.TestCase):
    def test_ownership_capacity_and_reuse(self):
        pool = SlotPool(5)
        first = pool.reserve("first", 2)
        second = pool.reserve("second", 3)
        self.assertEqual(set(first) | set(second), set(range(5)))
        self.assertTrue(set(first).isdisjoint(second))
        with self.assertRaises(MemoryError):
            pool.reserve("third", 1)
        with self.assertRaises(ValueError):
            pool.reserve("first", 1)
        pool.release("first")
        pool.release("first")
        self.assertEqual(pool.free_count, 2)
        self.assertEqual(pool.reserve("third", 2), first)
        pool.release("second")
        pool.release("third")
        self.assertEqual(pool.free_count, 5)


if __name__ == "__main__":
    unittest.main()
