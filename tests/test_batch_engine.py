"""Batch membership, atomic commit, and generated lifecycle/oracle checks."""

import random
import unittest

from minideepinfer import BatchEngine, Status


class PrefixRunner:
    def __init__(self):
        self.values = {}  # Dirty storage remains when ownership is released.
        self.batches = []
        self.released = []

    def run_batch(self, steps):
        self.batches.append(steps)
        tokens = []
        for step in steps:
            start = step.positions[0]
            total = self.values[step.context_slots[start - 1]] if start else 0
            for token, slot in zip(step.input_ids, step.slot_mapping):
                total += token
                self.values[slot] = total
            tokens.append(total % 17 if step.is_last_prompt_chunk else None)
        return tuple(tokens)

    def release(self, request_id):
        self.released.append(request_id)


class BatchEngineTests(unittest.TestCase):
    def test_invalid_configuration_and_request_leave_no_pending_work(self):
        for count, budget in ((0, 4), (True, 4), (2, 1), (2, True)):
            with self.subTest(count=count, budget=budget), self.assertRaises(ValueError):
                BatchEngine(PrefixRunner(), 8, count, budget)
        engine = BatchEngine(PrefixRunner(), 8, 2, 4)
        for prompt, limit in (([], 1), ([True], 1), ([-1], 1), ([1], 0), ([1] * 5, 1), ([1] * 4, 6)):
            with self.subTest(prompt=prompt, limit=limit), self.assertRaises(ValueError):
                engine.submit(prompt, limit)
        self.assertEqual(engine.step(), ())
        self.assertEqual(engine.cache.free_count, 8)

    def test_changing_membership_mixed_prefill_decode_and_reuse(self):
        runner = PrefixRunner()
        engine = BatchEngine(runner, 10, max_num_seqs=2)
        a = engine.submit([1, 2], 1)
        b = engine.submit([3], 3)
        first = engine.step()
        c = engine.submit([7, 8], 2)
        second = engine.step()
        self.assertEqual([o.request_id for o in first], [a.request_id, b.request_id])
        self.assertEqual([(o.request_id, o.phase) for o in second],
                         [(b.request_id, "decode"), (c.request_id, "prefill")])
        self.assertEqual(runner.batches[0][0].slot_mapping, runner.batches[1][1].slot_mapping)
        list(engine.run())
        self.assertEqual([a.output_ids, b.output_ids, c.output_ids], [[3], [3, 6, 12], [15, 13]])
        self.assertEqual(engine.cache.free_count, 10)
        self.assertEqual(set(runner.released), {a.request_id, b.request_id, c.request_id})

    def test_cache_pressure_and_fifo_head_keep_existing_decodes_running(self):
        runner = PrefixRunner()
        engine = BatchEngine(runner, 5, max_num_seqs=3)
        a = engine.submit([1], 3)
        b = engine.submit([2, 3], 2)
        c = engine.submit([4], 1)
        engine.step()
        self.assertEqual([a.status, b.status, c.status], [Status.RUNNING, Status.WAITING, Status.WAITING])
        list(engine.run())
        self.assertEqual([len(batch) for batch in runner.batches], [1, 1, 1, 2, 1])
        self.assertEqual(engine.cache.free_count, 5)

    def test_token_budget_limits_admission_without_starving_decodes(self):
        runner = PrefixRunner()
        engine = BatchEngine(runner, 20, max_num_seqs=3, max_num_batched_tokens=4)
        engine.submit([1, 2, 3], 3)
        engine.submit([4, 5], 2)
        engine.submit([6], 1)
        list(engine.run())
        self.assertEqual([sum(len(s.input_ids) for s in batch) for batch in runner.batches], [3, 4, 2])
        with self.assertRaisesRegex(ValueError, "whole prompt"):
            engine.submit([1] * 5, 1)
        self.assertEqual(engine.cache.free_count, 20)

    def test_partial_prefill_commits_cache_progress_without_emitting_tokens(self):
        runner = PrefixRunner()
        engine = BatchEngine(runner, 16, max_num_seqs=2, max_num_batched_tokens=2,
                             chunked_prefill=True)
        request = engine.submit([1, 2, 3, 4, 5], 2)
        for cached in (2, 4):
            output = engine.step()[0]
            self.assertIsNone(output.token_id)
            self.assertEqual(request.output_ids, [])
            self.assertEqual(request.num_cached_tokens, cached)
            self.assertEqual(request.status, Status.RUNNING)
        self.assertEqual(engine.step()[0].token_id, 15)
        self.assertEqual(request.num_cached_tokens, 5)
        list(engine.run())
        self.assertEqual(request.output_ids, [15, 13])
        self.assertEqual(engine.cache.free_count, 16)

    def test_decode_budget_is_reserved_while_long_prompt_is_split(self):
        runner = PrefixRunner()
        engine = BatchEngine(runner, 20, 2, 3, chunked_prefill=True)
        short = engine.submit([2], 4)
        long = engine.submit([1, 2, 3, 4, 5, 6], 2)
        engine.step()
        self.assertEqual(long.num_cached_tokens, 2)
        second = engine.step()
        self.assertEqual([(o.request_id, o.phase, o.token_id) for o in second],
                         [(short.request_id, "decode", 4), (long.request_id, "prefill", None)])
        list(engine.run())
        self.assertEqual(short.output_ids, [2, 4, 8, 16])
        self.assertEqual(long.output_ids, [4, 8])
        self.assertTrue(all(sum(len(p.input_ids) for p in b) <= 3 for b in runner.batches))
        self.assertEqual(engine.cache.free_count, 20)

    def test_cancel_partial_prompt_and_reuse_dirty_slots(self):
        runner = PrefixRunner()
        engine = BatchEngine(runner, 8, 2, 2, chunked_prefill=True)
        cancelled = engine.submit([1, 2, 3, 4], 2)
        engine.step()
        engine.cancel(cancelled.request_id)
        replacement = engine.submit([8, 9, 10], 1)
        list(engine.run())
        self.assertEqual(cancelled.output_ids, [])
        self.assertEqual(cancelled.num_cached_tokens, 2)
        self.assertEqual(replacement.output_ids, [10])
        self.assertEqual(engine.cache.free_count, 8)

    def test_partial_prefill_cannot_be_mistaken_for_a_sample(self):
        class WrongPartial(PrefixRunner):
            def run_batch(self, steps):
                return tuple(3 for _ in steps)

        engine = BatchEngine(WrongPartial(), 8, 2, 2, chunked_prefill=True)
        request = engine.submit([1, 2, 3], 1)
        with self.assertRaises(ValueError):
            engine.step()
        self.assertEqual(request.num_cached_tokens, 0)
        self.assertEqual(request.output_ids, [])
        self.assertEqual(request.status, Status.FAILED)
        self.assertEqual(engine.cache.free_count, 8)

    def test_cancellation_and_eos_release_only_their_request(self):
        runner = PrefixRunner()
        engine = BatchEngine(runner, 10, max_num_seqs=2)
        a = engine.submit([1], 4)
        b = engine.submit([4], 4, eos_token_id=4)
        waiting = engine.submit([8], 2)
        engine.step()
        self.assertEqual(b.finish_reason, "eos")
        self.assertTrue(engine.cancel(a.request_id))
        self.assertTrue(engine.cancel(waiting.request_id))
        self.assertFalse(engine.cancel(a.request_id))
        self.assertEqual(engine.step(), ())
        self.assertEqual(engine.cache.free_count, 10)

    def test_batch_failure_or_malformed_results_do_not_partially_commit(self):
        class Broken(PrefixRunner):
            def run_batch(self, steps):
                super().run_batch(steps)
                if self.mode == "raise":
                    raise RuntimeError("after cache write")
                return self.mode

        for result in ("raise", (1,), (1, -1), (1, True), (1, None)):
            with self.subTest(result=result):
                runner = Broken()
                runner.mode = result
                engine = BatchEngine(runner, 6, max_num_seqs=2)
                a, b, waiting = [engine.submit([i], 2) for i in (1, 2, 3)]
                with self.assertRaises((RuntimeError, ValueError)):
                    engine.step()
                self.assertEqual([a.output_ids, b.output_ids], [[], []])
                self.assertEqual([a.status, b.status], [Status.FAILED, Status.FAILED])
                self.assertEqual(waiting.status, Status.WAITING)
                self.assertEqual(engine.cache.free_count, 6)
                runner.mode = (7,)
                # Only the waiting request is admitted after the failure.
                list(engine.run())
                self.assertEqual(waiting.output_ids, [7, 7])

    def test_generated_streams_match_full_history_and_conserve_slots(self):
        # Standard-library generation: no new testing/runtime dependency.
        for seed in range(40):
            rng = random.Random(seed)
            runner = PrefixRunner()
            chunked = bool(seed % 2)
            engine = BatchEngine(runner, 16, max_num_seqs=3, max_num_batched_tokens=6,
                                 chunked_prefill=chunked)
            requests = []
            for _ in range(60):
                if rng.random() < 0.55:
                    prompt = [rng.randrange(17) for _ in range(rng.randint(1, 9 if chunked else 4))]
                    requests.append(engine.submit(prompt, rng.randint(1, 5), rng.choice([None, 0, 8])))
                if requests and rng.random() < 0.25:
                    engine.cancel(rng.choice(requests).request_id)
                engine.step()
                for request in requests:
                    for i, token in enumerate(request.output_ids):
                        self.assertEqual(token, sum(request.prompt_ids + tuple(request.output_ids[:i])) % 17,
                                         (seed, request.request_id, i))
                live = [r for r in requests if r.status == Status.RUNNING]
                reserved = sum(len(r.prompt_ids) + r.max_new_tokens - 1 for r in live)
                self.assertEqual(engine.cache.free_count + reserved, engine.cache.capacity, seed)
                for batch in runner.batches[-1:]:
                    contexts = [s for plan in batch for s in plan.context_slots]
                    self.assertEqual(len(contexts), len(set(contexts)))
                    self.assertLessEqual(sum(len(plan.input_ids) for plan in batch), 6)
            list(engine.run())
            for request in requests:
                for i, token in enumerate(request.output_ids):
                    self.assertEqual(token, sum(request.prompt_ids + tuple(request.output_ids[:i])) % 17,
                                     (seed, request.request_id, i))
            self.assertEqual(engine.cache.free_count, 16)
            self.assertTrue(all(r.status in (Status.FINISHED, Status.CANCELLED) for r in requests))


if __name__ == "__main__":
    unittest.main()
