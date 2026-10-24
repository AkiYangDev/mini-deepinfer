"""Request lifecycle while a batch owns device work and reserved cache slots."""

import random
import unittest

from minideepinfer import BatchEngine, Status
from test_batch_engine import PrefixRunner


class ControlledResult:
    def __init__(self, owner, steps):
        self.owner, self.steps = owner, steps
        self.tokens = owner.math.run_batch(steps)
        self.cancelled = set()
        self.releasable = False

    def wait(self):
        if self.owner.fail_wait:
            raise RuntimeError('wait failed')

    def result(self):
        tokens = self.owner.override or self.tokens
        self.releasable = True
        self.owner.pending = None
        return tuple(None if step.request_id in self.cancelled else token
                     for step, token in zip(self.steps, tokens))

    def cancel(self, request_id):
        self.cancelled.add(request_id)

    def close(self):
        if self.owner.fail_drain:
            raise RuntimeError('drain failed')
        self.releasable = True
        self.owner.pending = None


class ControlledRunner:
    def __init__(self):
        self.math = PrefixRunner()
        self.pending = None
        self.fail_wait = self.fail_drain = self.fail_launch = False
        self.override = None

    def launch(self, steps):
        self.pending = ControlledResult(self, steps)
        if self.fail_launch:
            self.pending.close()
            raise RuntimeError('launch failed after writes')
        return self.pending

    def release(self, request_id):
        if self.pending is not None:
            raise AssertionError('release before result or safe drain')
        self.math.release(request_id)


class DeferredEngineTests(unittest.TestCase):
    def engine(self, capacity=8, count=2, budget=2):
        runner = ControlledRunner()
        return runner, BatchEngine(runner, capacity, count, budget,
                                   chunked_prefill=True, deferred=True)

    def test_progress_only_commits_on_finish_and_pending_blocks_another_launch(self):
        runner, engine = self.engine()
        request = engine.submit([1, 2, 3], 2)
        self.assertFalse(engine.abort_step())
        with self.assertRaises(RuntimeError):
            engine.finish_step()
        plans = engine.launch_step()
        self.assertEqual(plans[0].input_ids, (1, 2))
        self.assertEqual((request.num_cached_tokens, request.output_ids), (0, []))
        self.assertEqual(engine.cache.free_count, 4)
        for operation in (engine.launch_step, engine.step):
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                operation()
        first = engine.finish_step()[0]
        self.assertEqual((first.token_id, request.num_cached_tokens), (None, 2))
        list(engine.run())
        self.assertEqual(request.output_ids, [6, 12])
        self.assertEqual(request.finish_reason, 'length')
        self.assertEqual(engine.cache.free_count, 8)

    def test_cancel_in_flight_holds_slots_then_new_request_reuses_them(self):
        runner, engine = self.engine(8, 2, 4)
        a, b = [engine.submit([i], 4) for i in (1, 4)]
        plans = engine.launch_step()
        a_slots = engine._running[a.request_id]
        engine.cancel(a.request_id)
        c = engine.submit([7], 1)
        self.assertEqual(engine.cache.free_count, 0)
        self.assertEqual(c.status, Status.WAITING)
        self.assertEqual(runner.math.released, [])
        self.assertEqual([o.token_id for o in engine.finish_step()], [None, 4])
        self.assertEqual((a.num_cached_tokens, a.output_ids, a.status), (0, [], Status.CANCELLED))
        self.assertEqual(b.output_ids, [4])
        engine.launch_step()
        self.assertTrue(set(engine._running[c.request_id]).issubset(a_slots))
        engine.finish_step()
        list(engine.run())
        self.assertEqual(b.output_ids, [4, 8, 16, 15])
        self.assertEqual(c.output_ids, [7])
        self.assertEqual(engine.cache.free_count, 8)

    def test_cancel_waiting_and_partial_request_during_a_batch(self):
        runner, engine = self.engine(16, 2, 3)
        a = engine.submit([1], 4)
        b = engine.submit([2] * 6, 2)
        engine.step()  # A samples; B has two cached prompt tokens.
        plans = engine.launch_step()  # A decodes while B computes another chunk.
        self.assertEqual([p.request_id for p in plans], [a.request_id, b.request_id])
        waiting = engine.submit([8], 1)
        engine.cancel(b.request_id)
        engine.cancel(waiting.request_id)
        self.assertIn(b.request_id, engine._running)
        engine.finish_step()
        self.assertEqual(b.num_cached_tokens, 2)
        self.assertEqual(b.output_ids, [])
        self.assertNotIn(b.request_id, engine._running)
        list(engine.run())
        self.assertEqual(engine.cache.free_count, 16)

    def test_eos_releases_only_after_its_result_and_cancelled_eos_stays_cancelled(self):
        runner, engine = self.engine(8, 2, 4)
        a = engine.submit([3], 3, eos_token_id=3)
        b = engine.submit([4], 3, eos_token_id=4)
        engine.launch_step()
        self.assertEqual(a.status, Status.RUNNING)
        engine.cancel(b.request_id)
        outputs = engine.finish_step()
        self.assertEqual([o.finish_reason for o in outputs], ['eos', 'cancelled'])
        self.assertEqual([a.output_ids, b.output_ids], [[3], []])
        self.assertEqual([a.status, b.status], [Status.FINISHED, Status.CANCELLED])
        self.assertEqual(engine.cache.free_count, 8)

    def test_bad_results_fail_whole_batch_and_leave_waiting_work_intact(self):
        for tokens in ((1,), (1, -1), (1, True), (1, None)):
            with self.subTest(tokens=tokens):
                runner, engine = self.engine(6, 2, 4)
                a, b, c = [engine.submit([i], 2) for i in (1, 2, 3)]
                engine.launch_step()
                runner.override = tokens
                with self.assertRaises(ValueError):
                    engine.finish_step()
                self.assertEqual([a.output_ids, b.output_ids], [[], []])
                self.assertEqual([a.status, b.status, c.status],
                                 [Status.FAILED, Status.FAILED, Status.WAITING])
                self.assertEqual(engine.cache.free_count, 6)
                runner.override = None
                list(engine.run())
                self.assertEqual(c.output_ids, [3, 6])

    def test_wait_and_drain_failure_keep_slots_until_explicit_successful_abort(self):
        runner, engine = self.engine(6, 2, 4)
        a, b, waiting = [engine.submit([i], 2) for i in (1, 2, 3)]
        engine.launch_step()
        engine.cancel(a.request_id)
        runner.fail_wait = runner.fail_drain = True
        with self.assertRaisesRegex(RuntimeError, 'wait failed'):
            engine.finish_step()
        pending = runner.pending
        for operation in (engine.abort_step, engine.step):
            with self.assertRaises(RuntimeError):
                operation()
        self.assertIs(runner.pending, pending)
        self.assertEqual(engine.cache.free_count, 2)
        self.assertEqual(runner.math.released, [])
        self.assertEqual([a.status, b.status, waiting.status],
                         [Status.CANCELLED, Status.FAILED, Status.WAITING])
        runner.fail_wait = runner.fail_drain = False
        self.assertTrue(engine.abort_step())
        self.assertEqual(engine.cache.free_count, 6)
        self.assertFalse(engine.abort_step())
        list(engine.run())
        self.assertEqual(waiting.output_ids, [3, 6])

    def test_launch_failure_with_safe_or_failed_drain(self):
        for unsafe in (False, True):
            with self.subTest(unsafe=unsafe):
                runner, engine = self.engine()
                request = engine.submit([1, 2, 3], 2)
                runner.fail_launch = True
                runner.fail_drain = unsafe
                with self.assertRaises(RuntimeError):
                    engine.launch_step()
                self.assertEqual(request.status, Status.FAILED)
                self.assertEqual((request.num_cached_tokens, request.output_ids), (0, []))
                self.assertEqual(engine.cache.free_count, 4 if unsafe else 8)
                if unsafe:
                    with self.assertRaises(RuntimeError):
                        engine.launch_step()
                    runner.fail_drain = False
                    engine.abort_step()
                self.assertEqual(engine.cache.free_count, 8)

    def test_abort_discards_existing_prefixes_and_preserves_waiting_requests(self):
        runner, engine = self.engine(16, 2, 2)
        a = engine.submit([1], 4)
        b = engine.submit([2, 3, 4, 5], 2)
        engine.step()
        self.assertEqual([a.num_cached_tokens, b.num_cached_tokens], [1, 1])
        waiting = engine.submit([6], 1)
        engine.launch_step()
        engine.abort_step()
        self.assertEqual([a.status, b.status, waiting.status],
                         [Status.FAILED, Status.FAILED, Status.WAITING])
        self.assertEqual([a.output_ids, b.output_ids], [[1], []])
        list(engine.run())
        self.assertEqual(waiting.output_ids, [6])
        self.assertEqual(engine.cache.free_count, 16)

    def test_generated_arrivals_and_midflight_cancels_match_full_history(self):
        for seed in range(20):
            rng = random.Random(seed)
            runner, engine = self.engine(16, 3, 6)
            requests = []
            for _ in range(40):
                if rng.random() < .7:
                    requests.append(engine.submit([rng.randrange(17) for _ in range(rng.randint(1, 9))],
                                                  rng.randint(1, 5), rng.choice([None, 0, 8])))
                plans = engine.launch_step()
                before = {r.request_id: (r.num_cached_tokens, tuple(r.output_ids)) for r in requests}
                if requests and rng.random() < .5:
                    engine.cancel(rng.choice(requests).request_id)
                self.assertTrue(all(before[r.request_id] == (r.num_cached_tokens, tuple(r.output_ids))
                                    for r in requests))
                reserved = sum(len(engine._running[rid]) for rid in engine._running)
                self.assertEqual(engine.cache.free_count + reserved, 16)
                if plans:
                    engine.finish_step()
                for request in requests:
                    for i, token in enumerate(request.output_ids):
                        self.assertEqual(token, sum(request.prompt_ids + tuple(request.output_ids[:i])) % 17,
                                         (seed, request.request_id, i))
                    if request.status == Status.CANCELLED:
                        self.assertEqual(before[request.request_id],
                                         (request.num_cached_tokens, tuple(request.output_ids)))
            list(engine.run())
            self.assertEqual(engine.cache.free_count, 16)
            self.assertTrue(all(r.status in (Status.FINISHED, Status.CANCELLED) for r in requests))

    def test_default_synchronous_engine_rejects_split_launch(self):
        engine = BatchEngine(PrefixRunner())
        with self.assertRaises(RuntimeError):
            engine.launch_step()


if __name__ == '__main__':
    unittest.main()
