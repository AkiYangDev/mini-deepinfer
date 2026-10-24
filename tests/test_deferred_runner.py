"""The model runs now; host prefixes and outputs commit only when read."""

import importlib.util
import sys
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer import BatchEngine, Status, Step
    from minideepinfer.deferred_runner import DeferredRunner
    from minideepinfer.kv_cache import SlotPool
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from minideepinfer.step_state import DeferredRelease


@unittest.skipUnless(AVAILABLE, 'install torch for deferred model execution tests')
class DeferredRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def runner(self):
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        return ModelRunner(Qwen3_5ForCausalLM(config).eval(), 32)

    def first(self, rid='A'):
        return Step(rid, 'prefill', (2, 3), (0, 1), (0, 1), (0, 1))

    def test_partial_prefill_and_prefixes_commit_only_on_result(self):
        runner = self.runner()
        deferred = DeferredRunner(runner)
        step = Step('A', 'prefill', (2, 3), (0, 1), (0, 1), (0, 1), False)
        pending = deferred.launch((step,))
        self.assertEqual(runner._prefixes, {})
        self.assertTrue(runner.cache.linear_states)
        with self.assertRaises(RuntimeError):
            deferred.release('A')
        with self.assertRaises(RuntimeError):
            deferred.launch((step,))
        with patch.object(pending.ready, 'query', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                pending.result()
            self.assertEqual(runner._prefixes, {})
        self.assertEqual(pending.wait().result(), (None,))
        self.assertFalse(runner._prefixes['A'].prompt_complete)
        self.assertEqual(runner._prefixes['A'].slots, (0, 1))
        deferred.release('A')
        self.assertEqual(runner.cache.linear_states, {})

    def test_mixed_chunks_decode_and_reordering_match_synchronous_runner(self):
        runner = self.runner()
        baseline = ModelRunner(runner.model, 32)
        deferred = DeferredRunner(runner)
        batches = [
            (Step('A', 'prefill', (2, 3), (0, 1), (0, 1), (0, 1), False),
             Step('B', 'prefill', (5,), (0,), (8,), (8,))),
            (Step('B', 'decode', (7,), (1,), (9,), (8, 9)),
             Step('A', 'prefill', (4,), (2,), (2,), (0, 1, 2))),
            (Step('A', 'decode', (9,), (3,), (3,), (0, 1, 2, 3)),
             Step('B', 'decode', (8,), (2,), (10,), (8, 9, 10))),
        ]
        for steps in batches:
            expected = baseline.forward_batch(steps)
            old = dict(runner._prefixes)
            pending = deferred.launch(steps)
            self.assertEqual(runner._prefixes, old)
            pending.wait()
            for actual, reference in zip(pending.logits, expected):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            expected_tokens = tuple(baseline.execute_sample(row) if step.is_last_prompt_chunk else None
                                    for step, row in zip(steps, expected))
            self.assertEqual(pending.result(), expected_tokens)
        for rid in ('A', 'B'):
            deferred.release(rid)
        self.assertEqual(runner._prefixes, {})
        self.assertEqual(runner.cache.linear_states, {})

    def test_cancel_drops_one_output_and_reclaims_only_after_commit(self):
        runner = self.runner()
        deferred = DeferredRunner(runner)
        pool = SlotPool(12)
        a = pool.reserve('A', 6)
        b = pool.reserve('B', 6)
        cleanup = DeferredRelease(pool, deferred.release)
        pending = deferred.launch((self.first(), Step('B', 'prefill', (4,), (0,), b[:1], b[:1])))
        cleanup.retain(pending)
        pending.cancel('A')
        cleanup.retire('A')
        self.assertEqual(cleanup.collect(), ())
        self.assertEqual(pool.free_count, 0)
        outputs = pending.wait().result()
        self.assertIsNone(outputs[0])
        self.assertIsInstance(outputs[1], int)
        self.assertNotIn('A', runner._prefixes)
        self.assertIn('B', runner._prefixes)
        self.assertTrue(any(rid == 'A' for rid, layer in runner.cache.linear_states))
        with self.assertRaises(ValueError):
            deferred.launch((self.first(),))
        self.assertEqual(cleanup.collect(), ('A',))
        self.assertEqual(pool.reserve('C', 6), a)
        self.assertFalse(any(rid == 'A' for rid, layer in runner.cache.linear_states))
        cleanup.retire('B')
        cleanup.collect()
        pool.release('C')
        self.assertEqual(pool.free_count, 12)

    def test_close_abandons_the_whole_batch_and_is_idempotent(self):
        deferred = DeferredRunner(self.runner())
        pending = deferred.launch((self.first(),))
        pending.close()
        pending.close()
        self.assertTrue(pending.closed and pending.releasable)
        self.assertEqual(deferred.runner.cache.linear_states, {})
        self.assertEqual(deferred.runner._prefixes, {})
        with self.assertRaises(RuntimeError):
            pending.wait()
        with self.assertRaises(RuntimeError):
            pending.result()
        with self.assertRaises(RuntimeError):
            pending.cancel('A')
        deferred.launch((self.first(),)).wait().result()
        deferred.release('A')

    def test_error_after_cache_writes_drains_then_clears_only_this_batch(self):
        runner = self.runner()
        deferred = DeferredRunner(runner)
        x = Step('X', 'prefill', (5,), (0,), (20,), (20,))
        deferred.launch((x,)).wait().result()
        original = runner.execute_main_model
        def broken(inputs):
            original(inputs)
            raise RuntimeError('host error after forward submission')
        with patch.object(runner, 'execute_main_model', broken):
            with self.assertRaisesRegex(RuntimeError, 'host error'):
                deferred.launch((self.first(),))
        self.assertIsNone(deferred._pending)
        self.assertEqual(set(runner._prefixes), {'X'})
        self.assertEqual({rid for rid, layer in runner.cache.linear_states}, {'X'})
        deferred.launch((self.first(),)).wait().result()
        deferred.release('A')
        deferred.release('X')

    def test_invalid_plan_does_not_destroy_an_existing_prefix(self):
        runner = self.runner()
        deferred = DeferredRunner(runner)
        deferred.launch((self.first(),)).wait().result()
        old = dict(runner._prefixes)
        keys = set(runner.cache.linear_states)
        bad = Step('A', 'decode', (4,), (4,), (4,), (0, 1, 4))
        with self.assertRaises(ValueError):
            deferred.launch((bad,))
        self.assertEqual(runner._prefixes, old)
        self.assertEqual(set(runner.cache.linear_states), keys)
        self.assertIsNone(deferred._pending)
        deferred.release('A')

    def test_failed_drain_keeps_buffers_and_blocks_reuse(self):
        runner = self.runner()
        deferred = DeferredRunner(runner)
        original = runner.execute_main_model
        def broken(inputs):
            original(inputs)
            raise RuntimeError('forward failed')
        with patch.object(runner, 'execute_main_model', broken), patch.object(
                deferred, '_synchronize', side_effect=RuntimeError('drain failed')):
            with self.assertRaisesRegex(RuntimeError, 'drain failed'):
                deferred.launch((self.first(),))
        pending = deferred._pending
        self.assertIsNotNone(pending)
        self.assertFalse(pending.releasable)
        self.assertTrue(pending._buffers)
        self.assertTrue(runner.cache.linear_states)
        for operation in (lambda: deferred.release('A'), lambda: pending.result(),
                          lambda: deferred.launch((self.first(),))):
            with self.assertRaises(RuntimeError):
                operation()
        pending.close()  # CPU-only recovery simulation; the patched drain is gone.
        self.assertEqual(runner.cache.linear_states, {})

    def test_wait_error_keeps_ownership_and_invalid_results_never_partially_commit(self):
        deferred = DeferredRunner(self.runner())
        pending = deferred.launch((self.first(),))
        with patch.object(pending.ready, 'synchronize', side_effect=RuntimeError('wait failed')):
            with self.assertRaisesRegex(RuntimeError, 'wait failed'):
                pending.wait()
        self.assertFalse(pending.releasable)
        with self.assertRaises(RuntimeError):
            deferred.release('A')
        pending.close()
        for values in ((-2,), (1, 2)):
            pending = deferred.launch((self.first(),))
            pending.host_tokens = torch.tensor(values, dtype=torch.long)
            with self.assertRaises(ValueError):
                pending.result()
            self.assertEqual(deferred.runner._prefixes, {})
            self.assertEqual(deferred.runner.cache.linear_states, {})
        with self.assertRaises(ValueError):
            deferred.launch(())

    def test_engine_lifecycle_matches_sync_with_real_hybrid_cache(self):
        from minideepinfer.verify_deferred_engine import run
        report = run(None, 'cpu')
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['steps']), 6)
        self.assertEqual(report['free_slots'], 32)
        self.assertEqual(max(c['max_abs'] for row in report['steps']
                             for c in row['comparisons']), 0)

    def test_engine_retains_failed_launch_and_wait_until_safe_abort(self):
        for stage in ('launch', 'wait'):
            with self.subTest(stage=stage):
                runner = self.runner()
                deferred = DeferredRunner(runner)
                engine = BatchEngine(deferred, 8, 2, 2, chunked_prefill=True, deferred=True)
                request = engine.submit([2, 3, 4], 2)
                if stage == 'launch':
                    original = runner.execute_main_model
                    def fail(inputs):
                        original(inputs)
                        raise RuntimeError('host forward failure')
                    with patch.object(runner, 'execute_main_model', fail), patch.object(
                            deferred, '_synchronize', side_effect=RuntimeError('drain failed')):
                        with self.assertRaisesRegex(RuntimeError, 'drain failed'):
                            engine.launch_step()
                else:
                    engine.launch_step()
                    with patch.object(deferred.pending.ready, 'synchronize',
                                      side_effect=RuntimeError('wait failed')):
                        with self.assertRaisesRegex(RuntimeError, 'wait failed'):
                            engine.finish_step()
                self.assertEqual(request.status, Status.FAILED)
                self.assertEqual(engine.cache.free_count, 4)
                self.assertTrue(runner.cache.linear_states)
                self.assertIsNotNone(deferred.pending)
                with self.assertRaises(RuntimeError):
                    engine.step()
                engine.abort_step()  # Controlled CPU failures removed; no device fault recovery claim.
                self.assertEqual(engine.cache.free_count, 8)
                self.assertEqual(runner.cache.linear_states, {})
                self.assertEqual(runner._prefixes, {})
                self.assertIsNone(deferred.pending)

    def test_engine_invalid_device_results_do_not_commit_any_request(self):
        deferred = DeferredRunner(self.runner())
        engine = BatchEngine(deferred, 8, 2, 4, deferred=True)
        a, b = [engine.submit([i], 2) for i in (2, 3)]
        engine.launch_step()
        deferred.pending.host_tokens = torch.tensor([1, -2], dtype=torch.long)
        with self.assertRaises(ValueError):
            engine.finish_step()
        self.assertEqual([a.output_ids, b.output_ids], [[], []])
        self.assertEqual([a.num_cached_tokens, b.num_cached_tokens], [0, 0])
        self.assertEqual([a.status, b.status], [Status.FAILED, Status.FAILED])
        self.assertEqual(engine.cache.free_count, 8)
        self.assertEqual(deferred.runner._prefixes, {})
        self.assertEqual(deferred.runner.cache.linear_states, {})

    def test_text_preparation_orders_match_with_real_model_and_hybrid_cache(self):
        from minideepinfer.benchmark_prepare import run_trial, TinyTokenizer
        model = self.runner().model
        after = run_trial(model, TinyTokenizer(), False)
        before = run_trial(model, TinyTokenizer(), True)
        self.assertEqual(after['plans'], before['plans'])
        for a, b in zip(after['requests'], before['requests']):
            self.assertEqual(a['output_ids'], b['output_ids'])
            self.assertEqual(a['finish_reason'], b['finish_reason'])
        self.assertTrue(after['clean'] and before['clean'])
        self.assertEqual(after['output_tokens'], 16)

    def test_sampler_index_upload_precedes_forward_and_partial_batches_skip_it(self):
        for sampling in (False, True):
            with self.subTest(sampling=sampling):
                runner = self.runner()
                deferred = DeferredRunner(runner)
                original_tensor, original_forward = torch.tensor, runner.execute_main_model
                calls = []
                def tensor(values, *args, **kwargs):
                    if sys._getframe(1).f_code.co_name == 'launch':
                        calls.append(('indices', list(values)))
                    return original_tensor(values, *args, **kwargs)
                def forward(inputs):
                    calls.append(('forward', None))
                    return original_forward(inputs)
                a = Step('A', 'prefill', (2, 3), (0, 1), (0, 1), (0, 1), False)
                b = Step('B', 'prefill', (4,), (0,), (8,), (8,), sampling)
                with patch.object(torch, 'tensor', tensor), patch.object(runner, 'execute_main_model', forward):
                    pending = deferred.launch((a, b))
                self.assertEqual(calls, [('indices', [1]), ('forward', None)] if sampling
                                 else [('forward', None)])
                values = pending.wait().result()
                self.assertIsNone(values[0])
                self.assertEqual(values[1] is not None, sampling)
                deferred.release('A')
                deferred.release('B')

    def test_index_upload_failure_keeps_existing_prefix_before_any_model_write(self):
        runner = self.runner()
        deferred = DeferredRunner(runner)
        deferred.launch((self.first(),)).wait().result()
        prefix = dict(runner._prefixes)
        states = dict(runner.cache.linear_states)
        original_tensor = torch.tensor
        def tensor(values, *args, **kwargs):
            if sys._getframe(1).f_code.co_name == 'launch':
                raise RuntimeError('sample index upload failed')
            return original_tensor(values, *args, **kwargs)
        step = Step('A', 'decode', (4,), (2,), (2,), (0, 1, 2))
        with patch.object(torch, 'tensor', tensor):
            with self.assertRaisesRegex(RuntimeError, 'sample index upload failed'):
                deferred.launch((step,))
        self.assertEqual(runner._prefixes, prefix)
        self.assertEqual(set(runner.cache.linear_states), set(states))
        for key, value in states.items():
            self.assertIs(runner.cache.linear_states[key], value)
        self.assertIsNone(deferred.pending)
        deferred.launch((step,)).wait().result()
        deferred.release('A')


if __name__ == '__main__':
    unittest.main()
