"""New prompts share a round with device-token decode only after admission."""

import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer import BatchEngine, Status, Step
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from minideepinfer.stopping_pipeline import StoppingDecodePipeline


@unittest.skipUnless(AVAILABLE, 'install torch for arrival checks')
class ArrivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pipeline(self, capacity=32):
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        return StoppingDecodePipeline(ModelRunner(Qwen3_5ForCausalLM(config).eval(), capacity))

    def reference(self, pipeline, prompts, count=4):
        engine = BatchEngine(ModelRunner(pipeline.runner.model, 64), 64, len(prompts), 32)
        requests = {rid: engine.submit(ids, count) for rid, ids in prompts.items()}
        list(engine.run())
        return {rid: r.output_ids for rid, r in requests.items()}

    def assert_clean(self, pipeline):
        self.assertFalse(pipeline._active or pipeline._states or pipeline._slots or pipeline._retired)
        self.assertFalse(pipeline._waiting or pipeline._issued or pipeline._in_result)
        self.assertFalse(pipeline.runner._prefixes or pipeline.runner.cache.linear_states)
        self.assertEqual(pipeline.pool.free_count, pipeline.pool.capacity)

    def test_wait_then_mixed_prefill_reuses_eos_slots_without_old_linear_state(self):
        pipeline = self.pipeline(12)
        expected = self.reference(pipeline, {'A': (2, 3, 4), 'B': (7,), 'C': (2, 3, 4)})
        plans, old_slots, reused, guarded = [], [], [], []
        forward, launch = pipeline.runner.execute_main_model, pipeline._launch
        def capture(inputs):
            plans.append([(s.request_id, s.phase) for s in inputs.steps])
            if inputs.steps[0].request_id == 'A':
                old_slots[:] = pipeline._slots['A']
            if any(s.request_id == 'C' and s.phase == 'prefill' for s in inputs.steps):
                self.assertNotIn('A', pipeline.runner._prefixes)
                self.assertFalse(any(rid in ('A', 'C') for rid, layer in pipeline.runner.cache.linear_states))
                reused.append(set(pipeline._slots['C']).issubset(old_slots))
            return forward(inputs)
        def no_host_read(index, steps, previous):
            if any(s.phase == 'prefill' for s in steps) and any(s.phase == 'decode' for s in steps):
                with patch.object(previous.host_tokens, 'tolist', side_effect=AssertionError('read prior host token')):
                    state = launch(index, steps, previous)
                guarded.append(True)
                return state
            return launch(index, steps, previous)
        def arrive(index, steps, tokens, logits):
            if index == 0:
                request = pipeline.submit('C', (2, 3, 4), 2)
                self.assertEqual(request.status, Status.WAITING)
                self.assertNotIn('C', pipeline._slots)
            if index == 1:
                self.assertEqual(pipeline.requests['C'].status, Status.WAITING)
                self.assertIn('A', pipeline._slots)  # Collection follows this callback.
        with patch.object(pipeline.runner, 'execute_main_model', capture), patch.object(pipeline, '_launch', no_host_read):
            result = pipeline.run({'A': (2, 3, 4), 'B': (7,)}, 4,
                                  eos_token_ids={'A': expected['A'][0]}, on_result=arrive)
        self.assertEqual(plans, [[('A', 'prefill'), ('B', 'prefill')], [('A', 'decode'), ('B', 'decode')],
                                 [('B', 'decode')], [('B', 'decode'), ('C', 'prefill')], [('C', 'decode')]])
        self.assertEqual(result['A'].output_ids, expected['A'][:1])
        self.assertEqual(result['B'].output_ids, expected['B'])
        self.assertEqual(result['C'].output_ids, expected['C'][:2])
        self.assertEqual(reused, [True])
        self.assertEqual(guarded, [True])
        self.assert_clean(pipeline)

    def test_submit_in_tail_callback_extends_beyond_initial_round_count(self):
        pipeline = self.pipeline(8)
        expected = self.reference(pipeline, {'A': (2, 3), 'C': (7,)})
        counts = []
        def arrive(index, steps, tokens, logits):
            counts.append(index)
            if index == 0:
                pipeline.submit('C', (7,), 3)
        result = pipeline.run({'A': (2, 3)}, 1, max_num_seqs=1, on_result=arrive)
        self.assertEqual(counts, [0, 1, 2, 3])
        self.assertEqual(result['A'].output_ids, expected['A'][:1])
        self.assertEqual(result['C'].output_ids, expected['C'][:3])
        self.assert_clean(pipeline)

    def test_waiting_cancellation_never_allocates_and_id_cannot_be_reused(self):
        pipeline = self.pipeline()
        seen = []
        forward = pipeline.runner.execute_main_model
        def capture(inputs):
            seen.extend(s.request_id for s in inputs.steps)
            return forward(inputs)
        def arrive(index, steps, tokens, logits):
            if index == 0:
                pipeline.submit('C', (2,), 2)
                self.assertTrue(pipeline.cancel('C'))
                self.assertNotIn('C', pipeline._slots)
                with self.assertRaisesRegex(ValueError, 'reused'):
                    pipeline.submit('C', (2,), 1)
                pipeline.submit('D', (5,), 1)
        with patch.object(pipeline.runner, 'execute_main_model', capture):
            result = pipeline.run({'A': (2,), 'B': (7,)}, 2, on_result=arrive)
        self.assertNotIn('C', seen)
        self.assertIn('D', seen)
        self.assertEqual(result['C'].status, Status.CANCELLED)
        self.assertEqual(result['C'].output_ids, [])
        self.assertEqual(result['D'].finish_reason, 'length')
        self.assert_clean(pipeline)

    def test_fifo_waiting_head_prevents_smaller_request_from_overtaking(self):
        pipeline = self.pipeline(10)
        plans = []
        forward = pipeline.runner.execute_main_model
        def capture(inputs):
            plans.append(tuple(s.request_id for s in inputs.steps))
            return forward(inputs)
        def arrive(index, steps, tokens, logits):
            if index == 0:
                pipeline.submit('C', (11, 12, 13, 14), 3)
                pipeline.submit('D', (5,), 1)
            if index == 1:
                self.assertEqual(pipeline.requests['D'].status, Status.WAITING)
                self.assertNotIn('D', pipeline._slots)
        with patch.object(pipeline.runner, 'execute_main_model', capture):
            result = pipeline.run({'A': (2, 3, 4), 'B': (7,)}, 3, on_result=arrive)
        first_c = next(i for i, ids in enumerate(plans) if 'C' in ids)
        first_d = next(i for i, ids in enumerate(plans) if 'D' in ids)
        self.assertLessEqual(first_c, first_d)
        self.assertTrue(all(r.finish_reason == 'length' for r in result.values()))
        self.assert_clean(pipeline)

    def test_initial_requests_can_queue_instead_of_requiring_total_capacity(self):
        pipeline = self.pipeline(4)
        result = pipeline.run({'A': (2, 3), 'B': (7, 8), 'C': (9,)}, 3, max_num_seqs=1)
        self.assertTrue(all(len(r.output_ids) == 3 for r in result.values()))
        self.assert_clean(pipeline)

    def test_mixed_pack_handles_prefill_first_and_rejects_invalid_overrides(self):
        runner = self.pipeline().runner
        runner.run_batch((Step('B', 'prefill', (2,), (0,), (0,), (0,)),))
        steps = (Step('C', 'prefill', (5, 6), (0, 1), (8, 9), (8, 9)),
                 Step('B', 'decode', (0,), (1,), (1,), (0, 1)))
        ids = torch.tensor([5, 6, 7])
        inputs = runner.prepare_mixed_batch(steps, ids)
        self.assertIs(inputs.input_ids, ids)
        self.assertEqual(tuple(inputs.input_ids.tolist()), (5, 6, 7))
        for value in (torch.tensor([[5, 6, 7]]), torch.tensor([5., 6., 7.]), torch.tensor([5, 7])):
            with self.assertRaisesRegex(ValueError, 'mixed'):
                runner.prepare_mixed_batch(steps, value)
        with self.assertRaisesRegex(ValueError, 'mixed'):
            runner.prepare_mixed_batch(steps[:1], torch.tensor([5, 6]))
        runner.release('B')

    def test_unknown_decode_remap_is_rejected_before_index_select(self):
        from types import SimpleNamespace
        pipeline = self.pipeline()
        pipeline._active = True
        previous = SimpleNamespace(request_ids=('A',), tokens=torch.tensor([7]))
        step = Step('new', 'decode', (0,), (1,), (1,), (0, 1))
        with patch.object(torch.Tensor, 'index_select', side_effect=AssertionError('gathered -1')):
            with self.assertRaisesRegex(ValueError, 'absent'):
                pipeline._launch(1, (step,), previous)
        pipeline.close()
        self.assert_clean(pipeline)

    def test_callback_error_keeps_waiting_request_and_buffers_on_failed_drain(self):
        pipeline = self.pipeline()
        class BrokenStream:
            def synchronize(self):
                raise RuntimeError('drain failed')
        def fail(*args):
            pipeline.submit('C', (5,), 2)
            pipeline.is_npu, pipeline.stream = True, BrokenStream()
            raise RuntimeError('callback failed')
        with self.assertRaisesRegex(RuntimeError, 'drain failed'):
            pipeline.run({'A': (2, 3)}, 3, on_result=fail)
        self.assertTrue(pipeline._active)
        self.assertEqual(len(pipeline._states), 2)
        self.assertIn('C', pipeline._waiting)
        self.assertEqual(pipeline.requests['C'].status, Status.FAILED)
        self.assertNotIn('C', pipeline._slots)
        pipeline.is_npu = False
        pipeline.close()
        self.assert_clean(pipeline)

    def test_invalid_arrival_can_be_caught_without_damaging_existing_progress(self):
        pipeline = self.pipeline()
        with self.assertRaisesRegex(RuntimeError, 'callback'):
            pipeline.submit('C', (2,), 1)
        def arrive(index, steps, tokens, logits):
            if index == 0:
                for ids, count, eos in (((97,), 1, None), ((2,), 0, None), ((2,), 1, 97), ((2,), 33, None)):
                    with self.assertRaises(ValueError):
                        pipeline.submit('C', ids, count, eos)
                    self.assertNotIn('C', pipeline.requests)
        result = pipeline.run({'A': (2, 3)}, 2, on_result=arrive)
        self.assertEqual(len(result['A'].output_ids), 2)
        self.assert_clean(pipeline)
        for limit in (0, True):
            with self.assertRaises(ValueError):
                pipeline.run({'A': (2,)}, 2, max_num_seqs=limit)


if __name__ == '__main__':
    unittest.main()
