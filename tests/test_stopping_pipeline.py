"""A finished request may still have one submitted cache reader/writer."""

import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer import BatchEngine, Status
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from minideepinfer.stopping_pipeline import StoppingDecodePipeline


@unittest.skipUnless(AVAILABLE, 'install torch for stopping pipeline tests')
class StoppingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pipeline(self):
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        return StoppingDecodePipeline(ModelRunner(Qwen3_5ForCausalLM(config).eval(), 32))

    def baseline(self, pipeline, prompts, count=4):
        engine = BatchEngine(ModelRunner(pipeline.runner.model, 32), 32, len(prompts), 16)
        requests = {rid: engine.submit(tokens, count) for rid, tokens in prompts.items()}
        list(engine.run())
        return {rid: r.output_ids for rid, r in requests.items()}

    def assert_clean(self, pipeline):
        self.assertFalse(pipeline._active or pipeline._states or pipeline._slots or pipeline._retired)
        self.assertFalse(pipeline._in_result)
        self.assertFalse(pipeline.runner._prefixes or pipeline.runner.cache.linear_states)
        self.assertEqual(pipeline.pool.free_count, 32)

    def test_early_first_result_is_collected_once_before_second_forward(self):
        pipeline = self.pipeline()
        expected = self.baseline(pipeline, {'A': (2, 3, 4)})
        calls, observed = [], []
        forward = pipeline.runner.execute_main_model
        def capture(inputs):
            calls.append(inputs.steps)
            return forward(inputs)
        def observe(index, steps, tokens, logits):
            observed.append((index, len(calls)))
        with patch.object(pipeline.runner, 'execute_main_model', capture):
            result = pipeline.run({'A': (2, 3, 4)}, 4, early_first_result=True, on_result=observe)
        self.assertEqual(observed, [(0, 1), (1, 3), (2, 4), (3, 4)])
        self.assertEqual(result['A'].output_ids, expected['A'])
        self.assert_clean(pipeline)

    def test_early_partial_chunk_preserves_output_and_next_round_input(self):
        pipeline = self.pipeline()
        prompts = {'A': (2, 3, 4, 5, 6)}
        expected = self.baseline(pipeline, prompts)
        observed = []
        result = pipeline.run(prompts, 4, max_num_seqs=1, max_num_batched_tokens=2,
                              chunked_prefill=True, early_first_result=True,
                              on_result=lambda i,s,t,l: observed.append((i,t)))
        self.assertEqual(result['A'].output_ids, expected['A'])
        self.assertEqual(observed[:2], [(0,(None,)), (1,(None,))])
        self.assertEqual([i for i,t in observed], list(range(6)))
        self.assert_clean(pipeline)

    def test_early_eos_and_one_token_tail_can_admit_callback_arrival(self):
        pipeline = self.pipeline()
        expected = self.baseline(pipeline, {'A': (2, 3), 'B': (7,)})
        seen = []
        def arrive(index, steps, tokens, logits):
            seen.append(index)
            if index == 0:
                pipeline.submit('B', (7,), max_new_tokens=1)
        result = pipeline.run({'A': (2, 3)}, 4, early_first_result=True,
                              eos_token_ids={'A':expected['A'][0]}, on_result=arrive)
        self.assertEqual(result['A'].output_ids, expected['A'][:1])
        self.assertEqual(result['A'].finish_reason, 'eos')
        self.assertEqual(result['B'].output_ids, expected['B'][:1])
        self.assertEqual(seen, [0, 1])
        self.assert_clean(pipeline)

    def test_early_observer_error_drains_and_runner_remains_reusable(self):
        pipeline = self.pipeline()
        with self.assertRaisesRegex(ValueError, 'observer'):
            pipeline.run({'A': (2, 3)}, 4, early_first_result=True,
                         on_result=lambda *args: (_ for _ in ()).throw(ValueError('observer')))
        self.assert_clean(pipeline)
        result = pipeline.run({'A': (2, 3)}, 1, early_first_result=True)
        self.assertEqual(len(result['A'].output_ids), 1)
        self.assert_clean(pipeline)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            pipeline.run({'A': (2, 3)}, 1, early_first_result=1)

    def test_early_cancel_keeps_survivor_device_token_mapping(self):
        pipeline = self.pipeline()
        prompts = {'A': (2,3), 'B': (7,)}
        expected = self.baseline(pipeline, prompts)
        def cancel(index, steps, tokens, logits):
            if index == 0:
                pipeline.cancel('A')
        result = pipeline.run(prompts,4,early_first_result=True,on_result=cancel,
                              orders=(('A','B'),('B','A'),('A','B'),('B','A')))
        self.assertEqual(result['A'].output_ids,[])
        self.assertEqual(result['A'].finish_reason,'cancelled')
        self.assertEqual(result['B'].output_ids,expected['B'])
        self.assert_clean(pipeline)

    def test_waiting_admission_collects_length_end_without_exceeding_ownership_limit(self):
        pipeline = self.pipeline()
        prompts = {rid:(2,3,4,5,6) for rid in ('A','B','C','D')}
        expected = self.baseline(pipeline,prompts,count=8)
        runs = []
        for enabled in (False,True):
            plans = []
            forward = pipeline.runner.execute_main_model
            def capture(inputs):
                self.assertLessEqual(len(pipeline._slots),2)
                owned = [set(slots) for slots in pipeline._slots.values()]
                if len(owned)==2:
                    self.assertFalse(owned[0]&owned[1])
                if 'C' in {s.request_id for s in inputs.steps}:
                    self.assertNotIn('A',pipeline._slots)
                    self.assertEqual(pipeline.requests['A'].finish_reason,'length')
                plans.append(tuple(s.request_id for s in inputs.steps))
                return forward(inputs)
            with patch.object(pipeline.runner,'execute_main_model',capture):
                result = pipeline.run(prompts,8,max_num_seqs=2,max_num_batched_tokens=8,
                                      chunked_prefill=True,collect_before_admit=enabled)
            self.assertEqual({rid:r.output_ids for rid,r in result.items()},expected)
            runs.append(plans)
            self.assert_clean(pipeline)
        self.assertEqual([len(plans) for plans in runs],[18,17])
        self.assertNotIn('C',runs[0][8])
        self.assertIn('C',runs[1][8])

    def test_admission_observer_error_drains_before_runner_reuse(self):
        pipeline = self.pipeline()
        seen = []
        def fail(index,steps,tokens,logits):
            seen.append(index)
            if pipeline._issued['A'].sampled_tokens==2:
                raise ValueError('admission observer')
        with self.assertRaisesRegex(ValueError,'admission observer'):
            pipeline.run({'A':(2,3),'B':(7,)},2,max_num_seqs=1,collect_before_admit=True,on_result=fail)
        self.assertEqual(seen,[0])
        self.assert_clean(pipeline)
        result=pipeline.run({'A':(2,3)},1,collect_before_admit=True)
        self.assertEqual(len(result['A'].output_ids),1)
        self.assert_clean(pipeline)
        with self.assertRaisesRegex(ValueError,'boolean'):
            pipeline.run({'A':(2,3)},1,collect_before_admit=1)

    def test_admission_failed_wait_and_drain_retain_owner_and_waiting_request(self):
        from minideepinfer.npu_events import RecordedEvent
        pipeline=self.pipeline()
        class BrokenStream:
            def synchronize(self):
                raise RuntimeError('drain failed')
        def fail_wait(event):
            pipeline.is_npu,pipeline.stream=True,BrokenStream()
            raise RuntimeError('event wait failed')
        with patch.object(RecordedEvent,'synchronize',fail_wait):
            with self.assertRaisesRegex(RuntimeError,'drain failed'):
                pipeline.run({'A':(2,3),'B':(7,)},1,max_num_seqs=1,collect_before_admit=True)
        self.assertTrue(pipeline._active)
        self.assertEqual(set(pipeline._slots),{'A'})
        self.assertEqual(pipeline.pool.free_count,30)
        self.assertEqual(len(pipeline._states),1)
        self.assertTrue(pipeline.runner._prefixes and pipeline.runner.cache.linear_states)
        with self.assertRaisesRegex(RuntimeError,'idle'):
            pipeline.run({'new':(2,)},1)
        pipeline.is_npu=False
        pipeline.close()
        self.assert_clean(pipeline)

    def test_admission_and_first_result_options_cover_eos_and_capacity_blocking(self):
        pipeline=self.pipeline()
        reference=BatchEngine(ModelRunner(pipeline.runner.model,32),32,3,32)
        requests={rid:reference.submit(p,4) for rid,p in
                  {'A':(2,3),'B':(7,),'C':tuple(range(2,24))}.items()}
        list(reference.run())
        expected={rid:r.output_ids for rid,r in requests.items()}
        for first in (False,True):
            result=pipeline.run({'A':(2,3),'B':(7,),'C':tuple(range(2,24))},4,
                eos_token_ids={'A':expected['A'][0]},max_num_seqs=2,max_num_batched_tokens=4,
                chunked_prefill=True,collect_before_admit=True,early_first_result=first)
            self.assertEqual(result['A'].output_ids,expected['A'][:1])
            self.assertEqual(result['B'].output_ids,expected['B'])
            self.assertEqual(result['C'].output_ids,expected['C'])
            self.assert_clean(pipeline)

    def test_eos_keeps_one_later_round_then_releases_and_survivor_matches_sync(self):
        pipeline = self.pipeline()
        prompts = {'A': (2, 3, 4), 'B': (7,)}
        expected = self.baseline(pipeline, prompts)
        plans, visible, collection = [], [], []
        forward, collect = pipeline.runner.execute_main_model, pipeline._collect
        def capture(inputs):
            plans.append(tuple(s.request_id for s in inputs.steps))
            return forward(inputs)
        def capture_collect():
            before = {rid: event.query() for rid, event in pipeline._retired.items()}
            free = pipeline.pool.free_count
            collect()
            collection.append((before, free, pipeline.pool.free_count))
        def observe(index, steps, tokens, logits):
            visible.append(dict(zip((s.request_id for s in steps), tokens)))
        with patch.object(pipeline.runner, 'execute_main_model', capture), patch.object(
                pipeline, '_collect', capture_collect):
            result = pipeline.run(prompts, 4, eos_token_ids={'A': expected['A'][0]},
                                  orders=(('A', 'B'), ('B', 'A'), ('A', 'B'), ('B', 'A')),
                                  on_result=observe)
        self.assertEqual(plans, [('A', 'B'), ('B', 'A'), ('B',), ('B',)])
        self.assertEqual(result['A'].output_ids, expected['A'][:1])
        self.assertEqual(result['A'].num_cached_tokens, 3)
        self.assertEqual(result['A'].finish_reason, 'eos')
        self.assertEqual(result['B'].output_ids, expected['B'])
        self.assertEqual(result['B'].finish_reason, 'length')
        self.assertIsNone(visible[1]['A'])
        self.assertEqual(collection[0], ({'A': False}, 22, 22))
        self.assertTrue(any(before.get('A') and after > free for before, free, after in collection))
        self.assert_clean(pipeline)

    def test_all_eos_drains_the_unwanted_tail_without_a_third_forward(self):
        pipeline = self.pipeline()
        prompts = {'A': (2, 3), 'B': (7,)}
        expected = self.baseline(pipeline, prompts)
        seen = []
        with patch.object(pipeline.runner, 'execute_main_model', wraps=pipeline.runner.execute_main_model) as forward:
            result = pipeline.run(prompts, 4, eos_token_ids={rid: t[0] for rid, t in expected.items()},
                                  on_result=lambda index, steps, tokens, logits: seen.append(tokens))
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(seen[-1], (None, None))
        self.assertTrue(all(r.status == Status.FINISHED and r.finish_reason == 'eos' for r in result.values()))
        self.assert_clean(pipeline)

    def test_cancel_before_first_commit_discards_both_pending_outputs(self):
        pipeline = self.pipeline()
        prompts = {'A': (2, 3), 'B': (7,)}
        expected = self.baseline(pipeline, prompts)
        def cancel(index, steps, tokens, logits):
            if index == 0:
                self.assertEqual(len(pipeline._states), 2)
                self.assertTrue(pipeline.cancel('A'))
                self.assertFalse(pipeline.cancel('A'))
                self.assertFalse(pipeline._retired['A'].query())
                self.assertIn('A', pipeline.runner._prefixes)
                self.assertTrue(any(rid == 'A' for rid, layer in pipeline.runner.cache.linear_states))
        result = pipeline.run(prompts, 4, on_result=cancel)
        self.assertEqual(result['A'].status, Status.CANCELLED)
        self.assertEqual(result['A'].finish_reason, 'cancelled')
        self.assertEqual(result['A'].output_ids, [])
        self.assertEqual(result['A'].num_cached_tokens, 0)
        self.assertEqual(result['B'].output_ids, expected['B'])
        self.assert_clean(pipeline)

    def test_later_cancel_preserves_prior_commit_and_rejects_reentrant_run(self):
        pipeline = self.pipeline()
        expected = self.baseline(pipeline, {'A': (2, 3), 'B': (7,)})
        def cancel(index, steps, tokens, logits):
            if index == 1:
                original = pipeline.requests
                with self.assertRaisesRegex(RuntimeError, 'idle'):
                    pipeline.run({'new': (2,)}, 1)
                self.assertIs(pipeline.requests, original)
                with self.assertRaisesRegex(RuntimeError, 'cancel'):
                    pipeline.close()
                self.assertTrue(pipeline.cancel('A'))
        result = pipeline.run({'A': (2, 3), 'B': (7,)}, 4, on_result=cancel)
        self.assertEqual(result['A'].output_ids, expected['A'][:1])
        self.assertEqual(result['A'].num_cached_tokens, 2)
        self.assertEqual(result['B'].output_ids, expected['B'])
        self.assert_clean(pipeline)

    def test_eos_at_length_boundary_and_single_round_have_no_extra_forward(self):
        pipeline = self.pipeline()
        expected = self.baseline(pipeline, {'A': (2, 3)}, 1)
        with patch.object(pipeline.runner, 'execute_main_model', wraps=pipeline.runner.execute_main_model) as forward:
            result = pipeline.run({'A': (2, 3)}, 1, eos_token_ids={'A': expected['A'][0]})
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(result['A'].finish_reason, 'eos')
        self.assert_clean(pipeline)
        later = pipeline.run({'A': (2, 3)}, 1)
        self.assertEqual(later['A'].output_ids, expected['A'])
        self.assertEqual(later['A'].finish_reason, 'length')
        self.assert_clean(pipeline)

    def test_stopped_eos_cannot_be_changed_to_cancelled(self):
        pipeline = self.pipeline()
        expected = self.baseline(pipeline, {'A': (2, 3), 'B': (7,)})
        def cancel(index, steps, tokens, logits):
            if index == 1:
                self.assertFalse(pipeline.cancel('A'))
                with self.assertRaises(KeyError):
                    pipeline.cancel('unknown')
        result = pipeline.run({'A': (2, 3), 'B': (7,)}, 4,
                              eos_token_ids={'A': expected['A'][0]}, on_result=cancel)
        self.assertEqual(result['A'].finish_reason, 'eos')
        self.assert_clean(pipeline)

    def test_callback_exception_marks_live_requests_failed_and_drains(self):
        pipeline = self.pipeline()
        def fail(*args):
            raise RuntimeError('post-process failed')
        with self.assertRaisesRegex(RuntimeError, 'post-process failed'):
            pipeline.run({'A': (2, 3)}, 3, on_result=fail)
        self.assertEqual(pipeline.requests['A'].status, Status.FAILED)
        self.assertEqual(pipeline.requests['A'].output_ids, [])
        self.assertIn('post-process failed', pipeline.requests['A'].error)
        self.assert_clean(pipeline)

    def test_failed_drain_keeps_retirement_event_and_both_rounds(self):
        pipeline = self.pipeline()
        class BrokenStream:
            def synchronize(self):
                raise RuntimeError('drain failed')
        def fail(*args):
            pipeline.cancel('A')
            pipeline.is_npu, pipeline.stream = True, BrokenStream()
            raise RuntimeError('post-process failed')
        with self.assertRaisesRegex(RuntimeError, 'drain failed'):
            pipeline.run({'A': (2, 3), 'B': (7,)}, 3, on_result=fail)
        self.assertTrue(pipeline._active)
        self.assertEqual(len(pipeline._states), 2)
        self.assertEqual(pipeline.pool.free_count, 25)
        self.assertIn('A', pipeline._retired)
        self.assertFalse(pipeline._retired['A'].query())
        self.assertTrue(pipeline.runner._prefixes and pipeline.runner.cache.linear_states)
        self.assertEqual(pipeline.requests['A'].status, Status.CANCELLED)
        self.assertEqual(pipeline.requests['B'].status, Status.FAILED)
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            pipeline.run({'new': (2,)}, 1)
        pipeline.is_npu = False
        pipeline.close()
        self.assert_clean(pipeline)

    def test_unready_retirement_can_wait_until_final_stream_drain(self):
        pipeline = self.pipeline()
        expected = self.baseline(pipeline, {'A': (2, 3), 'B': (7,)})
        from minideepinfer.npu_events import RecordedEvent
        with patch.object(RecordedEvent, 'query', return_value=False):
            result = pipeline.run({'A': (2, 3), 'B': (7,)}, 4, eos_token_ids={'A': expected['A'][0]})
        self.assertEqual(result['B'].output_ids, expected['B'])
        self.assert_clean(pipeline)

    def test_invalid_eos_and_outside_callback_cancellation_are_rejected(self):
        pipeline = self.pipeline()
        for eos in ({'A': -1}, {'A': True}, {'A': 97}, {'unknown': 2}):
            with patch.object(pipeline.runner, 'execute_main_model') as forward:
                with self.assertRaises(ValueError):
                    pipeline.run({'A': (2, 3)}, 3, eos_token_ids=eos)
                forward.assert_not_called()
            self.assert_clean(pipeline)
        with self.assertRaisesRegex(RuntimeError, 'callback'):
            pipeline.cancel('A')

    def test_three_requests_shrink_in_two_stages_and_reordered_survivor_matches_sync(self):
        pipeline = self.pipeline()
        prompts = {'A': (2, 3, 4), 'B': (7,), 'C': (11, 12)}
        expected = self.baseline(pipeline, prompts)
        plans = []
        original = pipeline.runner.execute_main_model
        def forward(inputs):
            plans.append(tuple(s.request_id for s in inputs.steps))
            return original(inputs)
        def cancel(index, steps, tokens, logits):
            if index == 1:
                pipeline.cancel('B')
        with patch.object(pipeline.runner, 'execute_main_model', forward):
            result = pipeline.run(prompts, 4, eos_token_ids={'A': expected['A'][0]}, on_result=cancel,
                                  orders=(('A', 'B', 'C'), ('C', 'B', 'A'), ('A', 'B', 'C'), ('B', 'C', 'A')))
        self.assertEqual(plans, [('A', 'B', 'C'), ('C', 'B', 'A'), ('B', 'C'), ('C',)])
        self.assertEqual(result['A'].output_ids, expected['A'][:1])
        self.assertEqual(result['B'].output_ids, expected['B'][:1])
        self.assertEqual(result['C'].output_ids, expected['C'])
        self.assertEqual([r.finish_reason for r in result.values()], ['eos', 'cancelled', 'length'])
        self.assert_clean(pipeline)


if __name__ == '__main__':
    unittest.main()
