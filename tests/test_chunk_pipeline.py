"""Partial prompts advance issued cache positions without generating tokens."""

import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer import Engine, Status
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from minideepinfer.stopping_pipeline import StoppingDecodePipeline


@unittest.skipUnless(AVAILABLE, 'install torch for chunk pipeline checks')
class ChunkPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pipeline(self):
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        return StoppingDecodePipeline(ModelRunner(Qwen3_5ForCausalLM(config).eval(), 24))

    def reference(self, pipeline, prompt, count):
        # The original single-request Engine does not use the shared planner.
        runner = ModelRunner(pipeline.runner.model, 64)
        rows = []
        forward = runner.execute_main_model
        def capture(inputs):
            logits = forward(inputs)
            rows.append(logits.clone())
            return logits
        runner.execute_main_model = capture
        engine = Engine(runner, 64)
        request = engine.submit(prompt, count)
        list(engine.run())
        return request.output_ids, rows

    def assert_clean(self, pipeline):
        self.assertFalse(pipeline._active or pipeline._states or pipeline._slots or pipeline._retired)
        self.assertFalse(pipeline._issued or pipeline._last_use or pipeline._waiting)
        self.assertFalse(pipeline.runner._prefixes or pipeline.runner.cache.linear_states)
        self.assertEqual(pipeline.pool.free_count, 24)

    def test_chunks_arrivals_and_decode_priority_match_full_recompute_trajectory(self):
        pipeline = self.pipeline()
        a, b = tuple(range(2, 7)), tuple(range(7, 15))
        refs = {rid: self.reference(pipeline, prompt, count) for rid, prompt, count in
                (('A', a, 4), ('B', b, 4), ('C', a, 2))}
        plans, clocks, partials = [], [], []
        forward = pipeline.runner.execute_main_model
        def capture(inputs):
            self.assertLessEqual(len(inputs.input_ids), 4)
            plans.append([(s.request_id, s.phase, len(s.input_ids), s.is_last_prompt_chunk) for s in inputs.steps])
            return forward(inputs)
        def observe(index, steps, tokens, logits):
            if index == 0:
                clocks.append((pipeline.requests['A'].num_cached_tokens, pipeline._issued['A'].cached_tokens,
                               pipeline._issued['A'].sampled_tokens))
                pipeline.submit('C', a, 2)
            for step, token, row in zip(steps, tokens, logits):
                prompt = pipeline.requests[step.request_id].prompt_ids
                local = 0 if step.phase == 'prefill' else step.positions[0] - len(prompt) + 1
                wanted = refs[step.request_id][1][local]
                if step.phase == 'prefill':
                    wanted = wanted[step.positions[0]:step.positions[-1] + 1]
                torch.testing.assert_close(row, wanted, atol=1e-5, rtol=1e-4)
                if not step.is_last_prompt_chunk:
                    self.assertIsNone(token)
                    partials.append(step.request_id)
        with patch.object(pipeline.runner, 'execute_main_model', capture):
            result = pipeline.run({'A': a, 'B': b}, 4, on_result=observe,
                                  max_num_batched_tokens=4, chunked_prefill=True)
        self.assertEqual(clocks, [(0, 5, 1)])
        self.assertEqual(plans[0], [('A', 'prefill', 4, False)])
        self.assertEqual(plans[2], [('A', 'decode', 1, True), ('B', 'prefill', 3, False)])
        self.assertIn('B', partials)
        self.assertIn('C', partials)
        self.assertTrue(all(r.output_ids == refs[rid][0] for rid, r in result.items()))
        self.assert_clean(pipeline)

    def test_partial_sentinel_and_sampler_indices_are_prepared_before_forward(self):
        pipeline = self.pipeline()
        calls, sentinels = [], []
        forward, sample, tensor = pipeline.runner.execute_main_model, pipeline._sample, torch.tensor
        def tensors(values, *args, **kwargs):
            import sys
            if sys._getframe(1).f_code.co_name == '_launch':
                calls.append('indices')
            return tensor(values, *args, **kwargs)
        def capture(inputs):
            calls.append('forward')
            state = pipeline._states[-1]
            if tuple(s.is_last_prompt_chunk for s in inputs.steps) == (True, False):
                self.assertEqual(state.sample_indices.tolist(), [0])
            elif not any(s.is_last_prompt_chunk for s in inputs.steps):
                self.assertIsNone(state.sample_indices)
            return forward(inputs)
        def sampling(state):
            sample(state)
            sentinels.append((tuple(s.is_last_prompt_chunk for s in state.steps), state.host_tokens.tolist()))
        with patch.object(pipeline.runner, 'execute_main_model', capture), patch.object(
                pipeline, '_sample', sampling), patch.object(torch, 'tensor', tensors):
            pipeline.run({'A': (2, 3, 4, 5, 6), 'B': tuple(range(7, 15))}, 2,
                         max_num_batched_tokens=4, chunked_prefill=True)
        self.assertEqual(sentinels[0], ((False,), [-1]))
        self.assertTrue(any(flags == (True, False) and tokens[1] == -1 for flags, tokens in sentinels))
        self.assertIn(['indices', 'forward'], [calls[i:i + 2] for i in range(len(calls) - 1)])
        self.assert_clean(pipeline)

    def test_cancel_partial_prompt_preserves_zero_outputs_and_reclaims_after_last_use(self):
        pipeline = self.pipeline()
        def cancel(index, steps, tokens, logits):
            if index == 0:
                pipeline.cancel('A')
                self.assertFalse(pipeline._retired['A'].query())
        result = pipeline.run({'A': (2, 3, 4, 5, 6), 'B': (7, 8, 9)}, 3,
                              max_num_batched_tokens=4, chunked_prefill=True, on_result=cancel)
        self.assertEqual(result['A'].status, Status.CANCELLED)
        self.assertEqual((result['A'].output_ids, result['A'].num_cached_tokens), ([], 0))
        self.assertEqual(len(result['B'].output_ids), 3)
        self.assert_clean(pipeline)

    def test_all_partial_batches_advance_until_one_final_sample(self):
        pipeline = self.pipeline()
        seen = []
        result = pipeline.run({'A': (2, 3, 4, 5, 6)}, 1, max_num_batched_tokens=2, chunked_prefill=True,
                              on_result=lambda index, steps, tokens, logits: seen.append(tokens))
        self.assertEqual(seen[:2], [(None,), (None,)])
        self.assertEqual(len(result['A'].output_ids), 1)
        self.assertEqual(result['A'].num_cached_tokens, 5)
        self.assert_clean(pipeline)

    def test_partial_host_error_and_failed_drain_keep_progress_and_last_use(self):
        pipeline = self.pipeline()
        class BrokenStream:
            def synchronize(self):
                raise RuntimeError('drain failed')
        def fail(*args):
            pipeline.cancel('A')
            pipeline.is_npu, pipeline.stream = True, BrokenStream()
            raise RuntimeError('partial result failed')
        with self.assertRaisesRegex(RuntimeError, 'drain failed'):
            pipeline.run({'A': tuple(range(2, 10))}, 2, max_num_batched_tokens=2,
                         chunked_prefill=True, on_result=fail)
        self.assertEqual(pipeline._issued['A'].cached_tokens, 4)
        self.assertEqual(pipeline._issued['A'].sampled_tokens, 0)
        self.assertIs(pipeline._last_use['A'], pipeline._states[-1].ready)
        self.assertTrue(pipeline._slots and pipeline._retired and pipeline._states)
        pipeline.is_npu = False
        pipeline.close()
        self.assert_clean(pipeline)

    def test_whole_prompt_and_invalid_budget_rejected_before_execution(self):
        pipeline = self.pipeline()
        for budget, chunked in ((1, True), (4, 1), (True, True), (4, False)):
            with patch.object(pipeline.runner, 'execute_main_model') as forward:
                with self.assertRaises(ValueError):
                    pipeline.run({'A': (2, 3, 4, 5, 6), 'B': (7,)}, 2,
                                 max_num_batched_tokens=budget, chunked_prefill=chunked)
                forward.assert_not_called()
        self.assert_clean(pipeline)

    def test_invalid_partial_sentinel_is_rejected_before_cpu_progress_commit(self):
        pipeline = self.pipeline()
        sample = pipeline._sample
        def corrupt(state):
            sample(state)
            if state.index == 0:
                state.host_tokens[0] = 0
        with patch.object(pipeline, '_sample', corrupt):
            with self.assertRaisesRegex(ValueError, 'invalid sampled'):
                pipeline.run({'A': (2, 3, 4, 5, 6)}, 2, max_num_batched_tokens=2, chunked_prefill=True)
        self.assertEqual(pipeline.requests['A'].num_cached_tokens, 0)
        self.assertEqual(pipeline.requests['A'].output_ids, [])
        self.assertEqual(pipeline.requests['A'].status, Status.FAILED)
        self.assert_clean(pipeline)

    def test_partial_sample_cannot_be_used_as_a_decode_source(self):
        from types import SimpleNamespace
        from minideepinfer import Step
        pipeline = self.pipeline()
        pipeline._active = True
        partial = Step('A', 'prefill', (2,), (0,), (0,), (0,), False)
        previous = SimpleNamespace(request_ids=('A',), steps=(partial,), tokens=torch.tensor([-1]))
        decode = Step('A', 'decode', (0,), (1,), (1,), (0, 1))
        with self.assertRaisesRegex(ValueError, 'no sampled token'):
            pipeline._launch(1, (decode,), previous)
        pipeline.close()
        self.assert_clean(pipeline)


if __name__ == '__main__':
    unittest.main()
