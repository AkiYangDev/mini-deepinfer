"""A real device-token dependency, with a deliberately fixed batch lifetime."""

import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer.decode_pipeline import FixedDecodePipeline
    from minideepinfer.engine import Step
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM


@unittest.skipUnless(AVAILABLE, 'install torch for device decode tests')
class DecodePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def runner(self, hybrid=True):
        torch.manual_seed(2036)
        if hybrid:
            config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                                  ('linear_attention',) * 3 + ('full_attention',),
                                  max_position_embeddings=64, partial_rotary_factor=0.5)
            model = Qwen3_5ForCausalLM(config)
        else:
            model = Qwen2ForCausalLM(Qwen2Config(97, 32, 64, 2, 4, 2, max_position_embeddings=64))
        return ModelRunner(model.eval(), 32)

    def assert_clean(self, pipeline):
        self.assertFalse(pipeline._active)
        self.assertFalse(pipeline._states)
        self.assertFalse(pipeline._slots)
        self.assertEqual(pipeline.pool.free_count, 32)
        self.assertFalse(pipeline.runner._prefixes)
        self.assertFalse(getattr(pipeline.runner.cache, 'linear_states', {}))

    def test_reordered_device_continuation_matches_synchronous_greedy_and_logits(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                runner = self.runner(hybrid)
                reference = ModelRunner(runner.model, 32)
                pipeline = FixedDecodePipeline(runner)
                prompts = {'A': (2, 3, 4), 'B': (7,)}
                expected = {rid: [] for rid in prompts}
                slots = {'A': tuple(range(7)), 'B': tuple(range(7, 12))}
                orders = (('A', 'B'), ('B', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'A'))
                calls = []
                original = runner.execute_main_model
                def forward(inputs):
                    calls.append('forward')
                    # Only the prior result is uncommitted; at most two rounds live.
                    self.assertLessEqual(len(pipeline._states), 2)
                    return original(inputs)
                def observe(index, steps, tokens, logits):
                    calls.append('result')
                    plans = []
                    for step in steps:
                        rid = step.request_id
                        prefix = prompts[rid] + tuple(expected[rid])
                        count = len(prompts[rid]) if index == 0 else 1
                        start = len(prefix) - count
                        plans.append(Step(rid, 'prefill' if index == 0 else 'decode', prefix[-count:],
                                          tuple(range(start, len(prefix))), slots[rid][start:len(prefix)],
                                          slots[rid][:len(prefix)]))
                    rows = reference.forward_batch(tuple(plans))
                    for step, token, actual, wanted in zip(steps, tokens, logits, rows):
                        torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
                        self.assertEqual(token, int(wanted[-1].argmax()))
                        expected[step.request_id].append(token)
                with patch.object(runner, 'execute_main_model', forward):
                    actual = pipeline.run(prompts, 5, orders=orders, on_result=observe)
                self.assertEqual(actual, expected)
                self.assertEqual(calls, ['forward', 'forward', 'result', 'forward', 'result',
                                         'forward', 'result', 'forward', 'result', 'result'])
                self.assert_clean(pipeline)

    def test_next_forward_does_not_read_or_upload_a_previous_host_token(self):
        runner = self.runner()
        pipeline = FixedDecodePipeline(runner)
        original = runner.prepare_batch
        seen = []
        def prepare(steps, *, device_input_ids=None):
            if steps[0].phase == 'decode':
                self.assertIsNotNone(device_input_ids)
                self.assertEqual(steps[0].input_ids, (0,))
                self.assertTrue(torch.equal(device_input_ids, pipeline._states[0].tokens))
                self.assertFalse(seen)  # No earlier result has reached the CPU callback.
                with patch.object(torch, 'tensor', wraps=torch.tensor) as tensor:
                    inputs = original(steps, device_input_ids=device_input_ids)
                self.assertIs(inputs.input_ids, device_input_ids)
                # Positions / slots / contexts / row IDs still upload; input IDs do not.
                self.assertEqual(tensor.call_count, 6)
                return inputs
            return original(steps)
        with patch.object(runner, 'prepare_batch', prepare):
            pipeline.run({'A': (2, 3)}, 2, on_result=lambda *args: seen.append(args[0]))
        self.assertEqual(seen, [0, 1])
        self.assert_clean(pipeline)

    def test_one_token_tail_and_reusing_the_same_runner(self):
        pipeline = FixedDecodePipeline(self.runner())
        first = pipeline.run({'A': (2, 3)}, 1)
        self.assertEqual(len(first['A']), 1)
        self.assert_clean(pipeline)
        self.assertEqual(pipeline.run({'A': (2, 3)}, 1), first)
        self.assert_clean(pipeline)

    def test_host_callback_failure_drains_a_later_forward_before_reuse(self):
        pipeline = FixedDecodePipeline(self.runner())
        def fail(*args):
            self.assertEqual(len(pipeline._states), 2)
            raise RuntimeError('CPU result handler failed')
        with self.assertRaisesRegex(RuntimeError, 'CPU result handler'):
            pipeline.run({'A': (2, 3)}, 3, on_result=fail)
        self.assert_clean(pipeline)
        pipeline.run({'A': (2, 3)}, 2)
        self.assert_clean(pipeline)

    def test_failed_drain_keeps_both_rounds_and_prevents_runner_reuse(self):
        pipeline = FixedDecodePipeline(self.runner())
        # Exercise NPU control flow without pretending this is a device-fault test.
        class BrokenStream:
            def synchronize(self):
                raise RuntimeError('drain failed')
        def fail(*args):
            pipeline.is_npu = True
            pipeline.stream = BrokenStream()
            raise RuntimeError('host failed')
        with self.assertRaisesRegex(RuntimeError, 'drain failed'):
            pipeline.run({'A': (2, 3)}, 3, on_result=fail)
        self.assertTrue(pipeline._active)
        self.assertEqual(len(pipeline._states), 2)
        self.assertEqual(pipeline.pool.free_count, 28)
        self.assertTrue(pipeline.runner._prefixes)
        self.assertTrue(pipeline.runner.cache.linear_states)
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            pipeline.run({'A': (2, 3)}, 1)
        pipeline.is_npu = False  # Restore the actual CPU backend for cleanup.
        pipeline.close()
        self.assert_clean(pipeline)

    def test_forward_error_keeps_buffers_until_drain_and_then_cleans(self):
        pipeline = FixedDecodePipeline(self.runner())
        original = pipeline.runner.execute_main_model
        def fail(inputs):
            output = original(inputs)
            if inputs.steps[0].phase == 'decode':
                raise RuntimeError('forward submission failed')
            return output
        with patch.object(pipeline.runner, 'execute_main_model', fail):
            with self.assertRaisesRegex(RuntimeError, 'forward submission'):
                pipeline.run({'A': (2, 3)}, 3)
        self.assert_clean(pipeline)

    def test_invalid_fixed_workload_is_rejected_before_any_submission(self):
        pipeline = FixedDecodePipeline(self.runner())
        for prompts, count, orders in (({}, 2, None), ({'A': ()}, 2, None),
                                      ({'A': (97,)}, 2, None), ({'A': (2,)}, 0, None),
                                      ({'A': (2,)}, 2, (('A',), ('B',))),
                                      ({'A': (2,)}, 33, None)):
            with self.subTest(prompts=prompts, count=count):
                with patch.object(pipeline.runner, 'execute_main_model') as forward:
                    with self.assertRaises((ValueError, MemoryError)):
                        pipeline.run(prompts, count, orders=orders)
                    forward.assert_not_called()
                self.assert_clean(pipeline)

    def test_device_input_override_rejects_wrong_shapes_and_prefill(self):
        runner = self.runner()
        prefill = (Step('A', 'prefill', (2,), (0,), (0,), (0,)),)
        runner.run_batch(prefill)
        decode = (Step('A', 'decode', (0,), (1,), (1,), (0, 1)),)
        for steps, tensor in ((prefill, torch.tensor([2])), (decode, torch.tensor([[2]])),
                              (decode, torch.tensor([2.])),
                              ((Step('A', 'decode', (3,), (1,), (1,), (0, 1)),), torch.tensor([2]))):
            with self.assertRaisesRegex(ValueError, 'placeholder'):
                runner.prepare_batch(steps, device_input_ids=tensor)
        runner.release('A')

    def test_an_occupied_runner_is_rejected_without_clearing_its_state(self):
        runner = self.runner()
        runner.run_batch((Step('live', 'prefill', (2,), (0,), (31,), (31,)),))
        prefixes = dict(runner._prefixes)
        states = dict(runner.cache.linear_states)
        pipeline = FixedDecodePipeline(runner)
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            pipeline.run({'A': (3,)}, 2)
        pipeline.close()
        self.assertEqual(runner._prefixes, prefixes)
        for key, state in states.items():
            self.assertIs(runner.cache.linear_states[key], state)
        self.assertEqual(pipeline.pool.free_count, 32)
        runner.release('live')


if __name__ == '__main__':
    unittest.main()
