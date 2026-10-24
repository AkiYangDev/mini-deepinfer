"""Generated schedules against independent, whole-prompt single-request runs."""

import importlib.util
import random
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer import BatchEngine, Engine, Status
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from minideepinfer.stopping_pipeline import StoppingDecodePipeline


@unittest.skipUnless(AVAILABLE, 'install torch for generated scheduling checks')
class SchedulerBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(2041)
        config = Qwen3_5Config(41, 16, 24, 2, 2, 1, 8, 1, 2, 4, 4,
                              ('linear_attention', 'full_attention'),
                              max_position_embeddings=32, partial_rotary_factor=0.5)
        cls.model = Qwen3_5ForCausalLM(config).eval()

    def oracle(self, prompt, count):
        # Engine runs a full prompt and has no shared planner or cross-round loop.
        runner = ModelRunner(self.model, 24)
        engine = Engine(runner, 24)
        request = engine.submit(prompt, count)
        list(engine.run())
        self.assertEqual(engine.cache.free_count, 24)
        return request.output_ids

    def assert_clean(self, pipeline):
        self.assertEqual(pipeline.pool.free_count, pipeline.pool.capacity)
        self.assertFalse(pipeline._active or pipeline._states or pipeline._slots
                         or pipeline._waiting or pipeline._retired or pipeline._issued or pipeline._last_use)
        self.assertFalse(pipeline.runner._prefixes or pipeline.runner.cache.linear_states)

    def test_generated_arrivals_cancellations_and_eos_preserve_oracle_prefixes(self):
        coverage = set()
        for seed in range(32):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                count, limit = rng.randint(1, 4), rng.randint(1, 3)
                budget = rng.randint(limit, 8)
                prompts = {rid: tuple(rng.randrange(2, 41) for _ in range(rng.randint(1, 9)))
                           for rid in ('A', 'B', 'C', 'D')}
                capacity = rng.randint(max(len(p) + count - 1 for p in prompts.values()), 24)
                refs = {rid: self.oracle(p, count) for rid, p in prompts.items()}
                pipeline = StoppingDecodePipeline(ModelRunner(self.model, capacity))
                cancelled, emitted, issued = {}, {rid: [] for rid in prompts}, {}
                eos = {'B': refs['B'][rng.randrange(count)]} if seed % 3 == 0 else {}
                original = pipeline.runner.execute_main_model

                def forward(inputs):
                    self.assertLessEqual(len(inputs.steps), limit)
                    self.assertLessEqual(len(inputs.input_ids), budget)
                    owners = [slot for slots in pipeline._slots.values() for slot in slots]
                    self.assertEqual(len(owners), len(set(owners)))
                    self.assertEqual(len(owners) + pipeline.pool.free_count, capacity)
                    phases = [s.phase for s in inputs.steps]
                    self.assertEqual(phases, sorted(phases))  # decode, then prefill
                    for step in inputs.steps:
                        rid = step.request_id
                        self.assertEqual(step.positions[0], issued.get(rid, 0))
                        issued[rid] = step.positions[-1] + 1
                        self.assertLessEqual(issued[rid], len(prompts[rid]) + count - 1)
                        if not step.is_last_prompt_chunk:
                            coverage.add('partial')
                    return original(inputs)

                def observe(index, steps, tokens, logits):
                    self.assertLess(index, 100, 'a finite workload must drain')
                    for rid, request in pipeline.requests.items():
                        self.assertEqual(request.output_ids, emitted[rid])
                    if index == 0:
                        pipeline.submit('C', prompts['C'], count)
                        pipeline.submit('D', prompts['D'], count)
                        self.assertNotIn('C', pipeline._slots)
                        coverage.add('arrival')
                    target = ('A', 'C', 'D')[seed % 3]
                    if index == seed % 4 and pipeline.requests[target].status in (Status.WAITING, Status.RUNNING):
                        request = pipeline.requests[target]
                        coverage.add('cancel_' + request.status.value)
                        cancelled[target] = (list(request.output_ids), request.num_cached_tokens)
                        pipeline.cancel(target)
                    for step, token in zip(steps, tokens):
                        request = pipeline.requests[step.request_id]
                        if not step.is_last_prompt_chunk:
                            self.assertIsNone(token)
                        elif request.status == Status.RUNNING:
                            wanted = refs[step.request_id][len(request.output_ids)]
                            self.assertEqual(token, wanted)
                            emitted[step.request_id].append(token)

                with patch.object(pipeline.runner, 'execute_main_model', forward):
                    result = pipeline.run({'A': prompts['A'], 'B': prompts['B']}, count,
                                          eos_token_ids=eos, max_num_seqs=limit,
                                          max_num_batched_tokens=budget, chunked_prefill=True,
                                          on_result=observe)
                self.assertEqual(set(result), set(prompts))
                for rid, request in result.items():
                    if rid in cancelled:
                        self.assertEqual(request.status, Status.CANCELLED)
                        self.assertEqual((request.output_ids, request.num_cached_tokens), cancelled[rid])
                    else:
                        expected = refs[rid]
                        if rid in eos:
                            expected = expected[:expected.index(eos[rid]) + 1]
                            self.assertEqual(request.finish_reason, 'eos')
                            coverage.add('eos')
                        else:
                            self.assertEqual(request.finish_reason, 'length')
                        self.assertEqual(request.output_ids, expected)
                        self.assertEqual(request.num_cached_tokens, len(prompts[rid]) + len(expected) - 1)
                self.assert_clean(pipeline)
        self.assertTrue({'partial', 'arrival', 'cancel_waiting', 'cancel_running', 'eos'} <= coverage)

    def test_exact_capacity_one_output_and_runner_reuse_across_budgets(self):
        prompt = (2, 3, 4, 5, 6, 7, 8)
        expected = self.oracle(prompt, 1)
        runner = ModelRunner(self.model, len(prompt))
        for budget in (1, 2, 3, 7, 12):
            with self.subTest(budget=budget):
                pipeline = StoppingDecodePipeline(runner)
                result = pipeline.run({'A': prompt, 'B': prompt}, 1, max_num_seqs=1,
                                      max_num_batched_tokens=budget, chunked_prefill=True)
                self.assertEqual([r.output_ids for r in result.values()], [expected, expected])
                self.assert_clean(pipeline)

    def test_sync_chunks_match_whole_prompt_oracle_across_sizes_and_budgets(self):
        for seed in range(16):
            with self.subTest(seed=seed):
                rng = random.Random(100 + seed)
                count, limit = rng.randint(1, 4), rng.randint(1, 3)
                prompts = [tuple(rng.randrange(2, 41) for _ in range(rng.randint(1, 9))) for _ in range(4)]
                capacity = max(len(p) + count - 1 for p in prompts) + rng.randint(0, 8)
                runner = ModelRunner(self.model, capacity)
                engine = BatchEngine(runner, capacity, limit, rng.randint(limit, 8), True)
                requests = [engine.submit(p, count) for p in prompts]
                list(engine.run())
                self.assertEqual([r.output_ids for r in requests], [self.oracle(p, count) for p in prompts])
                self.assertEqual(engine.cache.free_count, capacity)
                self.assertFalse(runner._prefixes or runner.cache.linear_states)


if __name__ == '__main__':
    unittest.main()
