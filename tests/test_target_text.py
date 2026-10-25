import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minideepinfer.generate import main, parse_args
from minideepinfer.target_text import run_target_text


class Tokenizer:
    eos_token_id = None

    def encode(self, text):
        return [int(t) for t in text.split()]

    def apply_chat_template(self, messages, **options):
        self.options = options
        return [1]+self.encode(messages[0]['content'])+[2]

    def decode(self, tokens, **options):
        return ' '.join(map(str, tokens))


class TargetArgumentsTests(unittest.TestCase):
    def test_invalid_combinations_fail_before_loading(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'torch' or name.startswith('transformers'):
                raise AssertionError('invalid arguments must fail before model imports')
            return original(name, *args, **kwargs)
        for extra in (['--draft-model', 'draft'], ['--draft-width', '0'], ['--check-greedy'],
                      ['--speculative'], ['--pipeline'], ['--deferred'], ['--prepare-during-step'],
                      ['--max-num-seqs', '1'], ['--max-batched-tokens', '4'], ['--chunked-prefill'],
                      ['--dtype', 'bfloat16'], ['--device', 'cuda:0'], ['--prompt', 'a', '--prompt', 'b'],
                      ['--max-new-tokens', '0'], ['--cache-capacity', '0'], ['--cancel-after-tokens', '0'],
                      ['--temperature', 'nan'], ['--top-k', '20'], ['--seed', '1']):
            with self.subTest(extra=extra), patch('builtins.__import__', guarded), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(['--model', 'unused', '--single', *extra])
            self.assertEqual(raised.exception.code, 2)
        args = parse_args(['--model', 'target', '--single', '--temperature', '.7', '--top-k', '20',
                           '--top-p', '.9', '--stream', '--trace-rounds', '--cancel-after-tokens', '2'])
        self.assertEqual(args.draft_width, 0)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'install torch for target text checks')
class TargetTextTests(unittest.TestCase):
    def model(self, hybrid=True):
        import torch
        from minideepinfer.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
        from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        if not hybrid:
            return Qwen2ForCausalLM(Qwen2Config(97, 16, 24, 2, 2, 1, max_position_embeddings=64)).eval()
        config = Qwen3_5Config(97, 16, 24, 2, 2, 1, 8, 1, 2, 4, 4,
                               ('linear_attention', 'full_attention'), max_position_embeddings=64,
                               partial_rotary_factor=.5, tie_word_embeddings=False)
        return Qwen3_5ForCausalLM(config).eval()

    def test_sampled_outputs_match_independent_full_prefix_and_keep_global_rng(self):
        import torch
        from minideepinfer.sampling import TemperatureSampler
        from minideepinfer.speculative import CachedTarget
        for hybrid in (False, True):
            model = self.model(hybrid)
            for filters in ({}, dict(top_k=20, top_p=.9)):
                with self.subTest(hybrid=hybrid, filters=filters):
                    state = torch.random.get_rng_state().clone()
                    sampler, expected = TemperatureSampler(.7, 2026, **filters), []
                    for _ in range(6):
                        cold = CachedTarget(model, tuple(range(7)), 7)
                        expected.append(sampler.sample(cold.forward((3, 4)+tuple(expected))[-1]))
                    groups = []
                    with patch('minideepinfer.model_draft.ModelDraft', side_effect=AssertionError('no draft')):
                        result = run_target_text(model, Tokenizer(), '3 4', raw=True, max_new_tokens=6,
                                 cache_capacity=32, temperature=.7, on_commit=groups.append, **filters)
                    self.assertEqual(result['output_ids'], expected)
                    self.assertEqual(groups, [(token,) for token in expected])
                    self.assertEqual(result['target_forward_calls'], 6)
                    self.assertEqual(result['target_input_tokens'], 7)
                    self.assertEqual(result['direct_target_tokens'], 6)
                    self.assertEqual(result['proposed_draft_tokens'], 0)
                    self.assertEqual(result['free_slots'], 7)
                    self.assertEqual(result['cache_budget'], 32)
                    torch.testing.assert_close(torch.random.get_rng_state(), state)

    def test_default_greedy_matches_engine_and_chat_template(self):
        from minideepinfer.engine import Engine
        from minideepinfer.model_runner import ModelRunner
        model, tokenizer = self.model(), Tokenizer()
        result = run_target_text(model, tokenizer, '3 4', max_new_tokens=5, thinking=True)
        engine = Engine(ModelRunner(model, 8), 8)
        request = engine.submit([1, 3, 4, 2], 5)
        list(engine.run())
        self.assertEqual(result['output_ids'], request.output_ids)
        self.assertEqual(result['finish_reason'], request.finish_reason)
        self.assertNotIn('sampling', result)
        self.assertTrue(tokenizer.options['enable_thinking'])
        self.assertEqual(result['free_slots'], 8)

    def test_reusable_verifier_checks_all_three_stops(self):
        from minideepinfer.verify_target_text import check_target
        report = check_target(self.model(), Tokenizer(), {}, prompt='3 4')
        self.assertTrue(report['passed'])
        self.assertEqual(report['total_forward_calls'], 15)
        self.assertEqual(report['total_input_tokens'], 39)
        self.assertTrue(all(case['snapshot_count'] == case['snapshot_bytes_total'] == 0
                            for case in report['cases'].values()))

    def test_cancellation_before_prefill_and_after_one_token(self):
        model = self.model()
        with patch.object(model, 'forward') as forward:
            result = run_target_text(model, Tokenizer(), '3 4', raw=True, should_cancel=lambda: True)
            forward.assert_not_called()
        self.assertEqual(result['output_ids'], [])
        self.assertEqual(result['finish_reason'], 'cancelled')
        self.assertEqual(result['free_slots'], result['cache_capacity'])
        groups = []
        result = run_target_text(model, Tokenizer(), '3 4', raw=True, on_commit=groups.append,
                                 should_cancel=lambda: bool(groups), temperature=.7)
        self.assertEqual(len(result['output_ids']), 1)
        self.assertEqual(result['target_forward_calls'], 1)
        self.assertEqual(result['finish_reason'], 'cancelled')
        self.assertEqual(result['free_slots'], result['cache_capacity'])

    def test_eos_and_length_take_precedence_over_cancellation(self):
        model = self.model()
        first = run_target_text(model, Tokenizer(), '3 4', raw=True, max_new_tokens=1)['output_ids'][0]
        for eos, maximum, reason in ((first, 8, 'eos'), (None, 1, 'length')):
            tokenizer, groups = Tokenizer(), []
            tokenizer.eos_token_id = eos
            result = run_target_text(model, tokenizer, '3 4', raw=True, max_new_tokens=maximum,
                                     on_commit=groups.append, should_cancel=lambda: bool(groups))
            self.assertEqual(result['output_ids'], [first])
            self.assertEqual(result['finish_reason'], reason)

    def test_consumer_failure_releases_slots_and_hybrid_state(self):
        from minideepinfer.hybrid_cache import HybridCache
        from minideepinfer.kv_cache import SlotPool
        released = []
        original_release = SlotPool.release
        original_state_release = HybridCache.release
        def release(pool, request):
            original_release(pool, request)
            released.append(pool.free_count == pool.capacity)
        def state_release(cache, request):
            original_state_release(cache, request)
            released.append(not cache.linear_states)
        def fail(tokens):
            raise RuntimeError('consumer failed')
        with patch.object(SlotPool, 'release', release), patch.object(HybridCache, 'release', state_release):
            with self.assertRaisesRegex(RuntimeError, 'consumer failed'):
                run_target_text(self.model(), Tokenizer(), '3 4', raw=True, on_commit=fail)
        self.assertEqual(released, [True, True])

    def test_invalid_request_does_not_forward(self):
        model = self.model()
        for prompt, options in (('97', {}), ('3 4', dict(cache_capacity=2)),
                                ('3 4', dict(max_new_tokens=64)), ('3 4', dict(top_p=.9))):
            with patch.object(model, 'forward') as forward, self.assertRaises(ValueError):
                run_target_text(model, Tokenizer(), prompt, raw=True, **options)
            forward.assert_not_called()

    def test_cli_loads_one_model_streams_cancels_and_saves_report(self):
        from minideepinfer.model_runner import ModelRunner
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            report, stdout = Path(directory)/'report.json', io.StringIO()
            argv = ['--model', 'target', '--single', '--device', 'cpu', '--raw', '--prompt', '3 4',
                    '--temperature', '.7', '--top-k', '20', '--top-p', '.9', '--max-new-tokens', '6',
                    '--stream', '--cancel-after-tokens', '2', '--trace-rounds', '--output', str(report)]
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                    patch.object(ModelRunner, 'from_pretrained', return_value=ModelRunner(model, 1)) as load, \
                    patch('minideepinfer.model_draft.ModelDraft', side_effect=AssertionError('no draft')), \
                    contextlib.redirect_stdout(stdout):
                result = main(argv)
            load.assert_called_once()
            self.assertEqual(json.loads(report.read_text(encoding='utf-8')), result)
            self.assertEqual(result['finish_reason'], 'cancelled')
            self.assertEqual(len(result['output_ids']), 2)
            self.assertEqual(result['free_slots'], result['cache_capacity'])
            self.assertNotIn('draft_work', result)
            self.assertNotIn('draft_model', result)
            self.assertNotIn('passed', result)
            self.assertEqual(len(result['source_sha256']), 12)
            self.assertIn(result['text'], stdout.getvalue())
            self.assertIn('round 1: input=', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
