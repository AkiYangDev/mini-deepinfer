import importlib.util
import math
import random
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('torch') is not None


class SamplingArgumentTests(unittest.TestCase):
    def test_bad_settings_fail_before_importing_models_or_loading_weights(self):
        from minideepinfer.sampling import TemperatureSampler
        for temperature in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                TemperatureSampler(temperature)
        for seed in (-1, 2**63, True, 1.5):
            with self.assertRaises(ValueError):
                TemperatureSampler(1, seed)
        from minideepinfer.generate import parse_args
        import contextlib
        import io
        base = ['--model', 'target', '--speculative', '--draft-model', 'draft']
        for extra in (['--temperature', '0'], ['--temperature', 'nan'], ['--seed', '3'],
                      ['--temperature', '1'], ['--device', 'cpu', '--temperature', '1', '--check-greedy']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base+extra)
        args = parse_args(base+['--device', 'cpu', '--temperature', '.7', '--seed', '8'])
        self.assertEqual((args.temperature, args.seed), (.7, 8))


class ScriptedRandom:
    def __init__(self, *values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


@unittest.skipUnless(AVAILABLE, 'install torch for probability sampling')
class ProbabilityTests(unittest.TestCase):
    def test_temperature_sparse_support_and_invalid_probabilities(self):
        import torch
        from minideepinfer.sampling import TemperatureSampler, distribution, draw
        logits = torch.tensor([0., math.log(4.)], dtype=torch.float64)
        p = TemperatureSampler(2).probabilities(logits)
        torch.testing.assert_close(p, torch.tensor([1/3, 2/3], dtype=torch.float64))
        self.assertEqual(draw([0., 1., 0.], ScriptedRandom(0)), 1)
        self.assertEqual(draw([0., 1., 0.], ScriptedRandom(math.nextafter(1., 0.))), 1)
        for values in ([0., 0.], [-.1, 1.1], [float('nan'), 1.], [[1.]], [.2, .3]):
            with self.assertRaises(ValueError):
                distribution(values)
        for values in ([float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                TemperatureSampler().probabilities(torch.tensor(values))

    def test_rejection_uses_residual_not_target_and_stops_at_first_reject(self):
        from minideepinfer.sampling import SampledDraft, verify_sampled
        proposal = SampledDraft((0, 1), ([.8, .2], [.5, .5]))
        result, checks = verify_sampled(proposal, ([.2, .8], [.9, .1], [.5, .5]),
                                         3, ScriptedRandom(.5, .1))
        self.assertEqual((result.tokens, result.accepted_draft, result.final_role), ((1,), 0, 'correction'))
        self.assertEqual(len(checks), 1)
        self.assertAlmostEqual(checks[0]['acceptance_probability'], .25)

    def test_acceptance_bonus_eos_and_empty_draft(self):
        from minideepinfer.sampling import SampledDraft, verify_sampled
        proposal = SampledDraft((0, 1), ([1., 0.], [0., 1.]))
        ps = ([1., 0.], [0., 1.], [1., 0.])
        result, _ = verify_sampled(proposal, ps, 3, ScriptedRandom(.9, .9, .9))
        self.assertEqual((result.tokens, result.final_role, result.finish_reason), ((0, 1, 0), 'bonus', 'length'))
        result, checks = verify_sampled(proposal, ps, 3, ScriptedRandom(.1, .1), eos_token_id=1)
        self.assertEqual((result.tokens, result.finish_reason, len(checks)), ((0, 1), 'eos', 2))
        result, _ = verify_sampled(SampledDraft((), ()), ([0., 1.],), 1, ScriptedRandom(.5))
        self.assertEqual((result.tokens, result.final_role), ((1,), 'target'))
        # p[x]=0 cannot accept even when the uniform draw is exactly zero.
        result, checks = verify_sampled(SampledDraft((0,), ([1., 0.],)),
                                         ([0., 1.], [1., 0.]), 2, ScriptedRandom(0., 0.))
        self.assertEqual(result.tokens, (1,))
        self.assertFalse(checks[0]['accepted'])

    def test_invalid_proposals_are_not_silently_used_as_greedy(self):
        from minideepinfer.sampling import SampledDraft, verify_sampled
        for proposal, target, remaining in (
                (SampledDraft((0,), ()), ([.5, .5],)*2, 2),
                (SampledDraft((0,), ([0., 1.],)), ([.5, .5],)*2, 2),
                (SampledDraft((2,), ([.5, .5],)), ([.5, .5],)*2, 2),
                (SampledDraft((0,), ([1.],)), ([.5, .5],)*2, 2),
                (SampledDraft((0,), ([.5, .5],)), ([.5, .5],)*2, 1)):
            with self.assertRaises(ValueError):
                verify_sampled(proposal, target, remaining, ScriptedRandom(.5))

    def test_sampled_two_token_joint_distribution_matches_target(self):
        from minideepinfer.sampling import SampledDraft, draw, verify_sampled
        # The second target distribution depends on the first *committed* token.
        p, q = [.25, .75], [.75, .25]
        after = ([.8, .2], [.1, .9])
        counts = [[0, 0], [0, 0]]
        rng = random.Random(2026)
        n = 10000
        for _ in range(n):
            candidate = draw(q, rng)
            decision, _ = verify_sampled(SampledDraft((candidate,), (q,)),
                                          (p, after[candidate]), 2, rng)
            first = decision.tokens[0]
            second = decision.tokens[1] if len(decision.tokens) == 2 else draw(after[first], rng)
            counts[first][second] += 1
        for first in range(2):
            for second in range(2):
                self.assertLess(abs(counts[first][second]/n-p[first]*after[first][second]), .015)


@unittest.skipUnless(AVAILABLE, 'install torch for sampled model decoding')
class SampledModelTests(unittest.TestCase):
    def models(self, hybrid=False):
        import torch
        from minideepinfer.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
        from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
        torch.set_num_threads(1)
        if hybrid:
            config = Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,
                ('linear_attention',)*3+('full_attention',), max_position_embeddings=64,
                partial_rotary_factor=.5, tie_word_embeddings=False)
            cls = Qwen3_5ForCausalLM
        else:
            config = Qwen2Config(97,32,64,2,4,2,max_position_embeddings=64)
            cls = Qwen2ForCausalLM
        torch.manual_seed(2026)
        target = cls(config).eval()
        torch.manual_seed(2036)
        return target, cls(config).eval()

    def test_draft_distributions_are_from_their_actual_conditional_prefixes(self):
        import torch
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.sampling import TemperatureSampler
        from minideepinfer.speculative import CachedTarget
        for hybrid in (False, True):
            _, small = self.models(hybrid)
            sampler = TemperatureSampler(.7, 4)
            draft = ModelDraft(small, 12, sampler=sampler)
            try:
                proposal = draft((2, 3, 4), 3)
                for index, q in enumerate(proposal.probabilities):
                    cold = CachedTarget(small, tuple(range(12)), 12)
                    logits = cold.forward((2, 3, 4)+proposal.tokens[:index])[-1]
                    torch.testing.assert_close(q, sampler.probabilities(logits), atol=1e-7, rtol=1e-6)
            finally:
                draft.close()
            self.assertEqual(draft.stats()['free_slots'], 12)

    def test_both_caches_match_cold_prefix_after_acceptance_and_rejection(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.sampling import TemperatureSampler
        from minideepinfer.speculative import generate
        from minideepinfer.spec_demo import compare_cache
        for hybrid in (False, True):
            target, small = self.models(hybrid)
            corrections = accepts = 0
            for seed in (3, 9, 2026):
                draft = ModelDraft(small, 14, sampler=TemperatureSampler(.15, seed+1), on_cache=compare_cache)
                try:
                    result = generate(target, (2, 3, 4), 12, draft, width=3,
                                      sampler=TemperatureSampler(.15, seed), on_cache=compare_cache)
                finally:
                    draft.close()
                accepts += result['accepted_draft_tokens']
                corrections += result['correction_tokens']
                self.assertEqual(result['free_slots'], 14)
                self.assertEqual(draft.stats()['free_slots'], 14)
                self.assertFalse(draft.stats()['snapshot_retained'])
                self.assertEqual(result['cached_tokens'], 14)
            self.assertGreater(corrections, 0)
            self.assertGreater(accepts, 0)

    def test_width_zero_matches_independent_full_prefix_temperature_sampling(self):
        from minideepinfer.sampling import TemperatureSampler, SampledDraft
        from minideepinfer.speculative import CachedTarget, generate
        for hybrid in (False, True):
            target, _ = self.models(hybrid)
            sampler = TemperatureSampler(.7, 2026)
            prefix, expected = (2, 3, 4), []
            for _ in range(7):
                cold = CachedTarget(target, tuple(range(16)), 16)
                token = sampler.sample(cold.forward(prefix+tuple(expected))[-1])
                expected.append(token)
            result = generate(target, prefix, 7, lambda *_: SampledDraft((), ()), width=0,
                              sampler=TemperatureSampler(.7, 2026))
            self.assertEqual(result['output_ids'], expected)

    def test_cancellation_and_observer_failure_release_both_pools(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.sampling import TemperatureSampler
        from minideepinfer.speculative import generate
        for fail in (False, True):
            target, small = self.models(True)
            draft = ModelDraft(small, 14, sampler=TemperatureSampler(.7, 9))
            committed = []
            def consume(tokens):
                committed.extend(tokens)
                if fail and len(committed) > 1:
                    raise RuntimeError('consumer')
            released = []
            from minideepinfer.kv_cache import SlotPool
            original = SlotPool.release
            def release(pool, rid):
                original(pool, rid)
                released.append(pool.free_count)
            try:
                with patch.object(SlotPool, 'release', release):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'consumer'):
                            generate(target, (2, 3, 4), 12, draft, width=3,
                                     sampler=TemperatureSampler(.7, 8), on_commit=consume)
                    else:
                        result = generate(target, (2, 3, 4), 12, draft, width=3,
                                          sampler=TemperatureSampler(.7, 8), on_commit=consume,
                                          should_cancel=lambda: len(committed) > 1)
                        self.assertEqual(result['finish_reason'], 'cancelled')
            finally:
                draft.close()
            self.assertEqual(released, [14])
            self.assertEqual(draft.stats()['free_slots'], 14)

    def test_text_entry_trace_seed_and_no_greedy_certification(self):
        from minideepinfer.speculative_text import run_speculative_text
        from test_speculative_text import Tokenizer
        target, small = self.models(True)
        committed = []
        options = dict(max_new_tokens=9, width=3, raw=True, temperature=.7, seed=8)
        first = run_speculative_text(target,small,Tokenizer(),'2 3 4',trace_rounds=True,
                                     on_commit=lambda tokens: committed.extend(tokens),**options)
        second = run_speculative_text(target,small,Tokenizer(),'2 3 4',**options)
        self.assertEqual(first['output_ids'], second['output_ids'])
        self.assertEqual(first['output_ids'], committed)
        self.assertEqual(first['greedy_check']['status'], 'not_run')
        self.assertTrue(first['trace']['rounds'])
        self.assertTrue(first['trace']['rounds'][0]['sampling_checks'])
        self.assertEqual(first['draft_work']['free_slots'], first['cache_capacity'])
        with self.assertRaisesRegex(ValueError, 'greedy check'):
            run_speculative_text(target,small,Tokenizer(),'2 3 4',check_greedy=True,**options)

    def test_sampled_eos_pre_cancel_and_draft_contract(self):
        from minideepinfer.sampling import TemperatureSampler, SampledDraft
        from minideepinfer.speculative import generate
        target, _ = self.models()
        cancelled = generate(target,(2,3),4,lambda *_: (),sampler=TemperatureSampler(),should_cancel=lambda: True)
        self.assertEqual((cancelled['output_ids'], cancelled['target_forward_calls']), ([],0))
        first = generate(target,(2,3),1,lambda *_: (),sampler=TemperatureSampler())['output_ids'][0]
        stopped = generate(target,(2,3),4,lambda *_: (),sampler=TemperatureSampler(),eos_token_id=first)
        self.assertEqual(stopped['finish_reason'], 'eos')
        with self.assertRaisesRegex(ValueError, 'SampledDraft'):
            generate(target,(2,3),4,lambda *_: (),sampler=TemperatureSampler())
        with self.assertRaisesRegex(ValueError, 'sampler'):
            generate(target,(2,3),4,lambda *_: SampledDraft((),()))
