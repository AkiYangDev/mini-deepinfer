import contextlib
import importlib.util
import io
import math
import random
import unittest


AVAILABLE = importlib.util.find_spec('torch') is not None


class FilterArgumentTests(unittest.TestCase):
    def test_invalid_filters_fail_before_weight_loading(self):
        from minideepinfer.sampling import TemperatureSampler
        for value in (-1, True, 1.5, None):
            with self.assertRaises(ValueError):
                TemperatureSampler(top_k=value)
        for value in (0, -1, 1.01, True, float('nan'), float('inf'), None):
            with self.assertRaises(ValueError):
                TemperatureSampler(top_p=value)
        from minideepinfer.generate import parse_args
        base = ['--model','target','--speculative','--draft-model','draft']
        for extra in (['--top-k','2'], ['--top-p','.9'],
                      ['--temperature','.7','--top-k','-1'],
                      ['--temperature','.7','--top-p','nan'],
                      ['--temperature','.7','--top-p','0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base+extra)
        args = parse_args(base+['--temperature','.7','--top-k','20','--top-p','.9'])
        self.assertEqual((args.top_k,args.top_p),(20,.9))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--model','target','--top-k','2'])


@unittest.skipUnless(AVAILABLE, 'install torch for sampling filters')
class FilterProbabilityTests(unittest.TestCase):
    def test_filter_order_and_renormalization(self):
        import torch
        from minideepinfer.sampling import TemperatureSampler
        logits = torch.tensor([.45,.30,.15,.10],dtype=torch.float64).log()
        # top-k leaves [.5, 1/3, 1/6]; nucleus .7 keeps the first two.
        p = TemperatureSampler(top_k=3,top_p=.7).probabilities(logits)
        torch.testing.assert_close(p,torch.tensor([.6,.4,0,0],dtype=torch.float64))
        # Temperature changes the nucleus: sqrt(.6)/sum(sqrt(.6),sqrt(.4)) < .6.
        logits = torch.tensor([.6,.4],dtype=torch.float64).log()
        self.assertEqual(int((TemperatureSampler(1,top_p=.6).probabilities(logits)>0).sum()),1)
        self.assertEqual(int((TemperatureSampler(2,top_p=.6).probabilities(logits)>0).sum()),2)

    def test_ties_boundaries_tiny_p_and_default_compatibility(self):
        import torch
        from minideepinfer.sampling import TemperatureSampler
        logits = torch.zeros(4,dtype=torch.float64)
        p = TemperatureSampler(top_k=2).probabilities(logits)
        self.assertEqual(p.tolist(),[.5,.5,0,0])
        self.assertEqual(TemperatureSampler(top_p=.5).probabilities(logits).tolist(),[.5,.5,0,0])
        self.assertEqual(TemperatureSampler(top_p=1e-300).probabilities(logits).tolist(),[1,0,0,0])
        self.assertEqual(TemperatureSampler(top_k=1).probabilities(logits).tolist(),[1,0,0,0])
        logits = torch.tensor([3.,-1.,2.,0.])
        default = TemperatureSampler(.7).probabilities(logits)
        self.assertTrue(torch.equal(default,TemperatureSampler(.7,top_k=0,top_p=1).probabilities(logits)))
        torch.testing.assert_close(default,TemperatureSampler(.7,top_k=100).probabilities(logits))
        self.assertEqual(TemperatureSampler(1e-300,top_k=2,top_p=.9).probabilities(logits).tolist(),[1,0,0,0])
        with self.assertRaises(ValueError):
            TemperatureSampler(top_k=1).probabilities(torch.tensor([]))

    def test_matches_independent_scalar_filter_and_preserves_logits(self):
        import torch
        from minideepinfer.sampling import TemperatureSampler
        rng = random.Random(41)
        for _ in range(20):
            values = [rng.uniform(-4,4) for _ in range(9)]
            temperature,k,p = rng.choice((.3,.7,2.)),rng.choice((0,1,4,20)),rng.choice((.1,.7,.95,1.))
            order = sorted(range(9),key=lambda i:(-values[i],i))[:k or 9]
            weights = [math.exp((values[i]-max(values))/temperature) for i in order]
            total = sum(weights)
            weights = [w/total for w in weights]
            kept, cumulative = [],0.
            for i,w in zip(order,weights):
                kept.append((i,w))
                cumulative += w
                if cumulative >= p:
                    break
            wanted = [0.]*9
            for i,w in kept:
                wanted[i] = w/sum(w for _,w in kept)
            logits = torch.tensor(values,dtype=torch.float64)
            original = logits.clone()
            actual = TemperatureSampler(temperature,top_k=k,top_p=p).probabilities(logits)
            torch.testing.assert_close(actual,torch.tensor(wanted,dtype=torch.float64),atol=1e-14,rtol=1e-14)
            self.assertTrue(torch.equal(logits,original))

    def test_zero_target_support_rejects_and_residual_has_target_distribution(self):
        import torch
        from minideepinfer.sampling import SampledDraft, TemperatureSampler, verify_sampled
        from test_sampling import ScriptedRandom
        sampler = TemperatureSampler(top_k=2,top_p=.9)
        p = sampler.probabilities(torch.tensor([.6,.3,.1]).double().log())
        q = sampler.probabilities(torch.tensor([.1,.3,.6]).double().log())
        result, checks = verify_sampled(SampledDraft((2,),(q,)),(p,p),2,ScriptedRandom(0.,.5))
        self.assertEqual(result.tokens,(0,))
        self.assertFalse(checks[0]['accepted'])
        # Exact one-step marginal, enumerating all proposal IDs, including zeros.
        residual = (p-q).clamp_min(0)
        mass = float(residual.sum())
        marginal = torch.minimum(p,q)+(1-torch.minimum(p,q).sum())*residual/mass
        torch.testing.assert_close(marginal,p,atol=1e-14,rtol=1e-14)


@unittest.skipUnless(AVAILABLE, 'install torch for filtered model generation')
class FilterModelTests(unittest.TestCase):
    def models(self, hybrid):
        from test_sampling import SampledModelTests
        return SampledModelTests().models(hybrid)

    def test_filtered_cold_probabilities_state_recovery_and_stops(self):
        from minideepinfer.verify_sampling import check_pair
        for hybrid in (False,True):
            target,small = self.models(hybrid)
            report = check_pair(target,small,(2,3,4),{},temperature=.7,top_k=7,top_p=.8)
            self.assertTrue(report['passed'])
            self.assertEqual((report['top_k'],report['top_p']),(7,.8))
            self.assertTrue(all(row['passed'] for row in report['probability_checks']+report['cache_checks']))
            sampled = report['cases']['sampled']
            self.assertGreater(sampled['correction_tokens'],0)
            self.assertEqual(sampled['sampling']['filter_order'],'temperature,top_k,top_p')
            self.assertTrue(all(row['matches_cold_decision'] for row in sampled['rounds']))

    def test_text_entry_uses_filters_on_both_models_and_preserves_callbacks(self):
        from unittest.mock import patch
        from minideepinfer.sampling import TemperatureSampler
        from minideepinfer.speculative_text import run_speculative_text
        from test_speculative_text import Tokenizer
        target,small = self.models(True)
        received = []
        observed = set()
        original = TemperatureSampler.probabilities
        def probabilities(sampler,logits):
            observed.add((sampler.seed,sampler.top_k,sampler.top_p))
            return original(sampler,logits)
        with patch.object(TemperatureSampler,'probabilities',probabilities):
            result = run_speculative_text(target,small,Tokenizer(),'2 3 4',raw=True,
                 max_new_tokens=9,width=3,temperature=.7,top_k=7,top_p=.8,seed=8,
                 trace_rounds=True,on_commit=lambda tokens:received.extend(tokens),
                 should_cancel=lambda:len(received)>=2)
        self.assertEqual(observed,{(8,7,.8),(9,7,.8)})
        self.assertEqual(result['output_ids'],received)
        self.assertEqual(result['finish_reason'],'cancelled')
        self.assertEqual(result['free_slots'],result['cache_capacity'])
        self.assertEqual(result['draft_work']['free_slots'],result['cache_capacity'])
        self.assertFalse(result['draft_work']['linear_states_remaining'])
        with self.assertRaisesRegex(ValueError,'temperature'):
            run_speculative_text(target,small,Tokenizer(),'2 3',raw=True,top_k=2)
