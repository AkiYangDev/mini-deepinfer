import importlib.util
import unittest
from unittest.mock import patch

from minideepinfer.speculative import CachedTarget, generate, verify_greedy


class GreedyDecisionTests(unittest.TestCase):
    def test_rejection_bonus_and_stop_boundaries(self):
        cases = [
            ((1,2,3),(9,2,3,4),None,(9,),0,'correction',None),
            ((1,2,3),(1,9,3,4),None,(1,9),1,'correction',None),
            ((1,2,3),(1,2,9,4),None,(1,2,9),2,'correction',None),
            ((1,2,3),(1,2,3,4),None,(1,2,3,4),3,'bonus','length'),
            ((1,2,3),(1,2,3,4),2,(1,2),2,None,'eos'),
            ((1,2,3),(1,9,3,4),9,(1,9),1,'correction','eos'),
            ((1,2,3),(1,2,3,4),4,(1,2,3,4),3,'bonus','eos'),
        ]
        for draft,target,eos,tokens,accepted,role,reason in cases:
            with self.subTest(draft=draft,target=target,eos=eos):
                result=verify_greedy(draft,target,4,eos)
                self.assertEqual((result.tokens,result.accepted_draft,result.final_role,result.finish_reason),
                                 (tokens,accepted,role,reason))
        result=verify_greedy((),(7,),1)
        self.assertEqual((result.tokens,result.final_role,result.finish_reason),((7,),'target','length'))

    def test_invalid_verification_shapes_and_ids_are_rejected(self):
        for draft,target,remaining,eos in [((1,),(1,),2,None),((1,),(1,2),1,None),
                ((),(1,),0,None),((),(1,),True,None),((True,),(1,2),2,None),
                ((),(-1,),1,None),((),(1,),1,-1)]:
            with self.assertRaises(ValueError):
                verify_greedy(draft,target,remaining,eos)


AVAILABLE=importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer import Engine, Step
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen2 import Qwen2Config,Qwen2ForCausalLM


@unittest.skipUnless(AVAILABLE,'install torch for cached greedy verification')
class SpeculativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self,seed=2026):
        torch.manual_seed(seed)
        return Qwen2ForCausalLM(Qwen2Config(97,32,64,2,4,2,max_position_embeddings=64)).eval()

    def baseline(self,model,prompt,count,eos=None):
        engine=Engine(ModelRunner(model,64),64)
        request=engine.submit(prompt,count,eos)
        list(engine.run())
        return request.output_ids

    def test_controlled_drafts_match_independent_greedy_across_widths_lengths_and_seeds(self):
        for seed in (2026,2036):
            model=self.model(seed)
            for count in (1,2,5,9):
                prompt=(2,3,4)
                expected=self.baseline(model,prompt,count)
                for width in (0,1,3,4):
                    for reject in (0,1,None):
                        def draft(prefix,n):
                            start=len(prefix)-len(prompt)
                            values=expected[start:start+n].copy()
                            if reject is not None and reject<len(values):
                                values[reject]=(values[reject]+1)%97
                            return values
                        result=generate(model,prompt,count,draft,width=width)
                        self.assertEqual(result['output_ids'],expected)
                        self.assertEqual(result['cached_tokens'],len(prompt)+count-1)
                        self.assertEqual(result['free_slots'],len(prompt)+count-1)
                        self.assertEqual(len(expected),sum(result[k] for k in
                            ('accepted_draft_tokens','correction_tokens','bonus_tokens','direct_target_tokens')))
                        for row in result['rounds']:
                            self.assertGreaterEqual(row['tentative_cached'],row['retained_cached'])
                            self.assertEqual(len(row['invalidated_slots']),row['tentative_cached']-row['retained_cached'])

    def test_packed_logits_and_poisoned_rejected_slots_match_cold_full_prefix(self):
        model=self.model()
        slots=(11,2,9,5,0,15,3,8,4,12,1,14)
        target=CachedTarget(model,slots,16)
        target.cache.keys.fill_(float('nan')); target.cache.values.fill_(float('nan'))
        def check(block):
            full=tuple(target.tokens)+tuple(block)
            actual=target.forward(block)
            runner=ModelRunner(model,64)
            step=Step('cold','prefill',full,tuple(range(len(full))),tuple(range(len(full))),tuple(range(len(full))))
            with torch.inference_mode():
                expected=runner.forward(step)[-len(block):]
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
        check((2,3,4))
        check((5,6,7,8))
        self.assertEqual(target.rollback(5),(15,3))
        for slot in slots[5:]:
            target.cache.keys[:,slot].fill_(float('nan'))
            target.cache.values[:,slot].fill_(float('nan'))
        check((9,))  # Rewrites one slot; the next poisoned slot must remain unread.
        check((10,))
        with self.assertRaises(ValueError):target.rollback(8)

    def test_eos_in_prefill_accepted_draft_and_final_target(self):
        model=self.model()
        prompt=(2,3,4)
        golden=self.baseline(model,prompt,12)
        eos=next(t for i,t in enumerate(golden) if i>0 and t not in golden[:i])
        expected=self.baseline(model,prompt,12,eos)
        for reject in (None,0):
            def draft(prefix,n):
                values=golden[len(prefix)-len(prompt):len(prefix)-len(prompt)+n].copy()
                if reject==0 and values:values[0]=(values[0]+1)%97
                return values
            result=generate(model,prompt,12,draft,eos_token_id=eos)
            self.assertEqual(result['output_ids'],expected)
            self.assertEqual(result['finish_reason'],'eos')
            self.assertEqual(result['cached_tokens'],len(prompt)+len(expected)-1)
        result=generate(model,prompt,12,lambda *a: self.fail('draft after prefill EOS'),eos_token_id=golden[0])
        self.assertEqual(result['output_ids'],golden[:1])

    def test_bad_draft_observer_error_and_hybrid_model_rejection(self):
        from minideepinfer.kv_cache import SlotPool
        from minideepinfer.speculative import CachedTarget
        model=self.model()
        released=[]
        original=SlotPool.release
        def release(pool,rid):
            original(pool,rid); released.append(pool.free_count)
        with patch.object(SlotPool,'release',release):
            with self.assertRaisesRegex(ValueError,'draft'):
                generate(model,(2,3),5,lambda p,n: [97]*n)
            with self.assertRaisesRegex(ValueError,'observer'):
                generate(model,(2,3),5,lambda p,n: [1]*n,
                         on_round=lambda r: (_ for _ in ()).throw(ValueError('observer')))
        self.assertEqual(released,[6,6])
        with self.assertRaises(ValueError):generate(model,(2,3),5,lambda p,n: [],cache_capacity=3)
        from minideepinfer.models.qwen3_5 import Qwen3_5Config,Qwen3_5ForCausalLM
        config=Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,('linear_attention',)*3+('full_attention',),
                            max_position_embeddings=64,partial_rotary_factor=0.5)
        with self.assertRaisesRegex(ValueError,'full-attention'):
            CachedTarget(Qwen3_5ForCausalLM(config),(0,1),2)

    def test_demo_reports_oracle_cost_and_real_target_work(self):
        from minideepinfer.spec_demo import run
        report=run(verbose=False)
        self.assertTrue(report['passed'])
        self.assertEqual(report['oracle']['forward_calls'],10)
        self.assertEqual(report['oracle']['input_tokens'],12)
        self.assertGreater(report['cases']['first_reject']['target_input_tokens'],12)
        self.assertEqual(report['cases']['all_accept']['target_input_tokens'],12)
        self.assertTrue(all(c['matches_greedy'] and c['free_slots']==12 for c in report['cases'].values()))

    def test_invalid_limits_fail_before_cache_or_forward(self):
        model=self.model()
        with patch('minideepinfer.speculative.CachedTarget') as target:
            for count,width in ((0,3),(True,3),(4,-1),(4,9),(4,True)):
                with self.assertRaises(ValueError):
                    generate(model,(2,3),count,lambda p,n: [1]*n,width=width)
            target.assert_not_called()
