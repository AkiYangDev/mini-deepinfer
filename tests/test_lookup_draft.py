import importlib.util
import unittest
from unittest.mock import patch

from minideepinfer.lookup_draft import LookupDraft


class LookupDraftTests(unittest.TestCase):
    def test_longest_match_before_latest_occurrence_and_short_tail(self):
        draft=LookupDraft(3)
        # The later singleton 2 has a continuation, but the earlier 1,2 wins.
        prefix=(1,2,8,9,2,7,1,2)
        self.assertEqual(draft(prefix,3),(8,9,2))
        self.assertEqual((draft.last['match_length'],draft.last['source_start']),(2,0))
        # Within the same match length, use the latest occurrence.
        self.assertEqual(draft((1,2,8,1,2,9,1,2),2),(9,1))
        self.assertEqual(draft.last['source_start'],3)
        self.assertEqual(draft((4,4),3),(4,))

    def test_no_match_and_zero_width_do_not_invent_candidates(self):
        draft=LookupDraft()
        self.assertEqual(draft((1,2,3),3),())
        work=draft.stats()['tokens_compared']
        self.assertEqual(draft((1,1,1),0),())
        self.assertEqual(draft.stats()['tokens_compared'],work)
        self.assertEqual(draft.stats()['model_forward_calls'],0)
        self.assertEqual(draft.stats()['proposed_tokens'],0)

    def test_proposals_are_always_committed_contiguous_tokens(self):
        import random
        random_source=random.Random(2026)
        for _ in range(200):
            prefix=tuple(random_source.randrange(5) for _ in range(random_source.randrange(1,30)))
            limit=random_source.randrange(5)
            draft=LookupDraft(4)
            proposed=draft(prefix,limit)
            self.assertLessEqual(len(proposed),limit)
            if proposed:
                size,start=draft.last['match_length'],draft.last['source_start']
                self.assertEqual(prefix[start:start+size],prefix[-size:])
                self.assertLess(start,len(prefix)-size)
                self.assertEqual(proposed,prefix[start+size:start+size+limit])
                self.assertEqual(draft.stats()['proposed_tokens'],len(proposed))

    def test_invalid_parameters(self):
        for size in (0,-1,True,1.5):
            with self.assertRaises(ValueError):LookupDraft(size)
        for prefix,limit in (((),1),((True,),1),((-1,),1),((1,),True),((1,),-1)):
            with self.assertRaises(ValueError):LookupDraft()(prefix,limit)


AVAILABLE=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(AVAILABLE,'install torch for independent draft checks')
class LookupModelTests(unittest.TestCase):
    def test_lookup_and_abstention_match_greedy_for_both_cache_types(self):
        import torch
        from minideepinfer.engine import Engine
        from minideepinfer.model_runner import ModelRunner
        from minideepinfer.models.qwen2 import Qwen2Config,Qwen2ForCausalLM
        from minideepinfer.models.qwen3_5 import Qwen3_5Config,Qwen3_5ForCausalLM
        from minideepinfer.spec_demo import compare_cache
        from minideepinfer.speculative import generate
        torch.set_num_threads(1);torch.manual_seed(2026)
        models=(Qwen2ForCausalLM(Qwen2Config(97,32,64,2,4,2,max_position_embeddings=64)).eval(),
                Qwen3_5ForCausalLM(Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,
                    ('linear_attention',)*3+('full_attention',),max_position_embeddings=64,partial_rotary_factor=0.5)).eval())
        for model in models:
            for prompt in ((2,3,4),(2,3,2,3,4)):
                for width in (0,1,3):
                    for count in (1,2,9):
                        engine=Engine(ModelRunner(model,32),32)
                        request=engine.submit(prompt,count);list(engine.run())
                        for eos in (None,request.output_ids[-1]):
                            draft=LookupDraft();seen=[]
                            result=generate(model,prompt,count,draft,width=width,eos_token_id=eos,
                                            on_cache=lambda t:seen.append(compare_cache(t)))
                            expected=request.output_ids
                            if eos is not None:expected=expected[:expected.index(eos)+1]
                            self.assertEqual(result['output_ids'],expected)
                            self.assertEqual(result['free_slots'],len(prompt)+count-1)
                            self.assertEqual(draft.stats()['proposed_tokens'],result['proposed_draft_tokens'])
                            self.assertEqual(len(seen),len(result['rounds']))

    def test_demo_runs_greedy_baseline_after_lookup(self):
        from minideepinfer.engine import Engine
        from minideepinfer.spec_demo import run
        from minideepinfer.speculative import generate
        events=[];run_engine=Engine.run
        def speculative(*args,**kwargs):
            events.append('lookup')
            return generate(*args,**kwargs)
        def greedy(engine,*args,**kwargs):
            events.append('baseline')
            return run_engine(engine,*args,**kwargs)
        with patch('minideepinfer.speculative.generate',speculative),patch.object(Engine,'run',greedy):
            report=run(verbose=False,hybrid=True,draft_mode='lookup')
        self.assertEqual(events,['lookup','baseline'])
        self.assertTrue(report['passed'])
        self.assertNotIn('oracle',report)
        self.assertTrue(report['cases']['lookup']['matches_greedy'])
        self.assertEqual(report['draft']['model_forward_calls'],0)
        self.assertEqual(report['draft']['proposed_tokens'],report['cases']['lookup']['proposed_draft_tokens'])

    def test_oversized_draft_still_fails_and_releases_slots(self):
        from minideepinfer.speculative import generate
        from minideepinfer.models.qwen2 import Qwen2Config,Qwen2ForCausalLM
        from minideepinfer.kv_cache import SlotPool
        model=Qwen2ForCausalLM(Qwen2Config(97,32,64,2,4,2,max_position_embeddings=64)).eval()
        release=SlotPool.release;freed=[]
        def observe(pool,rid):release(pool,rid);freed.append(pool.free_count)
        with patch.object(SlotPool,'release',observe):
            with self.assertRaisesRegex(ValueError,'at most'):
                generate(model,(2,3),5,lambda p,n:[4]*(n+1))
        self.assertEqual(freed,[6])
