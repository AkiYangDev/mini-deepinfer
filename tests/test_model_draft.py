import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minideepinfer.model_draft import check_tokenizer_pair


class TokenizerContractTests(unittest.TestCase):
    def test_identical_files_and_special_ids_required(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=[Path(directory)/name for name in ('draft','target')]
            for root in roots:
                root.mkdir()
                (root/'config.json').write_text(json.dumps(dict(text_config=dict(vocab_size=97,eos_token_id=7))))
                (root/'tokenizer.json').write_text('{"model":{}}')
                (root/'tokenizer_config.json').write_text('{}')
            self.assertTrue(check_tokenizer_pair(*roots)['compatible'])
            (roots[1]/'tokenizer.json').write_text('{"model":{"changed":true}}')
            with self.assertRaisesRegex(ValueError,'tokenizer files'):check_tokenizer_pair(*roots)
            (roots[1]/'tokenizer.json').write_text('{"model":{}}')
            (roots[1]/'config.json').write_text(json.dumps(dict(text_config=dict(vocab_size=97,eos_token_id=8))))
            with self.assertRaisesRegex(ValueError,'special token'):check_tokenizer_pair(*roots)

    def test_missing_and_one_sided_optional_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=[Path(directory)/name for name in ('draft','target')]
            for root in roots:
                root.mkdir();(root/'config.json').write_text('{"vocab_size":97}')
                (root/'tokenizer.json').write_text('{}');(root/'tokenizer_config.json').write_text('{}')
            (roots[0]/'chat_template.jinja').write_text('template')
            with self.assertRaisesRegex(ValueError,'chat_template'):check_tokenizer_pair(*roots)
            (roots[0]/'chat_template.jinja').unlink();(roots[1]/'tokenizer.json').unlink()
            with self.assertRaisesRegex(ValueError,'tokenizer.json'):check_tokenizer_pair(*roots)


AVAILABLE=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(AVAILABLE,'install torch for model draft checks')
class ModelDraftTests(unittest.TestCase):
    def models(self,hybrid=False,same=False):
        import copy
        import torch
        from minideepinfer.models.qwen2 import Qwen2Config,Qwen2ForCausalLM
        from minideepinfer.models.qwen3_5 import Qwen3_5Config,Qwen3_5ForCausalLM
        torch.set_num_threads(1);torch.manual_seed(2026)
        if hybrid:
            model=Qwen3_5ForCausalLM(Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,
                ('linear_attention',)*3+('full_attention',),max_position_embeddings=64,partial_rotary_factor=0.5)).eval()
        else:model=Qwen2ForCausalLM(Qwen2Config(97,32,64,2,4,2,max_position_embeddings=64)).eval()
        if same:return model,copy.deepcopy(model)
        torch.manual_seed(2036)
        return model,type(model)(model.config).eval()

    def test_both_models_align_with_cold_state_and_greedy(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.speculative import generate
        from minideepinfer.spec_demo import compare_cache
        from minideepinfer.model_runner import ModelRunner
        from minideepinfer.engine import Engine
        for hybrid in (False,True):
            for same in (False,True):
                target,small=self.models(hybrid,same)
                for width in (0,1,3):
                    for count in (1,2,9):
                        engine=Engine(ModelRunner(target,32),32)
                        request=engine.submit((2,3,4),count);list(engine.run())
                        checks=[]
                        draft=ModelDraft(small,3+count-1,on_cache=lambda t:checks.append(compare_cache(t)))
                        try:result=generate(target,(2,3,4),count,draft,width=width,on_cache=compare_cache)
                        finally:draft.close()
                        self.assertEqual(result['output_ids'],request.output_ids)
                        stats=draft.stats()
                        self.assertEqual(stats['free_slots'],3+count-1)
                        self.assertEqual(result['proposed_draft_tokens'],stats['proposed_tokens'])
                        self.assertEqual(stats['forward_calls'],stats['sync_forward_calls']+stats['proposal_forward_calls']+stats['replay_forward_calls'])
                        self.assertEqual(stats['input_tokens'],stats['sync_input_tokens']+stats['proposal_input_tokens']+stats['replayed_input_tokens'])
                        if same:self.assertEqual(result['accepted_draft_tokens'],result['proposed_draft_tokens'])
                        if count==1 or width==0:self.assertIsNone(draft.cache)

    def test_explicit_rejection_restores_draft_branch(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.spec_demo import compare_cache
        for hybrid in (False,True):
            _,model=self.models(hybrid)
            draft=ModelDraft(model,16,on_cache=compare_cache)
            candidates=draft((2,3,4),3)
            committed=(2,3,4,(candidates[0]+1)%97)
            draft(committed,1)
            self.assertEqual(tuple(draft.cache.tokens),committed)
            self.assertEqual(draft.stats()['recovered_branches'],1)
            compare_cache(draft.cache)
            draft.close();draft.close()
            self.assertEqual(draft.stats()['free_slots'],16)
            self.assertEqual(draft.stats()['linear_states_remaining'],0)
            self.assertFalse(draft.stats()['snapshot_retained'])
            with self.assertRaisesRegex(ValueError,'closed'):draft(committed+(2,),1)
            short=ModelDraft(model,16,on_cache=compare_cache)
            values=short((2,3,4),3)
            prefix=(2,3,4)+values[:1]
            short(prefix,1)
            self.assertEqual(tuple(short.cache.tokens),prefix)
            compare_cache(short.cache);short.close()

    def test_eos_and_commit_callback_cleanup(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.speculative import generate
        from minideepinfer.model_runner import ModelRunner
        from minideepinfer.engine import Engine
        from minideepinfer.kv_cache import SlotPool
        target,small=self.models(False,True)
        engine=Engine(ModelRunner(target,32),32);request=engine.submit((2,3,4),9);list(engine.run())
        eos=request.output_ids[1]
        draft=ModelDraft(small,11);committed=[]
        try:result=generate(target,(2,3,4),9,draft,eos_token_id=eos,on_commit=lambda group:committed.extend(group))
        finally:draft.close()
        self.assertEqual(result['output_ids'],committed)
        self.assertEqual(len(committed),2)  # EOS was accepted after prefill, not at the first output.
        self.assertEqual(result['finish_reason'],'eos')
        release=SlotPool.release;freed=[]
        def observe(pool,rid):release(pool,rid);freed.append((rid,pool.free_count))
        draft=ModelDraft(small,11)
        with patch.object(SlotPool,'release',observe):
            try:
                with self.assertRaisesRegex(RuntimeError,'commit failed'):
                    generate(target,(2,3,4),9,draft,on_commit=lambda group:(_ for _ in ()).throw(RuntimeError('commit failed')))
            finally:draft.close()
        self.assertIn(('spec',11),freed);self.assertIn(('draft',11),freed)

    def test_failed_draft_drain_keeps_slots_owned(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.speculative import CachedTarget
        _,model=self.models(True);draft=ModelDraft(model,16);draft((2,3,4),1)
        with patch.object(CachedTarget,'synchronize',side_effect=RuntimeError('drain failed')):
            with self.assertRaisesRegex(RuntimeError,'drain failed'):draft.close()
        self.assertEqual(draft.pool.free_count,0);self.assertFalse(draft.closed)
        draft.close();self.assertEqual(draft.pool.free_count,16)

    def test_invalid_prefix_and_capacity(self):
        from minideepinfer.model_draft import ModelDraft
        _,model=self.models()
        for capacity in (0,True,65):
            with self.assertRaises(ValueError):ModelDraft(model,capacity)
        draft=ModelDraft(model,16);draft((2,3,4),2)
        for prefix,width in (((2,3,4),2),((9,3,4,5),2),((2,3,4,True),2),((2,3,4,5),9)):
            with self.assertRaises(ValueError):draft(prefix,width)
        draft.close()
