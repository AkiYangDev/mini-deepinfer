"""Check FP32 model sampling against cold prefixes; no timing or weight download."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def check_pair(target, small, prompt, report, *, count=6, temperature=.7, seed=2026, top_k=0, top_p=1.0):
    """Reuse weights; each case owns fresh caches, probabilities and random state."""
    import torch
    from .model_draft import ModelDraft
    from .sampling import SampledDraft, TemperatureSampler, verify_sampled
    from .spec_demo import compare_cache
    from .speculative import CachedTarget, generate

    prompt = tuple(prompt)
    capacity = len(prompt)+count-1
    reference_sampler = TemperatureSampler(temperature, seed, top_k=top_k, top_p=top_p)
    report.update(prompt_ids=list(prompt),max_new_tokens=count,capacity=capacity,
                  temperature=temperature,target_seed=seed,draft_seed=seed+1,
                  probability_checks=[],cache_checks=[],cold_work=[],cases={})
    if top_k or top_p != 1:
        report.update(top_k=top_k,top_p=top_p,filter_order='temperature,top_k,top_p')

    def cold(model, prefix, label):
        cache = CachedTarget(model, tuple(range(capacity)), capacity)
        try:
            logits = cache.forward(prefix)[-1]
            return reference_sampler.probabilities(logits)
        finally:
            report['cold_work'].append(dict(kind=label,forward_calls=cache.forward_calls,
                                            input_tokens=cache.input_tokens))
            cache.synchronize()
            if cache.hybrid:
                cache.cache.release('spec')

    def probabilities(actual, expected, label):
        row = dict(kind=label,max_abs=float((actual-expected).abs().max()),
                   total_variation=float((actual-expected).abs().sum()/2),passed=True,
                   atol=1e-6,rtol=2e-4)
        report['probability_checks'].append(row)
        try:
            torch.testing.assert_close(actual,expected,atol=row['atol'],rtol=row['rtol'])
        except AssertionError:
            row['passed'] = False
            raise

    def state(cache, label):
        row = compare_cache(cache,atol=1e-4,rtol=2e-4,strict=False)
        row['kind'] = label
        report['cache_checks'].append(row)
        report['cold_work'].append(dict(kind=label,forward_calls=1,input_tokens=len(cache.tokens)))
        if not row['passed']:
            raise RuntimeError('cache differs from independent cold prefix: '+label)

    class ObservedSampler(TemperatureSampler):
        def __init__(self):
            super().__init__(temperature, seed, top_k=top_k, top_p=top_p)
            self.rows = []

        def probabilities(self, logits):
            p = super().probabilities(logits)
            self.rows.append(p)
            return p

    # Independent target-only sampling recomputes the entire prefix each time.
    expected, expected_rows = [], []
    for _ in range(count):
        p = cold(target,prompt+tuple(expected),'ordinary_full_prefix')
        expected_rows.append(p)
        from .sampling import draw
        expected.append(draw(p,reference_sampler.rng))
    sampler = ObservedSampler()
    ordinary = generate(target,prompt,count,lambda *_: SampledDraft((),()),width=0,sampler=sampler)
    report['cases']['ordinary'] = ordinary
    report['ordinary_full_prefix_ids'] = expected
    if ordinary['output_ids'] != expected:
        raise RuntimeError('ordinary sampled outputs differ from independent full-prefix sampling')
    for actual, wanted in zip(sampler.rows,expected_rows):
        probabilities(actual,wanted,'ordinary_probability')

    for name in ('sampled','cancelled','configured_eos'):
        sampler = ObservedSampler()
        draft = ModelDraft(small,capacity,sampler=TemperatureSampler(temperature,seed+1,top_k=top_k,top_p=top_p),
                           on_cache=lambda cache: state(cache,'draft_aligned'))
        committed, pending = [], {}

        def propose(prefix, limit):
            proposal = draft(prefix,limit)
            qs = []
            for index,q in enumerate(proposal.probabilities):
                wanted = cold(small,tuple(prefix)+proposal.tokens[:index],'draft_probability_reference')
                probabilities(q,wanted,'draft_probability')
                qs.append(wanted)
            pending.update(proposal=proposal,qs=tuple(qs),prefix=tuple(prefix),
                           row_start=len(sampler.rows),rng=sampler.rng.getstate())
            return proposal

        eos = expected[0] if name=='configured_eos' else None

        def observed(row):
            proposal, prefix = pending['proposal'], pending['prefix']
            actual_rows = sampler.rows[pending['row_start']:]
            if len(actual_rows) != len(proposal.tokens)+1:
                raise RuntimeError('target probability rows do not match verification inputs')
            ps = []
            for index,p in enumerate(actual_rows):
                wanted = cold(target,prefix+proposal.tokens[:index],'target_probability_reference')
                probabilities(p,wanted,'target_probability')
                ps.append(wanted)
            rng = random.Random()
            rng.setstate(pending['rng'])
            wanted, _ = verify_sampled(SampledDraft(proposal.tokens,pending['qs']),ps,
                                      count-(len(prefix)-len(prompt)),rng,eos)
            if (list(wanted.tokens) != row['committed']
                    or wanted.accepted_draft != row['accepted_draft']
                    or wanted.final_role != row['final_role']):
                raise RuntimeError('sampled decision differs from cold probabilities with identical random draws')
            row['matches_cold_decision'] = True

        try:
            result = generate(target,prompt,count,propose,width=3,sampler=sampler,eos_token_id=eos,
                              on_commit=lambda tokens: committed.extend(tokens),on_round=observed,
                              on_cache=lambda cache: state(cache,'target_retained'),
                              should_cancel=lambda: name=='cancelled' and len(committed)>=2)
        finally:
            draft.close()
        result['draft_work'] = draft.stats()
        result['configured_eos_token_id'] = eos
        report['cases'][name] = result
        probabilities(sampler.rows[0],expected_rows[0],name+'_prefill_probability')
        if result['output_ids'] != committed:
            raise RuntimeError('commit callback differs from delivered output')
        if (result['free_slots'] != capacity or draft.stats()['free_slots'] != capacity
                or draft.stats()['snapshot_retained'] or draft.stats()['linear_states_remaining']):
            raise RuntimeError('sampling case leaked request state')
        wanted_reason = dict(sampled='length',cancelled='cancelled',configured_eos='eos')[name]
        if result['finish_reason'] != wanted_reason:
            raise RuntimeError('unexpected stop reason in '+name)
    if ordinary['free_slots'] != capacity:
        raise RuntimeError('ordinary sampling leaked slots')
    report['total_forward_calls'] = sum(case['target_forward_calls']+case.get('draft_work',{}).get('forward_calls',0)
                                        for case in report['cases'].values())+sum(row['forward_calls'] for row in report['cold_work'])
    report['total_input_tokens'] = sum(case['target_input_tokens']+case.get('draft_work',{}).get('input_tokens',0)
                                       for case in report['cases'].values())+sum(row['input_tokens'] for row in report['cold_work'])
    report['passed'] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--draft-model',required=True)
    parser.add_argument('--device',default='npu:0')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--top-k',type=int,default=0)
    parser.add_argument('--top-p',type=float,default=1.0)
    args = parser.parse_args(argv)
    from .sampling import TemperatureSampler
    TemperatureSampler(top_k=args.top_k,top_p=args.top_p)  # Before loading weights.
    from .model_draft import check_tokenizer_pair
    contract = check_tokenizer_pair(args.draft_model,args.model)
    import torch
    from transformers import AutoTokenizer
    from .model_runner import ModelRunner
    import platform
    torch.set_num_threads(1)
    report = dict(passed=False,torch=torch.__version__,python=platform.python_version(),
                  device=args.device,dtype='float32',cpu_threads=1,tokenizer_contract=contract,
                  target_model=Path(args.model).name,draft_model=Path(args.draft_model).name,
                  weight_loads=0,source_sha256={},
                  scope='Fixed short-input FP32 sampling and cold-prefix checks; probabilities on CPU FP64; no timing or speedup claim. Configured EOS is the first sampled ID, not a natural language EOS test.')
    sources = ('verify_sampling.py','sampling.py','model_draft.py','speculative.py','spec_demo.py',
               'models/qwen2.py','models/qwen3_5.py','model_runner.py','tensor_cache.py','hybrid_cache.py','engine.py','kv_cache.py')
    report['source_sha256'] = {'minideepinfer/'+name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in sources}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary = args.output.with_name(args.output.name+'.tmp')
        temporary.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        temporary.replace(args.output)
    phase = 'loading'
    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        target = ModelRunner.from_pretrained(args.model,args.device,torch.float32,1).model
        report['weight_loads'] += 1
        small = ModelRunner.from_pretrained(args.draft_model,args.device,torch.float32,1).model
        report['weight_loads'] += 1
        if args.device.startswith('npu'):
            import torch_npu
            report.update(torch_npu=torch_npu.__version__,npu_name=torch.npu.get_device_name(0),
                          hf32=dict(matmul=torch.npu.matmul.allow_hf32,conv=torch.npu.conv.allow_hf32))
        phase = 'numerical_checks'
        check_pair(target,small,tokenizer.encode('The capital of France is'),report,
                   top_k=args.top_k,top_p=args.top_p)
    except BaseException as error:
        report['failure'] = dict(phase=phase,type=type(error).__name__,message=str(error))
        save()
        raise
    save()
    print(json.dumps(dict(passed=report['passed'],cases={k:v['finish_reason'] for k,v in report['cases'].items()},
                          probability_checks=len(report['probability_checks']),cache_checks=len(report['cache_checks']),
                          forward_calls=report['total_forward_calls'])),flush=True)
    return report


if __name__ == '__main__':
    main()
