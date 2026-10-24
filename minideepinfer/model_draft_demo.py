"""Single-request greedy draft/target reference; real checkpoints are local only."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns


def commit_metrics(stamps, started, ended):
    """CPU-observed commits; tokens in one accepted group share a timestamp."""
    first,last = stamps[0],stamps[-1]
    whole = (ended-started)/1e6
    gaps = [(b-a)/1e6 for a,b in zip(stamps,stamps[1:])]
    return dict(ttft_ms=(first-started)/1e6,e2e_ms=(last-started)/1e6,
                tpot_ms=(last-first)/1e6/(len(stamps)-1) if len(stamps)>1 else None,
                itl_ms=gaps,whole_ms=whole,output_tokens=len(stamps),
                output_tokens_per_second=len(stamps)*1000/whole)


def run(output=None, *, target_directory=None,draft_directory=None,device='cpu',width=1,repeats=0):
    if bool(target_directory) != bool(draft_directory):
        raise ValueError('provide both local checkpoint directories, or neither')
    if type(width) is not int or not 1 <= width <= 3 or type(repeats) is not int or not 0 <= repeats <= 3:
        raise ValueError('width must be 1..3 and repeats must be 0..3')
    import torch
    from .engine import Engine
    from .model_runner import ModelRunner,select_device
    from .model_draft import ModelDraft,check_tokenizer_pair
    from .models.qwen3_5 import Qwen3_5Config,Qwen3_5ForCausalLM
    from .spec_demo import compare_cache
    from .speculative import generate

    torch.set_num_threads(1)
    report=dict(passed=False,status='running',failure_count=0,device=device,dtype='float32',torch=torch.__version__,python=platform.python_version(),cpu_threads=1,
                prompt_ids=[2,3,4],output_length=10,draft_width=width,repeats=repeats,
                checks=[],checked_runs=[],warmups=[],trials=[],model_work={},source_sha256={},
                scope='one request; eager greedy; separate draft cache; no MTP or probability sampling',
                timing_scope='offline CPU commits; cache initialization and final cleanup included in whole; accepted groups share a commit timestamp; cold references excluded from timed trials')
    def save():
        if output is not None:
            path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.tmp')
            temporary.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode())
            temporary.replace(path)
    sources=('model_draft.py','model_draft_demo.py','speculative.py','spec_demo.py','models/qwen2.py',
             'models/qwen3_5.py','model_runner.py','tensor_cache.py','hybrid_cache.py','engine.py','kv_cache.py')
    for name in sources:
        report['source_sha256']['minideepinfer/'+name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    phase='setup'
    tokenizer=None
    save()
    try:
        if target_directory:
            report['tokenizer_contract']=check_tokenizer_pair(draft_directory,target_directory)
            save()  # Reject a bad pair before loading either set of weights.
            from transformers import AutoTokenizer
            started=perf_counter_ns()
            tokenizer=AutoTokenizer.from_pretrained(target_directory,local_files_only=True)
            report['prompt_ids']=tokenizer.encode('The capital of France is')
            target=ModelRunner.from_pretrained(target_directory,device,torch.float32,128).model
            small=ModelRunner.from_pretrained(draft_directory,device,torch.float32,128).model
            if device.startswith('npu'):
                torch.npu.synchronize()
            report['load_and_tokenize_ms']=(perf_counter_ns()-started)/1e6
            report['target_model']=Path(target_directory).name
            report['draft_model']=Path(draft_directory).name
            report['weight_loads']=2
        else:
            torch.manual_seed(2026)
            target=Qwen3_5ForCausalLM(Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,
                ('linear_attention',)*3+('full_attention',),max_position_embeddings=64,partial_rotary_factor=0.5)).to(select_device(device)).eval()
            torch.manual_seed(2036)
            small=Qwen3_5ForCausalLM(Qwen3_5Config(97,16,24,2,2,1,8,1,2,4,4,
                ('linear_attention','full_attention'),max_position_embeddings=64,partial_rotary_factor=0.5)).to(select_device(device)).eval()
            report.update(target_model='tiny-random-Qwen3.5-target',draft_model='tiny-random-Qwen3.5-draft',weight_loads=0)
        if target.config.vocab_size != small.config.vocab_size:
            raise ValueError('loaded model vocabularies differ')
        if device.startswith('npu'):
            import torch_npu
            report['torch_npu']=torch_npu.__version__
            report['hf32']=dict(matmul=torch.npu.matmul.allow_hf32,conv=torch.npu.conv.allow_hf32)
        for role,model in (('target',target),('draft',small)):
            original=model.forward
            def capture(*args,_original=original,_role=role,**kwargs):
                row=report['model_work'].setdefault(phase,{}).setdefault(_role,dict(forward_calls=0,input_tokens=0))
                row['forward_calls']+=1;row['input_tokens']+=len(args[0].input_ids)
                return _original(*args,**kwargs)
            model.forward=capture
        prompt=tuple(report['prompt_ids']);count=report['output_length'];capacity=len(prompt)+count-1
        def check_cache(cache,role,active_width):
            nonlocal phase
            previous=phase;phase=f'cold_{role}_width{active_width}'
            try:
                values=compare_cache(cache,atol=1e-4 if device.startswith('npu') else 1e-5,
                                     rtol=2e-4 if device.startswith('npu') else 1e-4,strict=False)
            finally:phase=previous
            report['checks'].append(dict(role=role,width=active_width,**values))
            save()
            if not values['passed']:
                raise RuntimeError(role+' cache differs from cold full-prefix reference')
        def baseline():
            stamps=[]
            started=perf_counter_ns()
            runner=ModelRunner(target,capacity);engine=Engine(runner,capacity)
            request=engine.submit(prompt,count)
            for group in engine.run():stamps.append(perf_counter_ns())
            runner.synchronize()
            ended=perf_counter_ns()
            return dict(path='baseline',output_ids=request.output_ids,free_slots=engine.cache.free_count,
                        forward_calls=count,input_tokens=len(prompt)+count-1,
                        metrics=commit_metrics(stamps,started,ended))
        def dual(active_width,checking=False):
            stamps=[]
            started=perf_counter_ns()
            draft=ModelDraft(small,capacity,on_cache=(lambda c:check_cache(c,'draft',active_width)) if checking else None)
            try:
                result=generate(target,prompt,count,draft,width=active_width,
                                on_cache=(lambda c:check_cache(c,'target',active_width)) if checking else None,
                                on_commit=lambda tokens:stamps.extend([perf_counter_ns()]*len(tokens)))
            finally:draft.close()
            ended=perf_counter_ns()
            result['draft_work']=draft.stats()
            rows=[row for row in result['rounds'] if row['draft']]
            result['acceptance']=dict(accepted=result['accepted_draft_tokens'],proposed=result['proposed_draft_tokens'],
                strict_rate=result['accepted_draft_tokens']/result['proposed_draft_tokens'] if result['proposed_draft_tokens'] else None,
                draft_rounds=len(rows),committed_tokens_in_draft_rounds=sum(len(row['committed']) for row in rows),
                mean_committed_per_draft_round=sum(len(row['committed']) for row in rows)/len(rows) if rows else None)
            if not checking:result['metrics']=commit_metrics(stamps,started,ended)
            return dict(path='dual',width=active_width,**result)
        phase='check_baseline'
        baseline_result=baseline()
        baseline_result.pop('metrics')
        report['baseline']=baseline_result
        save()
        def validate(result):
            if result['output_ids'] != report['baseline']['output_ids'] or result['free_slots'] != capacity:
                raise RuntimeError('output or target cleanup differs from independent greedy')
            if result['path']=='dual' and (not result['draft_work']['closed'] or result['draft_work']['free_slots'] != capacity
                    or result['draft_work']['linear_states_remaining'] or result['draft_work']['snapshot_retained']):
                raise RuntimeError('draft cleanup did not return all slots')
        for active_width in sorted({1,3,width}):
            phase=f'check_width{active_width}'
            result=dual(active_width,True)
            report['checked_runs'].append(result);save();validate(result)
        if repeats:
            for name in ('baseline','dual'):
                phase='warm_'+name
                result=baseline() if name=='baseline' else dual(width)
                report['warmups'].append(result);save();validate(result)
            for repeat in range(repeats):
                for name in (('baseline','dual') if repeat%2==0 else ('dual','baseline')):
                    phase=f'timed_{repeat}_{name}'
                    result=baseline() if name=='baseline' else dual(width)
                    result['repeat']=repeat
                    report['trials'].append(result);save();validate(result)
            report['median_metrics']={name:{key:median(row['metrics'][key] for row in report['trials'] if row['path']==name)
                for key in ('ttft_ms','tpot_ms','e2e_ms','whole_ms','output_tokens_per_second')} for name in ('baseline','dual')}
        if tokenizer:report['output_text']=tokenizer.decode(report['baseline']['output_ids'])
        report['total_forward_calls']=sum(row['forward_calls'] for roles in report['model_work'].values() for row in roles.values())
        report.update(passed=True,status='passed');save()
        return report
    except Exception as error:
        report.update(status='failed',failure_count=1,error=f'{type(error).__name__}: {error}')
        save()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model',type=Path)
    parser.add_argument('--draft-model',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--width',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=0,help='0 checks only; 1..3 paired timings after one warmup per path')
    parser.add_argument('--output',type=Path,default=Path('outputs/model-draft.json'))
    args=parser.parse_args()
    report=run(args.output,target_directory=args.target_model,draft_directory=args.draft_model,
               device=args.device,width=args.width,repeats=args.repeats)
    print('greedy outputs and both caches match:',report['passed'])
    for row in report['checked_runs']:
        print(f"width={row['width']}: accepted/proposed={row['acceptance']['accepted']}/{row['acceptance']['proposed']}; "
              f"target forward={row['target_forward_calls']}; draft forward={row['draft_work']['forward_calls']}")
    if report.get('median_metrics'):print(json.dumps(report['median_metrics'],ensure_ascii=False))


if __name__=='__main__':
    main()
