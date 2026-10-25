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


def run(output=None, *, target_directory=None,draft_directory=None,device='cpu',width=1,repeats=0,
        compare_draft_snapshots=False,trace_rounds=False,cpu_example='repeat'):
    if bool(target_directory) != bool(draft_directory):
        raise ValueError('provide both local checkpoint directories, or neither')
    if type(width) is not int or not 1 <= width <= 3 or type(repeats) is not int or not 0 <= repeats <= 3:
        raise ValueError('width must be 1..3 and repeats must be 0..3')
    if type(compare_draft_snapshots) is not bool or (compare_draft_snapshots and width != 1):
        raise ValueError('draft snapshot comparison requires width 1')
    if type(trace_rounds) is not bool or (trace_rounds and repeats):
        raise ValueError('round observations and timed trials must run separately')
    if cpu_example not in ('repeat','rejection') or (cpu_example=='rejection' and (target_directory or device!='cpu' or repeats)):
        raise ValueError('rejection is an untied random CPU example without checkpoints or timing')
    import torch
    from .engine import Engine
    from .model_runner import ModelRunner,select_device
    from .model_draft import ModelDraft,check_tokenizer_pair
    from .models.qwen3_5 import Qwen3_5Config,Qwen3_5ForCausalLM
    from .spec_demo import compare_cache
    from .speculative import generate
    from .model_draft_trace import DraftTrace

    torch.set_num_threads(1)
    report=dict(passed=False,status='running',failure_count=0,device=device,dtype='float32',torch=torch.__version__,python=platform.python_version(),cpu_threads=1,
                prompt_ids=[2,3,4],output_length=10,draft_width=width,repeats=repeats,
                checks=[],checked_runs=[],warmups=[],trials=[],model_work={},source_sha256={},
                scope='one request; eager greedy; separate draft cache; no MTP or probability sampling',
                timing_scope='offline CPU commits; cache initialization and final cleanup included in whole; accepted groups share a commit timestamp; cold references excluded from timed trials')
    if compare_draft_snapshots:
        report['comparison']='always_snapshot versus conditional_snapshot; both use the same aligned-cache observer; only the former checkpoints the draft at width 1; target snapshots unchanged'
    if cpu_example=='rejection':
        report['cpu_example']=dict(name='rejection',target_seed=2026,draft_seed=2036,
                                   tie_word_embeddings=False,trained=False,modified_candidates=False)
    if trace_rounds:
        report['trace_runs']=[]
        report['trace_scope']='Selected-width correctness runs only; CPU token lists and logical cache lengths, not device timestamps or profiling. No timed trials.'
    def save():
        if output is not None:
            path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.tmp')
            temporary.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode())
            temporary.replace(path)
    sources=('model_draft.py','model_draft_demo.py','model_draft_trace.py','speculative.py','spec_demo.py','models/qwen2.py',
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
                ('linear_attention',)*3+('full_attention',),max_position_embeddings=64,partial_rotary_factor=0.5,
                tie_word_embeddings=cpu_example=='repeat')).to(select_device(device)).eval()
            torch.manual_seed(2036)
            small=Qwen3_5ForCausalLM(Qwen3_5Config(97,16,24,2,2,1,8,1,2,4,4,
                ('linear_attention','full_attention'),max_position_embeddings=64,partial_rotary_factor=0.5,
                tie_word_embeddings=cpu_example=='repeat')).to(select_device(device)).eval()
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
                try:return _original(*args,**kwargs)
                except BaseException:
                    report['failed_operation']=dict(phase=phase,role=_role,operation='forward')
                    raise
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
                report['failed_operation']=dict(phase=phase,role=role,operation='cache_check')
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
        def dual(active_width,checking=False,snapshot_reference=False):
            stamps=[]
            trace=None
            started=perf_counter_ns()
            def aligned(cache):
                if trace is not None:trace.aligned(cache)
                if checking:check_cache(cache,'draft',active_width)
                # Reproduce the previous checkpoint location before argmax. Both timed
                # policies call this observer; the execution and target are identical.
                if snapshot_reference and active_width==1:cache.checkpoint()
            draft=ModelDraft(small,capacity,on_cache=aligned)
            if trace_rounds and checking and active_width==width:
                trace=DraftTrace(draft,prompt)
                trace.report.update(width=active_width,path='always_snapshot' if snapshot_reference else 'conditional_snapshot' if compare_draft_snapshots else 'dual')
                report['trace_runs'].append(trace.report)
            def committed(tokens):
                stamps.extend([perf_counter_ns()]*len(tokens))
                if trace is not None:trace.committed(tokens)
            result=generation_error=cleanup_error=None
            try:
                try:
                    result=generate(target,prompt,count,trace.propose if trace is not None else draft,width=active_width,
                                    on_cache=(lambda c:check_cache(c,'target',active_width)) if checking else None,
                                    on_commit=committed,on_round=trace.observed if trace is not None else None)
                except BaseException as error:
                    generation_error=error
                    raise
            finally:
                try:draft.close()
                except BaseException as error:
                    cleanup_error=error
                    raise
                finally:
                    if generation_error is not None or cleanup_error is not None:
                        report['failed_run']=dict(width=active_width,phase=phase,
                            snapshot_reference=snapshot_reference,generation_returned=result is not None,
                            generation_error=f'{type(generation_error).__name__}: {generation_error}' if generation_error is not None else None,
                            target_cleanup=dict(status='completed',free_slots=result['free_slots']) if result is not None else dict(status='NOT_OBSERVED'),
                            draft_cleanup=dict(status='failed' if cleanup_error is not None else 'completed',
                                error=f'{type(cleanup_error).__name__}: {cleanup_error}' if cleanup_error is not None else None,
                                stats=draft.stats()))
            ended=perf_counter_ns()
            result['draft_work']=draft.stats()
            if trace is not None:
                trace.closed()
                trace.report['target_free_slots']=result['free_slots']
            rows=[row for row in result['rounds'] if row['draft']]
            result['acceptance']=dict(accepted=result['accepted_draft_tokens'],proposed=result['proposed_draft_tokens'],
                strict_rate=result['accepted_draft_tokens']/result['proposed_draft_tokens'] if result['proposed_draft_tokens'] else None,
                draft_rounds=len(rows),committed_tokens_in_draft_rounds=sum(len(row['committed']) for row in rows),
                mean_committed_per_draft_round=sum(len(row['committed']) for row in rows)/len(rows) if rows else None)
            if not checking:result['metrics']=commit_metrics(stamps,started,ended)
            name=('always_snapshot' if snapshot_reference else 'conditional_snapshot') if compare_draft_snapshots else 'dual'
            return dict(path=name,width=active_width,**result)
        phase='check_baseline'
        baseline_result=baseline()
        baseline_result.pop('metrics')
        report['baseline']=baseline_result
        save()
        def validate(result):
            if result['output_ids'] != report['baseline']['output_ids'] or result['free_slots'] != capacity:
                raise RuntimeError('output or target cleanup differs from independent greedy')
            if result['path']!='baseline' and (not result['draft_work']['closed'] or result['draft_work']['free_slots'] != capacity
                    or result['draft_work']['linear_states_remaining'] or result['draft_work']['snapshot_retained']):
                raise RuntimeError('draft cleanup did not return all slots')
        def validate_pair(rows):
            def signature(row):
                return {key:({k:v for k,v in value.items() if k not in ('snapshot_count','snapshot_bytes_total','peak_snapshot_bytes')}
                        if key=='draft_work' else value) for key,value in row.items() if key not in ('path','metrics','repeat')}
            if signature(rows[0]) != signature(rows[1]):
                raise RuntimeError('snapshot policies changed output, state or model work')
        for active_width in sorted({1,3,width}):
            for reference in ((True,False) if compare_draft_snapshots else (False,)):
                phase=f'check_width{active_width}'+('_always' if reference else '_conditional')
                result=dual(active_width,True,reference)
                report['checked_runs'].append(result);save();validate(result)
            if compare_draft_snapshots:validate_pair(report['checked_runs'][-2:])
        paths=('always_snapshot','conditional_snapshot') if compare_draft_snapshots else ('baseline','dual')
        def execute(name):
            return baseline() if name=='baseline' else dual(width,snapshot_reference=name=='always_snapshot')
        if repeats:
            for name in paths:
                phase='warm_'+name
                result=execute(name)
                report['warmups'].append(result);save();validate(result)
            if compare_draft_snapshots:validate_pair(report['warmups'])
            for repeat in range(repeats):
                for name in (paths if repeat%2==0 else paths[::-1]):
                    phase=f'timed_{repeat}_{name}'
                    result=execute(name)
                    result['repeat']=repeat
                    report['trials'].append(result);save();validate(result)
                if compare_draft_snapshots:validate_pair(report['trials'][-2:])
            report['median_metrics']={name:{key:median(row['metrics'][key] for row in report['trials'] if row['path']==name)
                for key in ('ttft_ms','tpot_ms','e2e_ms','whole_ms','output_tokens_per_second')} for name in paths}
        if compare_draft_snapshots:
            report['same_work_except_draft_snapshots']=True
        if tokenizer:report['output_text']=tokenizer.decode(report['baseline']['output_ids'])
        report['total_forward_calls']=sum(row['forward_calls'] for roles in report['model_work'].values() for row in roles.values())
        report.update(passed=True,status='passed');save()
        return report
    except BaseException as error:
        chain=[];seen=set();current=error
        while current is not None and id(current) not in seen:
            seen.add(id(current));chain.append(f'{type(current).__name__}: {current}')
            if current.__cause__ is not None:current=current.__cause__
            elif current.__suppress_context__:break
            else:current=current.__context__
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      failure_count=1,error=f'{type(error).__name__}: {error}',error_chain=chain,failure_phase=phase,
                      total_forward_calls=sum(row['forward_calls'] for roles in report['model_work'].values() for row in roles.values()),
                      model_work_scope='Forward entries and input tokens count attempts, including a forward that raised; they do not certify completed computation.')
        save()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model',type=Path)
    parser.add_argument('--draft-model',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--width',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=0,help='0 checks only; 1..3 paired timings after one warmup per path')
    parser.add_argument('--compare-draft-snapshots',action='store_true',help='width 1: compare the previous draft snapshot policy with conditional snapshots')
    parser.add_argument('--trace-rounds',action='store_true',help='observe the selected width during correctness checks; cannot be combined with timing')
    parser.add_argument('--cpu-example',choices=('repeat','rejection'),default='repeat',help='rejection uses untrained random CPU models with untied output heads; no candidate injection')
    parser.add_argument('--output',type=Path,default=Path('outputs/model-draft.json'))
    args=parser.parse_args()
    report=run(args.output,target_directory=args.target_model,draft_directory=args.draft_model,
               device=args.device,width=args.width,repeats=args.repeats,compare_draft_snapshots=args.compare_draft_snapshots,
               trace_rounds=args.trace_rounds,cpu_example=args.cpu_example)
    if report.get('trace_runs'):
        from .model_draft_trace import print_trace
        print('以下显示 token ID 和主机观察到的缓存进度，不是设备时间线。')
        if args.cpu_example=='rejection':print('未训练的随机 CPU 模型；输出头与输入嵌入不共享权重，候选来自实际模型预测。')
        for trace in report['trace_runs']:print_trace(trace)
    print('greedy outputs and both caches match:',report['passed'])
    for row in report['checked_runs']:
        print(f"{row['path']} width={row['width']}: accepted/proposed={row['acceptance']['accepted']}/{row['acceptance']['proposed']}; "
              f"target forward={row['target_forward_calls']}; draft forward={row['draft_work']['forward_calls']}")
    if report.get('median_metrics'):print(json.dumps(report['median_metrics'],ensure_ascii=False))


if __name__=='__main__':
    main()
