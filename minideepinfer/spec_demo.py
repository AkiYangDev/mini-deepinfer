"""Controlled greedy verification, with optional Qwen3.5 state rollback checks."""

import argparse
import hashlib
import json
from pathlib import Path


def compare_cache(target,atol=1e-5,rtol=1e-4,*,strict=True):
    """Independent cold full-prefix calculation; counts as an extra model call."""
    import torch
    from .speculative import CachedTarget
    cold=CachedTarget(target.model,target.slots,target.cache.capacity)
    cold_logits=cold.forward(tuple(target.tokens))
    rows=torch.tensor(target.slots[:len(target.tokens)],dtype=torch.long,device=target.device)
    result=dict(cached_tokens=len(target.tokens),reference_forward_calls=1,reference_input_tokens=len(target.tokens),passed=True)
    pairs={'keys':(target.cache.keys.index_select(1,rows),cold.cache.keys.index_select(1,rows)),
           'values':(target.cache.values.index_select(1,rows),cold.cache.values.index_select(1,rows))}
    if target.hybrid:
        pairs['logits']=(target.last_logits,cold_logits[-1])
        if target.cache.linear_states.keys()!=cold.cache.linear_states.keys():
            raise ValueError('linear-state owners differ from cold reference')
        for (rid,layer),(history,state) in target.cache.linear_states.items():
            wanted=cold.cache.linear_states[(rid,layer)]
            pairs[f'conv_{layer}']=(history,wanted[0]);pairs[f'recurrence_{layer}']=(state,wanted[1])
    for name,(actual,expected) in pairs.items():
        actual,expected=actual.detach().float().cpu(),expected.detach().float().cpu()
        result[name+'_max_abs']=float((actual-expected).abs().max()) if actual.numel() else 0.0
        try:
            torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
        except AssertionError:
            if strict:raise
            result['passed']=False
            result.setdefault('mismatches',[]).append(name)
    return result


def run(output=None, *, verbose=True,hybrid=False,device='cpu',model_directory=None,draft_mode='controlled'):
    if draft_mode not in ('controlled','lookup'):
        raise ValueError('draft_mode must be controlled or lookup')
    import torch
    from .engine import Engine
    from .model_runner import ModelRunner,select_device
    from .models.qwen2 import Qwen2Config,Qwen2ForCausalLM
    from .speculative import generate

    torch.set_num_threads(1)
    torch.manual_seed(2026)
    prompt,count,width=(2,3,4),10,3
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model_directory,local_files_only=True)
        prompt=tuple(tokenizer.encode('The capital of France is'))
        model=ModelRunner.from_pretrained(model_directory,device,torch.float32,128).model
        hybrid=hasattr(model.config,'layer_types')
    elif hybrid:
        from .models.qwen3_5 import Qwen3_5Config,Qwen3_5ForCausalLM
        config=Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,('linear_attention',)*3+('full_attention',),
                            max_position_embeddings=64,partial_rotary_factor=0.5)
        model=Qwen3_5ForCausalLM(config).to(select_device(device)).eval()
    else:
        model=Qwen2ForCausalLM(Qwen2Config(97,32,64,2,4,2,max_position_embeddings=64)).to(select_device(device)).eval()
    capacity=len(prompt)+count-1
    runner=ModelRunner(model,capacity)
    engine=Engine(runner,capacity)
    calls=[]
    forward=runner.execute_main_model
    def capture(inputs):
        calls.append(len(inputs.input_ids))
        return forward(inputs)
    runner.execute_main_model=capture
    def baseline():
        request=engine.submit(prompt,count)
        list(engine.run())
        return dict(output_ids=request.output_ids,forward_calls=len(calls),input_tokens=sum(calls),free_slots=engine.cache.free_count)
    reference=baseline() if draft_mode=='controlled' else None
    golden=reference['output_ids'] if reference else None
    report=dict(passed=False,model=Path(model_directory).name if model_directory else 'tiny-random-Qwen3.5' if hybrid else 'tiny-random-Qwen2',device=device,dtype='float32',seed=2026,
                torch=torch.__version__,prompt_ids=list(prompt),output_length=count,draft_width=width,
                scope=('controlled candidates from a precomputed greedy oracle' if reference else
                       'lookup candidates from committed context; independent greedy baseline runs afterwards')+'; no timing or speedup claim',
                cases={},source_sha256={},hybrid=hybrid,cache_checks=[])
    if reference:report['oracle']=reference
    if model_directory:
        report['checkpoint_config_sha256']=hashlib.sha256((Path(model_directory)/'config.json').read_bytes()).hexdigest()
    if device.startswith('npu'):
        import torch_npu
        report['torch_npu']=torch_npu.__version__
    if verbose:
        if reference:
            print(f'{device} FP32；候选来自预先计算的 greedy 答案，用来控制拒绝位置。')
            print(f'准备答案用了 {len(calls)} 次 target forward；这些成本不计入各实验的 target 次数，不用于宣称加速。')
            print(f'普通 greedy 输出：{golden}')
        else:
            print(f'{device} FP32；候选只从已提交的上下文查找，不运行 draft 模型。')
            print('先完成 lookup 生成，再用独立缓存运行普通 greedy 对照；不计时、不宣称加速。')
    sources=['speculative.py','spec_demo.py','models/qwen2.py','model_runner.py','tensor_cache.py','engine.py','kv_cache.py']
    if hybrid:sources+=['hybrid_cache.py','models/qwen3_5.py']
    if draft_mode=='lookup':sources+=['lookup_draft.py']
    for name in sources:
        report['source_sha256']['minideepinfer/'+name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    report.update(status='running',failure_count=0)
    def save():
        if output is not None:
            path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.tmp')
            temporary.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
            temporary.replace(path)
    save()
    try:
        cases=(('lookup',None),) if draft_mode=='lookup' else (('first_reject',0),('middle_reject',1),('all_accept',None))
        for name,reject in cases:
            lookup=None
            if draft_mode=='lookup':
                from .lookup_draft import LookupDraft
                lookup=LookupDraft()
                draft=lookup
            else:
                def draft(prefix,n):
                    start=len(prefix)-len(prompt)
                    values=golden[start:start+n].copy()
                    if reject is not None and reject<len(values):
                        values[reject]=(values[reject]+1)%model.config.vocab_size
                    return values
            if verbose:
                print(f'\n{name}：每轮先消费上轮已提交、尚未计算的末尾 token。')
            def observe(row):
                if lookup is not None:row['lookup']=lookup.last.copy()
                if verbose:
                    print(f"  输入 {row['input_ids']} → target {row['target']}")
                    print(f"  接受 {row['accepted_draft']} 个候选；提交 {row['committed']}；最后来源={row['final_role']}")
                    print(f"  KV 暂存 {row['tentative_cached']} → 保留 {row['retained_cached']}；"
                          f"失效槽位 {row['invalidated_slots']}；待计算 token={row['pending_token']}")
                    if hybrid:print(f"  恢复后重算 {row['replay_tokens']} 个输入，额外 forward={row['replay_calls']}")
            def check(target):
                if lookup is not None:report['draft']=lookup.stats()
                report['in_progress']=dict(case=name,target_forward_calls=target.forward_calls,
                                           target_input_tokens=target.input_tokens,replay_forward_calls=target.replay_calls,
                                           replayed_input_tokens=target.replay_tokens)
                save()
                values=compare_cache(target,atol=1e-4 if device.startswith('npu') else 1e-5,
                                     rtol=2e-4 if device.startswith('npu') else 1e-4,strict=False)
                report['cache_checks'].append(dict(case=name,**values))
                save()
                if not values['passed']:
                    raise RuntimeError('hybrid state differs from the cold full-prefix reference')
            result=generate(model,prompt,count,draft,width=width,on_round=observe,on_cache=check if hybrid else None)
            report['cases'][name]=result
            report.pop('in_progress',None)
            if lookup is not None:report['draft']=lookup.stats()
            save()
            if lookup is not None:
                reference=baseline()
                report['baseline']=reference
                golden=reference['output_ids']
            result['matches_greedy']=result['output_ids']==golden
            save()
            if not result['matches_greedy'] or result['free_slots']!=capacity or reference['free_slots']!=capacity:
                raise RuntimeError('speculative output or cleanup differs from the independent greedy run')
            if verbose:
                print(f"  输出相同；target forward={result['target_forward_calls']}，"
                      f"输入 token={result['target_input_tokens']}，槽位归还={result['free_slots']}/{capacity}")
    except Exception as error:
        report.update(status='failed',failure_count=1,error=f'{type(error).__name__}: {error}')
        save()
        raise
    report['reference_forward_calls']=sum(c['reference_forward_calls'] for c in report['cache_checks'])
    report['reference_input_tokens']=sum(c['reference_input_tokens'] for c in report['cache_checks'])
    report['total_forward_calls']=len(calls)+sum(c['target_forward_calls'] for c in report['cases'].values())+report['reference_forward_calls']
    report.update(passed=True,status='passed')
    save()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/spec-demo.json'))
    parser.add_argument('--hybrid',action='store_true',help='use the tiny Qwen3.5 snapshot/replay reference')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--model',help='existing local Qwen2/Qwen3.5 checkpoint; no downloads')
    parser.add_argument('--draft',choices=('controlled','lookup'),default='controlled',help='lookup uses committed context only; its greedy baseline runs afterwards')
    args=parser.parse_args()
    run(args.output,hybrid=args.hybrid,device=args.device,model_directory=args.model,draft_mode=args.draft)


if __name__=='__main__':
    main()
