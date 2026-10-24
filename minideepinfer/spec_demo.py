"""Controlled greedy speculative decoding on a tiny CPU full-attention model."""

import argparse
import hashlib
import json
from pathlib import Path


def run(output=None, *, verbose=True):
    import torch
    from .engine import Engine
    from .model_runner import ModelRunner
    from .models.qwen2 import Qwen2Config,Qwen2ForCausalLM
    from .speculative import generate

    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model=Qwen2ForCausalLM(Qwen2Config(97,32,64,2,4,2,max_position_embeddings=64)).eval()
    prompt,count,width=(2,3,4),10,3
    runner=ModelRunner(model,12)
    engine=Engine(runner,12)
    calls=[]
    forward=runner.execute_main_model
    def capture(inputs):
        calls.append(len(inputs.input_ids))
        return forward(inputs)
    runner.execute_main_model=capture
    request=engine.submit(prompt,count)
    list(engine.run())
    golden=request.output_ids
    report=dict(passed=False,model='tiny-random-Qwen2',device='cpu',dtype='float32',seed=2026,
                torch=torch.__version__,prompt_ids=list(prompt),output_length=count,draft_width=width,
                scope='controlled candidates from a precomputed greedy oracle; no timing or speedup claim',
                oracle=dict(output_ids=golden,forward_calls=len(calls),input_tokens=sum(calls),free_slots=engine.cache.free_count),
                cases={},source_sha256={})
    if verbose:
        print('CPU FP32；随机小模型，无需权重。候选来自预先计算的 greedy 答案，用来控制拒绝位置。')
        print(f'准备答案用了 {len(calls)} 次 target forward；这些成本不计入各实验的 target 次数，不用于宣称加速。')
        print(f'普通 greedy 输出：{golden}')
    for name,reject in (('first_reject',0),('middle_reject',1),('all_accept',None)):
        def draft(prefix,n):
            start=len(prefix)-len(prompt)
            values=golden[start:start+n].copy()
            if reject is not None and reject<len(values):
                values[reject]=(values[reject]+1)%97
            return values
        if verbose:
            print(f'\n{name}：prefill 已提交 {golden[0]}，它的 KV 仍待下一轮写入。')
        def observe(row):
            if verbose:
                print(f"  输入 {row['input_ids']} → target {row['target']}")
                print(f"  接受 {row['accepted_draft']} 个候选；提交 {row['committed']}；最后来源={row['final_role']}")
                print(f"  KV 暂存 {row['tentative_cached']} → 保留 {row['retained_cached']}；"
                      f"失效槽位 {row['invalidated_slots']}；待计算 token={row['pending_token']}")
        result=generate(model,prompt,count,draft,width=width,on_round=observe)
        result['matches_greedy']=result['output_ids']==golden
        if not result['matches_greedy'] or result['free_slots']!=12:
            raise RuntimeError('speculative output or cleanup differs from the independent greedy run')
        report['cases'][name]=result
        if verbose:
            print(f"  输出相同；target forward={result['target_forward_calls']}，"
                  f"输入 token={result['target_input_tokens']}，槽位归还={result['free_slots']}/12")
    for name in ('speculative.py','spec_demo.py','models/qwen2.py','model_runner.py',
                 'tensor_cache.py','engine.py','kv_cache.py'):
        report['source_sha256']['minideepinfer/'+name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    report['passed']=True
    if output is not None:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/spec-demo.json'))
    args=parser.parse_args()
    run(args.output)


if __name__=='__main__':
    main()
