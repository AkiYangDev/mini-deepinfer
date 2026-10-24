"""Compare first-round collection with identical fixed-member pipeline plans."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns

from .benchmark_pipeline import TEXTS, run_trial, save_report


def run(model_directory=None, device='cpu', repeats=3, output=None):
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError('repeats must be between 1 and 3')
    import torch
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM

    torch.set_num_threads(1)
    start = perf_counter_ns()
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        prompt = tuple(tokenizer.encode(TEXTS[0]))
        model = ModelRunner.from_pretrained(model_directory, device, torch.float32, 128).model
        name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',)*3+('full_attention',),
                              max_position_embeddings=128, partial_rotary_factor=0.5)
        model = Qwen3_5ForCausalLM(config).to(select_device(device)).eval()
        prompt, name = (2,3,4,5,6), 'tiny-random-Qwen3.5'
    if device.startswith('npu'):
        torch.npu.synchronize()
    sources = ('benchmark_first_result.py', 'benchmark_pipeline.py', 'stopping_pipeline.py',
               'decode_pipeline.py', 'batch_engine.py', 'model_runner.py', 'planning.py', 'engine.py',
               'npu_events.py', 'step_state.py', 'deferred_runner.py', 'kv_cache.py',
               'hybrid_cache.py', 'tensor_cache.py', 'models/qwen3_5.py', 'models/qwen2.py')
    report = dict(model=name, device=device, dtype='float32', python=platform.python_version(),
                  torch=torch.__version__, setup_model_and_tokenization_ms=(perf_counter_ns()-start)/1e6,
                  prompt_ids=prompt, raw_text=TEXTS[0] if model_directory else None,
                  repeats=repeats, output_length=8, eos_disabled=True, capacity=128,
                  max_num_seqs=2, token_budget=16, chunked_prefill=True,
                  scope='offline common arrival; CPU commit including round cleanup; whole includes final drain',
                  passed=False, status='running', failure_count=0, checks=[], cases={},
                  source_sha256={'minideepinfer/'+p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
                                 for p in sources})
    if device.startswith('npu'):
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    if model_directory:
        report['checkpoint_config_sha256'] = hashlib.sha256((Path(model_directory)/'config.json').read_bytes()).hexdigest()
    save_report(output, report)

    def measure(prompts, early, count=8):
        trial = run_trial(model, prompts, 'pipeline', count=count, budget=16, early_first_result=early)
        trial['first_result'] = 'early' if early else 'delayed'
        return trial

    try:
        for early in (False, True):
            trial = measure([prompt], early, 1)
            report['checks'].append(trial)
            save_report(output, report)
            if not trial['passed'] or trial['outputs'] != report['checks'][0]['outputs']:
                raise ValueError('one-token final drain mismatch')
        for case, prompts in (('single',[prompt]), ('two_active',[prompt,prompt])):
            result = dict(prompt_lengths=list(map(len,prompts)), warmups=[], trials=[])
            report['cases'][case] = result
            reference = None
            for pair in range(-1,repeats):
                for early in ((False,True) if pair<0 or pair%2==0 else (True,False)):
                    trial = measure(prompts,early)
                    trial['pair'] = pair
                    if reference is None:
                        reference = trial
                    trial['outputs_match'] = trial['outputs']==reference['outputs']
                    trial['plans_match'] = trial['plans']==reference['plans']
                    result['warmups' if pair==-1 else 'trials'].append(trial)
                    save_report(output, report)
                    if not (trial['passed'] and trial['outputs_match'] and trial['plans_match']):
                        raise ValueError(f'{case}: output, plan, work, or cleanup mismatch')
                    print(f"{case} pair={pair} {trial['first_result']}: {trial['elapsed_ms']:.3f} ms",flush=True)
            result['summary'] = {}
            for mode in ('delayed','early'):
                group = [t for t in result['trials'] if t['first_result']==mode]
                summary = {f'median_mean_{key}':median(sum(r[key] for r in t['requests'].values())/
                                                      len(t['requests']) for t in group)
                           for key in ('ttft_ms','tpot_ms','e2e_ms')}
                summary.update(median_elapsed_ms=median(t['elapsed_ms'] for t in group),
                               min_elapsed_ms=min(t['elapsed_ms'] for t in group),
                               max_elapsed_ms=max(t['elapsed_ms'] for t in group),
                               median_output_tokens_per_s=median(t['output_tokens_per_s'] for t in group))
                result['summary'][mode] = summary
        report.update(passed=True,status='complete')
    except Exception as error:
        report.update(status='failed',failure_count=1,error=f'{type(error).__name__}: {error}')
        save_report(output,report)
        raise
    report['total_forward_calls'] = sum(t['forward_calls'] for t in report['checks'])+sum(
        t['forward_calls'] for c in report['cases'].values() for t in c['warmups']+c['trials'])
    save_report(output,report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',help='existing local Qwen3.5 checkpoint; otherwise a tiny random model')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--repeats',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--output',type=Path,default=Path('outputs/first-result.json'))
    args = parser.parse_args()
    report = run(args.model,args.device,args.repeats,args.output)
    print(f"passed={report['passed']}; forwards={report['total_forward_calls']}")


if __name__ == '__main__':
    main()
