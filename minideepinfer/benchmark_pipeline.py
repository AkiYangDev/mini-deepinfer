"""Offline sync/pipeline comparison: identical queued inputs, separate caches."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns

TEXTS = ('The capital of France is', 'What is 1 plus 1?',
         'A cache stores earlier keys and values so decoding can reuse them.', 'Hello')


def metrics(times, start, end):
    """CPU commit timestamps; the whole workload also includes final draining."""
    if end <= start or not times or any(not values for values in times.values()):
        raise ValueError('a successful trial needs a positive window and outputs for every request')
    requests = {}
    for rid, values in times.items():
        if values[0] < start or values[-1] > end or any(b < a for a, b in zip(values, values[1:])):
            raise ValueError('commit timestamps must be ordered inside the measured window')
        ttft, e2e = (values[0]-start)/1e6, (values[-1]-start)/1e6
        requests[rid] = dict(ttft_ms=ttft, e2e_ms=e2e, output_tokens=len(values),
                            tpot_ms=(e2e-ttft)/(len(values)-1) if len(values)>1 else None,
                            itl_ms=[(b-a)/1e6 for a,b in zip(values, values[1:])])
    elapsed = (end-start)/1e6
    return dict(elapsed_ms=elapsed, output_tokens_per_s=sum(map(len, times.values()))*1000/elapsed,
                drain_after_last_commit_ms=(end-max(v[-1] for v in times.values()))/1e6,
                requests=requests)


def run_trial(model, prompts, mode, *, count=8, capacity=128, limit=2, budget=8,
              early_first_result=False):
    from .batch_engine import BatchEngine
    from .model_runner import ModelRunner
    from .stopping_pipeline import StoppingDecodePipeline

    if mode not in ('sync', 'pipeline', 'collect_before_admit'):
        raise ValueError('choose sync, pipeline, or collect_before_admit')
    allocation = perf_counter_ns()
    runner = ModelRunner(model, capacity)
    plans, commits = [], []
    forward = runner.execute_main_model
    def capture(inputs):
        plans.append(inputs.steps)  # Immutable host metadata; no logits/tensor reads.
        return forward(inputs)
    runner.execute_main_model = capture

    class MeasuredPipeline(StoppingDecodePipeline):
        def _finish(self, state, outputs, on_result):
            super()._finish(state, outputs, on_result)
            stamp = perf_counter_ns()  # Includes this round's CPU commit and reclamation.
            commits.append((stamp, tuple((s.request_id, len(self.requests[s.request_id].output_ids))
                                         for s in state.steps)))

    if mode == 'sync':
        engine = BatchEngine(runner, capacity, limit, budget, chunked_prefill=True)
    else:
        engine = MeasuredPipeline(runner)
    runner.synchronize()
    setup_ms = (perf_counter_ns()-allocation)/1e6
    start = perf_counter_ns()  # Every request has this same logical arrival time.
    try:
        if mode == 'sync':
            requests = {f'req-{i}': engine.submit(p, count) for i,p in enumerate(prompts)}
            while outputs := engine.step():
                stamp = perf_counter_ns()
                commits.append((stamp, tuple((o.request_id, len(requests[o.request_id].output_ids))
                                             for o in outputs)))
            pool = engine.cache
        else:
            requests = engine.run({f'req-{i}': p for i,p in enumerate(prompts)}, count,
                                  max_num_seqs=limit, max_num_batched_tokens=budget, chunked_prefill=True,
                                  early_first_result=early_first_result,
                                  collect_before_admit=mode == 'collect_before_admit')
            pool = engine.pool
        runner.synchronize()
        end = perf_counter_ns()
    finally:
        runner.execute_main_model = forward
    times = {rid: [] for rid in requests}
    for stamp, rows in commits:
        for rid, length in rows:
            if length == len(times[rid])+1:
                times[rid].append(stamp)
            elif length != len(times[rid]):
                raise ValueError('greedy commit must add zero or one token per request')
    clean = pool.free_count == capacity and not runner._prefixes and not runner.cache.linear_states
    normalized = [[dict(request_id=s.request_id, phase=s.phase, positions=s.positions,
                        partial=not s.is_last_prompt_chunk) for s in batch] for batch in plans]
    processed = sum(len(s.input_ids) for batch in plans for s in batch)
    expected = sum(len(p)+count-1 for p in prompts)
    valid = (clean and processed == expected and all(len(r.output_ids)==count and r.finish_reason=='length'
             and len(times[rid])==count for rid,r in requests.items()))
    return dict(mode=mode, **metrics(times, start, end), setup_ms=setup_ms, forward_calls=len(plans),
                processed_input_tokens=processed, expected_input_tokens=expected,
                extra_input_tokens=processed-expected, plans=normalized,
                outputs={rid:r.output_ids for rid,r in requests.items()}, free_slots=pool.free_count,
                clean=clean, passed=valid)


def summarize(trials, modes=('sync', 'pipeline')):
    summary = {}
    for mode in modes:
        group = [t for t in trials if t['mode']==mode]
        means = lambda field: [sum(r[field] for r in t['requests'].values())/len(t['requests']) for t in group]
        summary[mode] = dict(median_elapsed_ms=median(t['elapsed_ms'] for t in group),
                            min_elapsed_ms=min(t['elapsed_ms'] for t in group),
                            max_elapsed_ms=max(t['elapsed_ms'] for t in group),
                            median_output_tokens_per_s=median(t['output_tokens_per_s'] for t in group),
                            median_mean_ttft_ms=median(means('ttft_ms')),
                            median_mean_e2e_ms=median(means('e2e_ms')),
                            median_mean_tpot_ms=median(means('tpot_ms')) if all(
                                r['tpot_ms'] is not None for t in group for r in t['requests'].values()) else None)
    return summary


def save_report(path, report):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.partial')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)


def run(model_directory=None, device='cpu', repeats=3, output=None, *, compare_admission=False):
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError('repeats must be between 1 and 3')
    if type(compare_admission) is not bool:
        raise ValueError('compare_admission must be boolean')
    import torch
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM

    torch.set_num_threads(1)
    load_start = perf_counter_ns()
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        model = ModelRunner.from_pretrained(model_directory, device, torch.float32, 128).model
        prompts = [tuple(tokenizer.encode(text)) for text in TEXTS]
        name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',)*3+('full_attention',),
                              max_position_embeddings=128, partial_rotary_factor=0.5)
        model = Qwen3_5ForCausalLM(config).to(select_device(device)).eval()
        prompts = [tuple(range(2, 2+n)) for n in (5, 8, 15, 1)]
        name = 'tiny-random-Qwen3.5'
    if str(next(model.parameters()).device).startswith('npu'):
        torch.npu.synchronize()
    sources = ['benchmark_pipeline.py', 'batch_engine.py', 'stopping_pipeline.py', 'decode_pipeline.py',
               'planning.py', 'model_runner.py', 'engine.py', 'step_state.py', 'npu_events.py',
               'deferred_runner.py', 'kv_cache.py', 'hybrid_cache.py', 'tensor_cache.py',
               'models/qwen3_5.py', 'models/qwen2.py']
    modes = ('pipeline', 'collect_before_admit') if compare_admission else ('sync', 'pipeline')
    report = dict(model=name, device=device, dtype='float32', python=platform.python_version(),
                  torch=torch.__version__, setup_model_and_tokenization_ms=(perf_counter_ns()-load_start)/1e6,
                  raw_texts=list(TEXTS) if model_directory else None, repeats=repeats,
                  scope='offline pre-tokenized requests, common arrival, CPU commit including round cleanup',
                  elapsed_scope='submission through final drain; cache initialization and model loading excluded',
                  eos_disabled=True, output_length=8, capacity=128, max_num_seqs=2, token_budget=8,
                  failure_count=0, status='running', passed=False, cases={}, comparison=list(modes),
                  source_sha256={'minideepinfer/'+p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
                                 for p in sources})
    if device.startswith('npu'):
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    if model_directory:
        report['checkpoint_config_sha256'] = hashlib.sha256((Path(model_directory)/'config.json').read_bytes()).hexdigest()
    workloads = {'single':[prompts[0]], 'equal_batch':[prompts[0]]*4, 'mixed_batch':prompts}
    save_report(output, report)
    try:
        for case, inputs in workloads.items():
            result = dict(prompt_ids=inputs, prompt_lengths=list(map(len, inputs)), warmups=[], trials=[])
            report['cases'][case] = result
            reference = None
            for pair in range(-1, repeats):
                for mode in (modes if pair < 0 or pair%2==0 else modes[::-1]):
                    trial = run_trial(model, inputs, mode)
                    trial['pair'] = pair
                    if reference is None:
                        reference = trial['outputs']
                    trial['outputs_match'] = trial['outputs'] == reference
                    result['warmups' if pair==-1 else 'trials'].append(trial)
                    save_report(output, report)  # Persist before validation or another costly run.
                    if not trial['passed'] or not trial['outputs_match']:
                        raise ValueError(f'{case}/{mode}: output, workload, or cleanup mismatch')
                    print(f"{case} pair={pair} {mode}: {trial['elapsed_ms']:.3f} ms", flush=True)
            result['summary'] = summarize(result['trials'], modes)
            result['same_plans'] = all(t['plans']==result['warmups'][0]['plans']
                                       for t in result['warmups']+result['trials'])
        report.update(passed=True, status='complete')
    except Exception as error:
        report.update(failure_count=1, status='failed', error=f'{type(error).__name__}: {error}')
        save_report(output, report)
        raise
    report['total_forward_calls'] = sum(t['forward_calls'] for c in report['cases'].values()
                                       for t in c['warmups']+c['trials'])
    save_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='existing local Qwen3.5 checkpoint; otherwise a tiny CPU/NPU model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeats', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--compare-admission', action='store_true',
                        help='compare two pipeline modes, changing only collection before waiting admission')
    parser.add_argument('--output', type=Path, default=Path('outputs/pipeline-benchmark.json'))
    args = parser.parse_args()
    report = run(args.model, args.device, args.repeats, args.output, compare_admission=args.compare_admission)
    print(f"passed={report['passed']}; forwards={report['total_forward_calls']}")


if __name__ == '__main__':
    main()
