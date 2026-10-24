"""Compare the pinned late-upload runner with the current early-upload runner.

The baseline is loaded only for this experiment. Both arms use the same model,
request loop and cache layout. Optional traces are collected after timing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
from unittest.mock import patch

BASELINE_SHA256 = 'f5a747a9b30df5b7a2802d43f7a42a60ea4020efea1aa3f37438de1feabcbf10'


def load_baseline(path):
    source = path.read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
        raise ValueError('baseline must be deferred_runner.py from commit 84a5f55')
    namespace = dict(__name__='minideepinfer._late_sampler', __package__='minideepinfer')
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace['DeferredRunner']


def collect(model, tokenizer, runner_class, output):
    import torch
    import torch_npu
    import minideepinfer.deferred_runner as adapter
    from minideepinfer.benchmark_prepare import MeasuredEngine, MeasuredTokenizer, run_trial
    from minideepinfer.model_runner import ModelRunner

    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
        record_shapes=False, profile_memory=False, with_stack=False, experimental_config=config)
    original_tensor = torch.tensor
    def tensor(values, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == 'launch':
            with torch.profiler.record_function('mini.sampler_indices'):
                return original_tensor(values, *args, **kwargs)
        return original_tensor(values, *args, **kwargs)
    def annotated(method, name, advance=False):
        def call(*values, **kwargs):
            with torch.profiler.record_function(name):
                result = method(*values, **kwargs)
            if advance:
                profiler.step()
            return result
        return call
    with patch.object(adapter, 'DeferredRunner', runner_class), \
         patch.object(torch, 'tensor', tensor), \
         patch.object(ModelRunner, 'execute_main_model', annotated(ModelRunner.execute_main_model, 'mini.forward')), \
         patch.object(MeasuredTokenizer, 'encode', annotated(MeasuredTokenizer.encode, 'mini.tokenize')), \
         patch.object(MeasuredEngine, 'launch_step', annotated(MeasuredEngine.launch_step, 'mini.launch')), \
         patch.object(MeasuredEngine, 'finish_step', annotated(MeasuredEngine.finish_step, 'mini.finish', True)):
        with profiler:
            result = run_trial(model, tokenizer, True)
    return result


def run(args):
    import torch
    import minideepinfer.deferred_runner as adapter
    from minideepinfer.benchmark_prepare import TinyTokenizer, run_trial
    from minideepinfer.model_runner import ModelRunner, select_device
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM

    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    if args.trace_output and not args.device.startswith('npu:'):
        raise ValueError('trace collection requires an NPU')
    late = load_baseline(args.baseline_source)  # Validate before loading weights.
    variants = {'late': late, 'early': adapter.DeferredRunner}
    torch.set_num_threads(1)
    if args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = ModelRunner.from_pretrained(args.model, args.device, torch.float32, 128).model
        model_name = Path(args.model).name
    else:
        tokenizer = TinyTokenizer()
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=128, partial_rotary_factor=0.5)
        model = Qwen3_5ForCausalLM(config).to(select_device(args.device)).eval()
        model_name = 'tiny-random-Qwen3.5'
    def trial(variant):
        with patch.object(adapter, 'DeferredRunner', variants[variant]):
            return run_trial(model, tokenizer, True)
    def signature(result):
        return (result['plans'], [(r['prompt_ids'], r['output_ids'], r['finish_reason'])
                                 for r in result['requests']])

    # Numerical controls copy full logits to CPU; none of their timings enter
    # the performance samples. The actual model implementation is unchanged.
    captures, controls = {}, {}
    original = ModelRunner.execute_main_model
    for variant in variants:
        values = []
        def forward(self, inputs):
            logits = original(self, inputs)
            values.append(logits.float().cpu())
            return logits
        with patch.object(ModelRunner, 'execute_main_model', forward):
            controls[variant] = trial(variant)
        captures[variant] = values
    if len(captures['late']) != len(captures['early']):
        raise RuntimeError('numerical trajectories have different lengths')
    comparisons = [dict(max_abs=float((a-b).abs().max()),
                        passed=bool(torch.allclose(a, b, atol=1e-4, rtol=2e-4)))
                   for a, b in zip(captures['late'], captures['early'])]
    warmups = [trial(variant) for variant in variants]
    trials = []
    for pair in range(args.repeats):
        for variant in (('late', 'early') if pair % 2 == 0 else ('early', 'late')):
            result = trial(variant)
            result.update(variant=variant, pair=pair)
            trials.append(result)
            print(f"第 {pair + 1} 组 {variant}：{result['elapsed_ms']:.3f} ms", flush=True)
    reference = signature(controls['late'])
    all_runs = list(controls.values()) + warmups + trials
    same = all(signature(result) == reference for result in all_runs)
    summary = {}
    for variant in variants:
        group = [r for r in trials if r['variant'] == variant]
        summary[variant] = dict(median_elapsed_ms=median(r['elapsed_ms'] for r in group),
            median_output_tokens_per_s=median(r['output_tokens_per_s'] for r in group),
            median_mean_ttft_ms=median(sum(q['ttft_ms'] for q in r['requests'])/4 for r in group),
            median_mean_tpot_ms=median(sum(q['tpot_ms'] for q in r['requests'])/4 for r in group),
            median_mean_e2e_ms=median(sum(q['e2e_ms'] for q in r['requests'])/4 for r in group))
    # Deduplicate identical batch plans; preserve every timing and token sample.
    for result in trials:
        if result.pop('plans') != reference[0]:
            raise RuntimeError('measured batch plans differ')
    root = Path(__file__).resolve().parents[1]
    names = ('minideepinfer/deferred_runner.py', 'minideepinfer/benchmark_prepare.py',
             'minideepinfer/text_loop.py', 'minideepinfer/batch_engine.py',
             'minideepinfer/model_runner.py', 'minideepinfer/engine.py',
             'minideepinfer/hybrid_cache.py', 'minideepinfer/tensor_cache.py',
             'minideepinfer/kv_cache.py', 'minideepinfer/npu_events.py',
             'minideepinfer/models/qwen3_5.py', 'minideepinfer/models/qwen2.py',
             'examples/compare_sampler_order.py')
    measured_ok = same and all(c['passed'] for c in comparisons) and all(r['clean'] for r in all_runs)
    report = dict(model=model_name, device=str(next(model.parameters()).device), dtype='float32',
        python=platform.python_version(), torch=torch.__version__,
        scope='paired late/early index upload; one shared model; fixed four-request workload',
        latency_origin='start of each tokenizer.encode to CPU token commit; no network',
        max_num_seqs=2, token_budget=4, cache_capacity=128, max_new_tokens=4, eos_disabled=True,
        numerical_control_trials=2, warmup_trials=2, repeats=args.repeats, failure_count=0,
        baseline_sha256=BASELINE_SHA256, shared_plans=reference[0], trials=trials, summary=summary,
        logit_comparisons=comparisons, atol=1e-4, rtol=2e-4, traces={},
        same_plans_and_outputs=same,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        measurement_passed=measured_ok, profile_requested=bool(args.trace_output),
        profile_complete=not bool(args.trace_output), passed=measured_ok and not args.trace_output)
    if args.model:
        report['checkpoint_config_sha256'] = hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest()
    if args.device.startswith('npu:'):
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + '.partial')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output)
    save()  # Preserve valid measurements even if later trace collection fails.
    if args.trace_output and measured_ok:
        for variant, runner_class in variants.items():
            result = collect(model, tokenizer, runner_class, args.trace_output / variant)
            report['traces'][variant] = dict(clean=result['clean'], same_trajectory=signature(result)==reference)
            save()
        report['profile_complete'] = True
        report['passed'] = measured_ok and all(t['clean'] and t['same_trajectory']
                                              for t in report['traces'].values())
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--model', help='local weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trace-output', type=Path, help='collect three steps for each variant after timing')
    parser.add_argument('--output', type=Path, default=Path('outputs/sampler-order.json'))
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"相同批次与输出={report['same_plans_and_outputs']}；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
