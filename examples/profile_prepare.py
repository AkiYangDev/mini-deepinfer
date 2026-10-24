"""Collect three warm Ascend steps separately from the unprofiled comparison.

Run from the repository root with: python -m examples.profile_prepare --help
This wrapper annotates existing methods; it does not replace model operations.
"""

import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='local Qwen3.5 checkpoint')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--output', type=Path, default=Path('outputs/prepare-profile'))
    args = parser.parse_args()
    if not args.device.startswith('npu:'):
        parser.error('this wrapper collects an Ascend NPU trace')

    import torch
    import torch_npu
    from transformers import AutoTokenizer
    from minideepinfer.benchmark_prepare import MeasuredEngine, MeasuredTokenizer, run_trial
    from minideepinfer.model_runner import ModelRunner

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = ModelRunner.from_pretrained(args.model, args.device, torch.float32, 128).model
    warmup = run_trial(model, tokenizer, True)
    if not warmup['clean']:
        raise RuntimeError('warmup did not release its request state')
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
        record_shapes=False, profile_memory=False, with_stack=False,
        experimental_config=config)

    def annotated(method, name, advance=False):
        @wraps(method)
        def call(*values, **kwargs):
            with torch.profiler.record_function(name):
                result = method(*values, **kwargs)
            if advance:
                profiler.step()
            return result
        return call

    with patch.object(MeasuredTokenizer, 'encode', annotated(MeasuredTokenizer.encode, 'mini.tokenize')), \
         patch.object(MeasuredEngine, 'launch_step', annotated(MeasuredEngine.launch_step, 'mini.launch')), \
         patch.object(MeasuredEngine, 'finish_step', annotated(MeasuredEngine.finish_step, 'mini.finish', True)):
        with profiler:
            trial = run_trial(model, tokenizer, True)
    if not trial['clean'] or trial['plans'] != warmup['plans']:
        raise RuntimeError('profiled trajectory or cleanup differs from warmup')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope='three warmed steps; profiler overhead excluded from benchmark report',
                  python_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  active_steps=3, level=0, cpu=True, npu=True, shapes=False, stacks=False,
                  passed=trial['clean'], plans_match_warmup=True, output_tokens=trial['output_tokens'])
    (args.output / 'collection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
