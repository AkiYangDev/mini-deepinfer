"""Compare preparing the next text before versus after collecting a batch.

This measures an offline four-request workload, not an HTTP serving benchmark.
Both orders use the same single-flight DeferredRunner and batch scheduler.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns

from .batch_engine import BatchEngine
from .text_loop import run_text_requests

PROMPTS = ('The capital of France is', 'A KV cache',
           'Explain prefill in one sentence.', 'What is 1 plus 1?')


class TinyTokenizer:
    eos_token_id = None

    def encode(self, text):
        return [1 + sum(word.encode()) % 96 for word in text.split()]


class MeasuredTokenizer:
    eos_token_id = None  # Fixed output length for this controlled comparison.

    def __init__(self, tokenizer):
        self.tokenizer, self.spans = tokenizer, []

    def encode(self, text):
        start = perf_counter_ns()
        tokens = self.tokenizer.encode(text)
        self.spans.append((start, perf_counter_ns(), tokens))
        return tokens


class MeasuredEngine(BatchEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plans, self.commits = [], []

    def launch_step(self):
        plans = super().launch_step()
        if plans:
            self.plans.append([asdict(p) for p in plans])
        return plans

    def finish_step(self):
        outputs = super().finish_step()
        self.commits.append((perf_counter_ns(), outputs))
        return outputs


def run_trial(model, tokenizer, during):
    from .deferred_runner import DeferredRunner
    from .model_runner import ModelRunner

    runner = ModelRunner(model, 128)
    deferred = DeferredRunner(runner)
    engine = MeasuredEngine(deferred, 128, 2, 4, chunked_prefill=True, deferred=True)
    measured = MeasuredTokenizer(tokenizer)
    runner.synchronize()  # Cache initialization is outside the measured window.
    start = perf_counter_ns()
    try:
        pairs = run_text_requests(engine, measured, PROMPTS, raw=True,
                                  max_new_tokens=4, prepare_during_step=during)
        end = perf_counter_ns()  # Includes tokenization, queueing, generation, and release.
        requests = []
        for (request, text), (arrival, prepared, tokens) in zip(pairs, measured.spans):
            times = [when for when, outputs in engine.commits
                     for output in outputs if output.request_id == request.request_id
                     and output.token_id is not None]
            ttft, e2e = (times[0] - arrival) / 1e6, (times[-1] - arrival) / 1e6
            requests.append(dict(request_id=request.request_id, prompt=text, prompt_ids=list(tokens),
                output_ids=list(request.output_ids), finish_reason=request.finish_reason,
                tokenize_ms=(prepared - arrival) / 1e6, ttft_ms=ttft, e2e_ms=e2e,
                tpot_ms=(e2e - ttft) / (len(times) - 1),
                itl_ms=[(b - a) / 1e6 for a, b in zip(times, times[1:])]))
        elapsed = (end - start) / 1e6
        clean = (engine.cache.free_count == 128 and not runner._prefixes
                 and not runner.cache.linear_states and deferred.pending is None)
        return dict(order='before_finish' if during else 'after_finish', elapsed_ms=elapsed,
            output_tokens=sum(len(r['output_ids']) for r in requests),
            output_tokens_per_s=sum(len(r['output_ids']) for r in requests) * 1000 / elapsed,
            tokenize_ms=sum(r['tokenize_ms'] for r in requests), requests=requests,
            plans=engine.plans, free_slots=engine.cache.free_count, clean=clean)
    finally:
        if deferred.pending is not None:
            engine.abort_step()
        runner.synchronize()


def run(model_directory, device, repeats=3):
    import torch
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM

    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        model = ModelRunner.from_pretrained(model_directory, device, torch.float32, 128).model
        model_name = Path(model_directory).name
    else:
        tokenizer = TinyTokenizer()
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=128, partial_rotary_factor=0.5)
        model = Qwen3_5ForCausalLM(config).to(select_device(device)).eval()
        model_name = 'tiny-random-Qwen3.5'
    warmups = [run_trial(model, tokenizer, during) for during in (False, True)]
    trials = []
    for pair in range(repeats):
        for during in ((False, True) if pair % 2 == 0 else (True, False)):
            result = run_trial(model, tokenizer, during)
            result['pair'] = pair
            trials.append(result)
            print(f"第 {pair + 1} 组 {result['order']}：{result['elapsed_ms']:.3f} ms，"
                  f"分词 {result['tokenize_ms']:.3f} ms")
    def signature(trial):
        return (trial['plans'], [(r['prompt_ids'], r['output_ids'], r['finish_reason'])
                                for r in trial['requests']])
    all_runs = warmups + trials
    same = all(signature(trial) == signature(all_runs[0]) for trial in all_runs)
    summary = {}
    for order in ('after_finish', 'before_finish'):
        group = [r for r in trials if r['order'] == order]
        summary[order] = dict(median_elapsed_ms=median(r['elapsed_ms'] for r in group),
            median_output_tokens_per_s=median(r['output_tokens_per_s'] for r in group),
            median_tokenize_ms=median(r['tokenize_ms'] for r in group),
            median_mean_ttft_ms=median(sum(q['ttft_ms'] for q in r['requests']) / 4 for r in group),
            median_mean_tpot_ms=median(sum(q['tpot_ms'] for q in r['requests']) / 4 for r in group),
            median_mean_e2e_ms=median(sum(q['e2e_ms'] for q in r['requests']) / 4 for r in group))
    report = dict(model=model_name, device=str(next(model.parameters()).device), dtype='float32',
        python=platform.python_version(), torch=torch.__version__,
        scope='offline raw-text arrivals; same single-flight runner; cache allocation and warmup excluded',
        latency_origin='start of each tokenizer.encode; end at CPU token commit; no network',
        max_num_seqs=2, token_budget=4, cache_capacity=128, max_new_tokens=4,
        eos_disabled=True, warmup_trials=2, repeats=repeats, summary=summary, trials=trials,
        same_plans_and_outputs=same, failure_count=0,
        passed=same and all(r['clean'] and r['output_tokens'] == 16 for r in all_runs))
    if next(model.parameters()).device.type == 'npu':
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    if model_directory:
        report['checkpoint_config_sha256'] = hashlib.sha256(
            (Path(model_directory) / 'config.json').read_bytes()).hexdigest()
    root = Path(__file__).parent
    sources = ('benchmark_prepare.py', 'text_loop.py', 'batch_engine.py', 'deferred_runner.py',
               'model_runner.py', 'engine.py', 'hybrid_cache.py', 'tensor_cache.py',
               'kv_cache.py', 'npu_events.py', 'models/qwen3_5.py', 'models/qwen2.py')
    report['source_sha256'] = {'minideepinfer/' + name: hashlib.sha256(
        (root / name).read_bytes()).hexdigest() for name in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='local Qwen3.5 weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('outputs/prepare-comparison.json'))
    args = parser.parse_args()
    report = run(args.model, args.device, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"相同批次与输出={report['same_plans_and_outputs']}；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
