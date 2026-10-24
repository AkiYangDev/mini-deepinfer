"""Check fixed-batch device-token continuation against synchronous greedy decode."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform


def run(model_directory, device):
    import torch
    from .decode_pipeline import FixedDecodePipeline
    from .engine import Step
    from .kv_cache import SlotPool
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM

    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        prompts = {rid: tuple(tokenizer.encode(text)) for rid, text in
                   (('A', 'The capital of France is'), ('B', 'A KV cache'))}
        candidate = ModelRunner.from_pretrained(model_directory, device, torch.float32, 32)
        model_name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        candidate = ModelRunner(Qwen3_5ForCausalLM(config).to(select_device(device)).eval(), 32)
        prompts = {'A': (2, 3, 4, 5, 6), 'B': (7, 8, 9)}
        model_name = 'tiny-random-Qwen3.5'
    baseline = ModelRunner(candidate.model, 32)
    pipeline = FixedDecodePipeline(candidate)
    orders = (('A', 'B'), ('B', 'A'), ('A', 'B'), ('B', 'A'))
    expected = {rid: [] for rid in prompts}
    pool = SlotPool(32)
    slots = {rid: pool.reserve(rid, len(p) + 3) for rid, p in prompts.items()}
    references = []
    # Complete the reference first. Its work never enters the pipeline's gap.
    try:
        for index, order in enumerate(orders):
            steps = []
            for rid in order:
                tokens = prompts[rid] if index == 0 else (expected[rid][-1],)
                start = 0 if index == 0 else len(prompts[rid]) + index - 1
                end = start + len(tokens)
                steps.append(Step(rid, 'prefill' if index == 0 else 'decode', tokens,
                                  tuple(range(start, end)), slots[rid][start:end], slots[rid][:end]))
            logits = baseline.forward_batch(tuple(steps))
            tokens = tuple(int(row[-1].argmax()) for row in logits)
            references.append((tuple(steps), tokens, tuple(row.float().cpu() for row in logits)))
            for rid, token in zip(order, tokens):
                expected[rid].append(token)
    finally:
        baseline.synchronize()
        for rid in prompts:
            baseline.release(rid)
            pool.release(rid)

    rows, sequence = [], []
    original_forward = candidate.execute_main_model
    def forward(inputs):
        sequence.append('forward')
        return original_forward(inputs)
    def observe(index, steps, tokens, logits):
        sequence.append('result')
        plans, wanted_tokens, wanted_logits = references[index]
        same_metadata = all(replace(s, input_ids=p.input_ids) == p for s, p in zip(steps, plans))
        checks = []
        for actual, wanted in zip(logits, wanted_logits):
            actual = actual.float().cpu()
            checks.append(dict(max_abs=float((actual - wanted).abs().max()),
                               passed=bool(torch.allclose(actual, wanted, atol=1e-4, rtol=2e-4))))
        rows.append(dict(index=index, request_ids=[s.request_id for s in steps],
                         same_metadata=same_metadata, tokens=tokens, expected_tokens=wanted_tokens,
                         comparisons=checks, retained_rounds=len(pipeline._states)))
    candidate.execute_main_model = forward
    try:
        actual = pipeline.run(prompts, 4, orders=orders, on_result=observe)
    finally:
        candidate.execute_main_model = original_forward
    normal_clean = (not pipeline._active and not pipeline._states and not candidate._prefixes
                    and not candidate.cache.linear_states and pipeline.pool.free_count == 32)
    def fail_on_result(*args):
        if len(pipeline._states) != 2:
            raise AssertionError('expected a later forward to have been submitted')
        raise RuntimeError('injected CPU post-process error')
    injected = False
    try:
        pipeline.run(prompts, 3, on_result=fail_on_result)
    except RuntimeError as error:
        if str(error) != 'injected CPU post-process error':
            raise
        injected = True
    error_clean = (injected and not pipeline._active and not pipeline._states
                   and not candidate._prefixes and not candidate.cache.linear_states
                   and pipeline.pool.free_count == 32)
    reused = pipeline.run(prompts, 1)
    reuse_clean = (not candidate._prefixes and not candidate.cache.linear_states
                  and pipeline.pool.free_count == 32)
    report = dict(model=model_name, device=str(candidate.device), dtype='float32',
                  python=platform.python_version(), torch=torch.__version__,
                  scope='fixed membership and length, EOS disabled; no dynamic scheduling or performance claim',
                  reference='synchronous greedy execution, shared read-only weights, independent caches',
                  prompt_ids=prompts, max_new_tokens=4, steps=rows, call_order=sequence,
                  output_ids=actual, expected_output_ids=expected,
                  normal_clean=normal_clean, host_error_cleaned=error_clean,
                  reuse_clean=reuse_clean, reuse_output_ids=reused,
                  free_slots=pipeline.pool.free_count, remaining_prefixes=len(candidate._prefixes),
                  remaining_linear_states=len(candidate.cache.linear_states), atol=1e-4, rtol=2e-4)
    report['passed'] = (actual == expected and normal_clean and error_clean and reuse_clean
        and all(row['same_metadata'] and row['tokens'] == row['expected_tokens']
                and all(c['passed'] for c in row['comparisons']) for row in rows)
        and sequence == ['forward', 'forward', 'result', 'forward', 'result', 'forward', 'result', 'result']
        and all(reused[rid] == expected[rid][:1] for rid in prompts))
    if candidate.device.type == 'npu':
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    if model_directory:
        report['checkpoint_config_sha256'] = hashlib.sha256(
            (Path(model_directory) / 'config.json').read_bytes()).hexdigest()
    root = Path(__file__).parent
    sources = ('decode_pipeline.py', 'verify_decode_pipeline.py', 'model_runner.py', 'deferred_runner.py',
               'engine.py', 'step_state.py', 'npu_events.py', 'kv_cache.py', 'hybrid_cache.py',
               'tensor_cache.py', 'models/qwen3_5.py', 'models/qwen2.py')
    report['source_sha256'] = {'minideepinfer/' + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='local Qwen3.5 weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/decode-pipeline.json'))
    args = parser.parse_args()
    report = run(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    maximum = max(c['max_abs'] for row in report['steps'] for c in row['comparisons'])
    print(f"设备 token 接续：logits 最大差异={maximum}；异常后清理={report['host_error_cleaned']}；"
          f"缓存归还={report['free_slots']}/32；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
