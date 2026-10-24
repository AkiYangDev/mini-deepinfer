"""Compare synchronous and split-step BatchEngine request lifecycles."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform


def run(model_directory, device):
    import torch
    from .batch_engine import BatchEngine
    from .deferred_runner import DeferredRunner
    from .engine import Status
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM

    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        a_ids = tokenizer.encode('The capital of France is')
        b_ids = tokenizer.encode('A KV cache')
        candidate = ModelRunner.from_pretrained(model_directory, device, torch.float32, 32)
        model_name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        candidate = ModelRunner(Qwen3_5ForCausalLM(config).to(select_device(device)).eval(), 32)
        a_ids, b_ids = [2, 3, 4, 5, 6], [7, 8, 9]
        model_name = 'tiny-random-Qwen3.5'
    if len(a_ids) != 5 or len(b_ids) != 3:
        raise ValueError('the example requires prompt lengths 5/3')
    baseline = ModelRunner(candidate.model, 32)
    deferred = DeferredRunner(candidate)
    sync = BatchEngine(baseline, 32, 2, 4, chunked_prefill=True)
    engine = BatchEngine(deferred, 32, 2, 4, chunked_prefill=True, deferred=True)
    reference_logits, reference_plans = [], []
    original_forward, original_run = baseline.execute_main_model, baseline.run_batch

    def capture_forward(inputs):
        logits = original_forward(inputs)
        reference_logits[:] = [logits]
        return logits

    def capture_run(steps):
        reference_plans[:] = steps
        return original_run(steps)

    baseline.execute_main_model, baseline.run_batch = capture_forward, capture_run
    pairs = [(sync.submit(a_ids, 4), engine.submit(a_ids, 4)),
             (sync.submit(b_ids, 4), engine.submit(b_ids, 4))]
    rows, reused, held = [], False, False
    a_slots = ()
    try:
        for iteration in range(6):
            if iteration == 5:
                # Use B's observed first token as EOS for a separate copy of B.
                # This exercises the lifecycle without assuming a model's answer.
                eos = pairs[1][0].output_ids[0]
                pairs.append((sync.submit(b_ids, 3, eos), engine.submit(b_ids, 3, eos)))
            expected = sync.step()
            before = {r.request_id: (r.num_cached_tokens, tuple(r.output_ids)) for _, r in pairs}
            prefixes = dict(candidate._prefixes)
            plans = engine.launch_step()
            unchanged = (prefixes == candidate._prefixes and all(
                before[r.request_id] == (r.num_cached_tokens, tuple(r.output_ids)) for _, r in pairs))
            if iteration == 0:
                a_slots = engine._running[pairs[0][1].request_id]
            if iteration == 2:
                # The sync reference has completed A's last unwanted token;
                # the split path discards it before committing request progress.
                free_before = engine.cache.free_count
                sync.cancel(pairs[0][0].request_id)
                engine.cancel(pairs[0][1].request_id)
                pairs.append((sync.submit(b_ids, 2), engine.submit(b_ids, 2)))
                held = engine.cache.free_count == free_before
            if iteration == 3:
                reused = set(engine._running[pairs[2][1].request_id]).issubset(a_slots)
            pending = deferred.pending
            pending.wait()  # Diagnostic logits are readable only after the event.
            reference = reference_logits[0].split([len(p.input_ids) for p in plans])
            checks = []
            for actual, wanted in zip(pending.logits, reference):
                actual, wanted = actual.float().cpu(), wanted.float().cpu()
                checks.append(dict(max_abs=float((actual - wanted).abs().max()),
                                   passed=bool(torch.allclose(actual, wanted, atol=1e-4, rtol=2e-4))))
            outputs = engine.finish_step()
            expected_tokens = [None if iteration == 2 and o.request_id == pairs[0][0].request_id
                               else o.token_id for o in expected]
            progress_matches = all(
                (a.output_ids, a.num_cached_tokens, a.status, a.finish_reason) ==
                (b.output_ids, b.num_cached_tokens, b.status, b.finish_reason)
                for a, b in pairs if a is not pairs[0][0] or iteration < 2)
            cancelled_unchanged = (iteration != 2 or before[pairs[0][1].request_id] ==
                (pairs[0][1].num_cached_tokens, tuple(pairs[0][1].output_ids)))
            rows.append(dict(iteration=iteration, plans=[asdict(p) for p in plans],
                same_plans=tuple(reference_plans) == plans, progress_deferred=unchanged,
                outputs=[asdict(o) for o in outputs], expected_tokens=expected_tokens,
                progress_matches=progress_matches, cancelled_progress_unchanged=cancelled_unchanged,
                comparisons=checks, free_slots=engine.cache.free_count))
            print(f'第 {iteration + 1} 轮：{[p.request_id for p in plans]}；'
                  f'输出={[o.token_id for o in outputs]}；提交前进度不变={unchanged}')

        original = candidate.execute_main_model
        def fail_after_forward(inputs):
            original(inputs)
            raise RuntimeError('injected host error after forward submission')
        candidate.execute_main_model = fail_after_forward
        failed = engine.submit(b_ids, 2)
        try:
            engine.launch_step()
        except RuntimeError as error:
            if str(error) != 'injected host error after forward submission':
                raise
        finally:
            candidate.execute_main_model = original
        error_cleaned = (failed.status == Status.FAILED and not failed.output_ids
            and failed.num_cached_tokens == 0 and deferred.pending is None
            and not candidate._prefixes and not candidate.cache.linear_states
            and engine.cache.free_count == 32)
        report = dict(model=model_name, device=str(candidate.device), dtype='float32',
            python=platform.python_version(), torch=torch.__version__,
            scope='same BatchEngine scheduler, same read-only weights, independent caches; no overlap benchmark',
            cancellation_reference='sync finishes then cancels A; deferred cancels before commit and discards that row',
            prompt_ids={'A': a_ids, 'B_C_EOS': b_ids}, steps=rows,
            cancelled_slots_held=held, cancelled_slots_reused=reused,
            requests=[asdict(r) for _, r in pairs], host_error_cleaned=error_cleaned,
            free_slots=engine.cache.free_count, remaining_prefixes=len(candidate._prefixes),
            remaining_linear_states=len(candidate.cache.linear_states), atol=1e-4, rtol=2e-4)
        if candidate.device.type == 'npu':
            import torch_npu
            report['torch_npu'] = torch_npu.__version__
        if model_directory:
            report['checkpoint_config_sha256'] = hashlib.sha256(
                (Path(model_directory) / 'config.json').read_bytes()).hexdigest()
        sources = ('batch_engine.py', 'deferred_runner.py', 'verify_deferred_engine.py', 'engine.py',
                   'model_runner.py', 'npu_events.py', 'hybrid_cache.py', 'tensor_cache.py',
                   'kv_cache.py', 'models/qwen3_5.py', 'models/qwen2.py')
        root = Path(__file__).parent
        report['source_sha256'] = {'minideepinfer/' + name: hashlib.sha256(
            (root / name).read_bytes()).hexdigest() for name in sources}
        report['passed'] = (all(row['same_plans'] and row['progress_deferred'] and row['progress_matches']
            and row['cancelled_progress_unchanged'] and row['expected_tokens'] ==
            [o['token_id'] for o in row['outputs']] and all(c['passed'] for c in row['comparisons'])
            for row in rows) and held and reused and error_cleaned
            and pairs[-1][1].finish_reason == 'eos'
            and all(r.finish_reason == 'length' for _, r in pairs[1:3])
            and sync.cache.free_count == 32 and not baseline.cache.linear_states and not baseline._prefixes)
        return report
    finally:
        if engine._plans:
            engine.abort_step()
        candidate.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='local Qwen3.5 weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/deferred-engine.json'))
    args = parser.parse_args()
    report = run(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    maximum = max(c['max_abs'] for row in report['steps'] for c in row['comparisons'])
    print(f"最大 logits 差异={maximum}；取消后槽位复用={report['cancelled_slots_reused']}；"
          f"异常后清理={report['host_error_cleaned']}；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
