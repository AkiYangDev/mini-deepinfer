"""Compare synchronous and deferred Qwen3.5 execution on the same fixed steps."""

import argparse
import hashlib
import json
from pathlib import Path
import platform


def run(model_directory, device):
    import torch
    from .deferred_runner import DeferredRunner
    from .engine import Step
    from .kv_cache import SlotPool
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from .step_state import DeferredRelease

    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        a_ids = tokenizer.encode('The capital of France is')
        b_ids = tokenizer.encode('A KV cache')
        forced = tokenizer.encode(' Paris. The answer is clear.', add_special_tokens=False)[:4]
        if len(a_ids) != 5 or len(b_ids) != 3 or len(forced) != 4:
            raise ValueError('this fixed example requires prompt lengths 5/3 and four continuation IDs')
        candidate = ModelRunner.from_pretrained(model_directory, device, torch.float32, 32)
        model_name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        candidate = ModelRunner(Qwen3_5ForCausalLM(config).to(select_device(device)).eval(), 32)
        a_ids, b_ids, forced = [2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13]
        model_name = 'tiny-random-Qwen3.5'
    if not hasattr(candidate.cache, 'linear_states'):
        raise ValueError('this lesson checks Qwen3.5 hybrid state')
    baseline = ModelRunner(candidate.model, 32)  # Same weights; independent caches.
    deferred = DeferredRunner(candidate)
    pool = SlotPool(32)
    slots = {'A': pool.reserve('A', 9), 'B': pool.reserve('B', 7)}
    cleanup = DeferredRelease(pool, deferred.release)

    def plan(rid, phase, tokens, start, last=True):
        end = start + len(tokens)
        return Step(rid, phase, tuple(tokens), tuple(range(start, end)),
                    slots[rid][start:end], slots[rid][:end], last)

    rows, reuse = [], False
    try:
        for iteration in range(6):
            if iteration == 0:
                steps = (plan('A', 'prefill', a_ids[:2], 0, False),
                         plan('B', 'prefill', b_ids[:2], 0, False))
            elif iteration == 1:
                steps = (plan('B', 'prefill', b_ids[2:], 2),
                         plan('A', 'prefill', a_ids[2:4], 2, False))
            elif iteration == 2:
                steps = (plan('B', 'decode', forced[:1], 3), plan('A', 'prefill', a_ids[4:], 4))
            elif iteration == 3:
                steps = (plan('A', 'decode', forced[:1], 5), plan('B', 'decode', forced[1:2], 4))
            elif iteration == 4:
                slots['C'] = pool.reserve('C', 5)
                reuse = set(slots['C']).issubset(slots['A'])
                steps = (plan('B', 'decode', forced[2:3], 5), plan('C', 'prefill', b_ids[:2], 0, False))
            else:
                steps = (plan('C', 'prefill', b_ids[2:], 2), plan('B', 'decode', forced[3:], 6))
            reference = baseline.forward_batch(steps)
            expected = tuple(baseline.execute_sample(row) if step.is_last_prompt_chunk else None
                             for step, row in zip(steps, reference))
            before = dict(candidate._prefixes)
            pending = deferred.launch(steps)
            cleanup.retain(pending)
            unchanged = before == candidate._prefixes
            early_release = ()
            if iteration == 3:
                pending.cancel('A')
                cleanup.retire('A')
                early_release = cleanup.collect()
            pending.wait()
            checks = []
            for actual, wanted in zip(pending.logits, reference):
                actual, wanted = actual.float().cpu(), wanted.float().cpu()
                checks.append(dict(max_abs=float((actual - wanted).abs().max()),
                    passed=bool(torch.allclose(actual, wanted, atol=1e-4, rtol=2e-4))))
            outputs = pending.result()
            if iteration == 3:
                expected = (None, expected[1])
            reclaimed = cleanup.collect()
            rows.append(dict(iteration=iteration, request_ids=list(pending.request_ids),
                phases=[s.phase for s in steps], input_lengths=[len(s.input_ids) for s in steps],
                final_prompt_chunks=[s.is_last_prompt_chunk for s in steps],
                prefix_commit_deferred=unchanged, outputs=list(outputs), expected=list(expected),
                cancelled=sorted(pending.cancelled), release_before_result=list(early_release),
                reclaimed=list(reclaimed), comparisons=checks))
            print(f"第 {iteration + 1} 轮 {list(pending.request_ids)}：输出={list(outputs)}；提交前进度不变={unchanged}")
            if iteration == 3:
                baseline.release('A')
        for rid in ('B', 'C'):
            cleanup.retire(rid)
            baseline.release(rid)
        cleanup.collect()

        # A controlled host exception after real forward submission, not a
        # simulated device failure. Drain first, then reclaim hybrid state.
        error_slots = pool.reserve('error', 3)
        error_step = Step('error', 'prefill', tuple(a_ids[:2]), (0, 1),
                          error_slots[:2], error_slots[:2], False)
        original = candidate.execute_main_model
        def fail_after_forward(inputs):
            original(inputs)
            raise RuntimeError('injected host error after forward submission')
        candidate.execute_main_model = fail_after_forward
        injected_error = None
        try:
            deferred.launch((error_step,))
        except RuntimeError as error:
            if str(error) != 'injected host error after forward submission':
                raise
            injected_error = str(error)
        finally:
            candidate.execute_main_model = original
        error_drained = (injected_error is not None and deferred._pending is None
                         and not deferred._poisoned and not candidate.cache.linear_states
                         and not candidate._prefixes)
        if error_drained:
            pool.release('error')
        report = dict(model=model_name, device=str(candidate.device), dtype='float32',
            python=platform.python_version(), torch=torch.__version__,
            scope='same weights, fixed steps, independent synchronous/deferred caches; no overlap benchmark',
            prompt_ids={'A': a_ids, 'B': b_ids}, forced_ids=forced, steps=rows,
            cancelled_slots_reused=reuse, host_error_drained=error_drained,
            free_slots=pool.free_count, remaining_prefixes=len(candidate._prefixes),
            remaining_linear_states=len(candidate.cache.linear_states), atol=1e-4, rtol=2e-4)
        if candidate.device.type == 'npu':
            import torch_npu
            report['torch_npu'] = torch_npu.__version__
        if model_directory:
            report['checkpoint_config_sha256'] = hashlib.sha256(
                (Path(model_directory) / 'config.json').read_bytes()).hexdigest()
        sources = ('deferred_runner.py', 'verify_deferred.py', 'model_runner.py', 'npu_events.py',
                   'step_state.py', 'hybrid_cache.py', 'tensor_cache.py', 'kv_cache.py',
                   'models/qwen3_5.py', 'models/qwen2.py')
        root = Path(__file__).parent
        report['source_sha256'] = {'minideepinfer/' + name: hashlib.sha256(
            (root / name).read_bytes()).hexdigest() for name in sources}
        report['passed'] = (all(row['prefix_commit_deferred'] and row['outputs'] == row['expected']
            and not row['release_before_result'] and all(c['passed'] for c in row['comparisons'])
            for row in rows) and rows[3]['reclaimed'] == ['A'] and reuse and error_drained
            and pool.free_count == 32 and not candidate._prefixes and not candidate.cache.linear_states)
        return report
    finally:
        if deferred._pending is not None:
            deferred._pending.close()
        candidate.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='local Qwen3.5 weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/deferred-runner.json'))
    args = parser.parse_args()
    report = run(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    maximum = max(c['max_abs'] for row in report['steps'] for c in row['comparisons'])
    print(f"最大 logits 差异={maximum}；取消后的槽位已复用={report['cancelled_slots_reused']}")
    print(f"异常后已等待并清理={report['host_error_drained']}；槽位回收={report['free_slots']}/32；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
