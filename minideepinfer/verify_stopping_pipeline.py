"""Check EOS, cancellation, and last-use release in a shrinking decode batch."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform


def run(model_directory, device):
    import torch
    from .engine import Step, Status
    from .kv_cache import SlotPool
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from .stopping_pipeline import StoppingDecodePipeline

    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        prompts = {rid: tuple(tokenizer.encode(text)) for rid, text in
                   (('A', 'The capital of France is'), ('B', 'A KV cache'), ('C', 'What is 1 plus 1?'))}
        candidate = ModelRunner.from_pretrained(model_directory, device, torch.float32, 64)
        model_name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        candidate = ModelRunner(Qwen3_5ForCausalLM(config).to(select_device(device)).eval(), 64)
        prompts = {'A': (2, 3, 4), 'B': (7,), 'C': (11, 12)}
        model_name = 'tiny-random-Qwen3.5'
    baseline = ModelRunner(candidate.model, 64)
    pipeline = StoppingDecodePipeline(candidate)
    orders = (('A', 'B', 'C'), ('C', 'B', 'A'), ('A', 'B', 'C'), ('B', 'C', 'A'))
    pool = SlotPool(64)
    slots = {rid: pool.reserve(rid, len(p) + 3) for rid, p in prompts.items()}
    expected = {rid: [] for rid in prompts}
    references = []
    try:
        for index, order in enumerate(orders):
            steps = []
            for rid in order:
                ids = prompts[rid] if index == 0 else (expected[rid][-1],)
                start = 0 if index == 0 else len(prompts[rid]) + index - 1
                end = start + len(ids)
                steps.append(Step(rid, 'prefill' if index == 0 else 'decode', ids,
                                  tuple(range(start, end)), slots[rid][start:end], slots[rid][:end]))
            rows = baseline.forward_batch(tuple(steps))
            reference = {}
            for step, logits in zip(steps, rows):
                token = int(logits[-1].argmax())
                expected[step.request_id].append(token)
                reference[step.request_id] = (step, token, logits.float().cpu())
            references.append(reference)
    finally:
        baseline.synchronize()
        for rid in prompts:
            baseline.release(rid)
            pool.release(rid)

    def clean():
        return (not pipeline._active and not pipeline._states and not pipeline._slots and not pipeline._retired
                and not candidate._prefixes and not candidate.cache.linear_states and pipeline.pool.free_count == 64)

    comparisons, plans, released, guards, call_order = [], [], [], [], []
    original_forward, original_release, original_collect = candidate.execute_main_model, candidate.release, pipeline._collect
    def forward(inputs):
        call_order.append('forward')
        plans.append([s.request_id for s in inputs.steps])
        return original_forward(inputs)
    def release(rid):
        event = pipeline._retired.get(rid)
        released.append(dict(request_id=rid, last_use_recorded=event is not None and event._recorded,
                             last_use_ready=event is not None and event.query(),
                             live_rounds=[s.index for s in pipeline._states]))
        original_release(rid)
    def collect():
        unrecorded = [rid for rid, event in pipeline._retired.items() if not event._recorded]
        before = pipeline.pool.free_count
        original_collect()
        for rid in unrecorded:
            guards.append(dict(request_id=rid, slots_held=rid in pipeline._slots,
                               prefix_held=rid in candidate._prefixes, free_slots_before=before,
                               free_slots_after=pipeline.pool.free_count))
    def observe(index, steps, tokens, logits):
        call_order.append('result')
        if index == 1:
            pipeline.cancel('B')  # Keep B's first token; discard this and the submitted next result.
        for step, token, row in zip(steps, tokens, logits):
            wanted_step, wanted_token, wanted = references[index][step.request_id]
            actual = row.float().cpu()
            comparisons.append(dict(index=index, request_id=step.request_id, visible_token=token,
                expected_token=wanted_token, metadata_matches=replace(step, input_ids=wanted_step.input_ids) == wanted_step,
                max_abs=float((actual - wanted).abs().max()),
                passed=bool(torch.allclose(actual, wanted, atol=1e-4, rtol=2e-4)),
                greedy_matches=int(actual[-1].argmax()) == wanted_token))
    candidate.execute_main_model, candidate.release, pipeline._collect = forward, release, collect
    try:
        mixed = pipeline.run(prompts, 4, eos_token_ids={'A': expected['A'][0]}, orders=orders, on_result=observe)
    finally:
        candidate.execute_main_model, candidate.release, pipeline._collect = original_forward, original_release, original_collect
    mixed_clean = clean()
    mixed_expected = {'A': expected['A'][:1], 'B': expected['B'][:1], 'C': expected['C']}
    mixed_match = (all(r.output_ids == mixed_expected[rid] for rid, r in mixed.items())
        and [r.finish_reason for r in mixed.values()] == ['eos', 'cancelled', 'length']
        and [r.num_cached_tokens for r in mixed.values()] == [len(prompts['A']), len(prompts['B']), len(prompts['C']) + 3])

    all_eos_calls, discarded = [], []
    def count_forward(inputs):
        all_eos_calls.append([s.request_id for s in inputs.steps])
        return original_forward(inputs)
    def tail(index, steps, tokens, logits):
        discarded.append(list(tokens))
    candidate.execute_main_model = count_forward
    try:
        all_eos = pipeline.run(prompts, 4, eos_token_ids={rid: tokens[0] for rid, tokens in expected.items()}, on_result=tail)
    finally:
        candidate.execute_main_model = original_forward
    all_eos_match = (len(all_eos_calls) == 2 and discarded[-1] == [None] * 3 and clean()
                    and all(r.output_ids == expected[rid][:1] and r.finish_reason == 'eos' for rid, r in all_eos.items()))
    error_held, injected = False, False
    def fail(*args):
        nonlocal error_held
        pipeline.cancel('B')
        error_held = (len(pipeline._states) == 2 and not pipeline._retired['B'].query()
                      and 'B' in pipeline._slots and 'B' in candidate._prefixes)
        raise RuntimeError('injected CPU post-process error after cancellation')
    try:
        pipeline.run(prompts, 3, on_result=fail)
    except RuntimeError as error:
        if str(error) != 'injected CPU post-process error after cancellation':
            raise
        injected = True
    error_clean = (injected and error_held and clean() and pipeline.requests['B'].status == Status.CANCELLED
                   and all(pipeline.requests[rid].status == Status.FAILED for rid in ('A', 'C')))
    reused = pipeline.run(prompts, 1)
    reuse_match = clean() and all(r.output_ids == expected[rid][:1] and r.finish_reason == 'length'
                                  for rid, r in reused.items())
    report = dict(model=model_name, device=str(candidate.device), dtype='float32',
        python=platform.python_version(), torch=torch.__version__,
        scope='admit once, shrinking membership; no new arrivals, rollback, or performance measurement',
        eos_scope='A uses its observed first greedy token as controlled EOS; this is not natural text termination',
        cancellation_scope='B cancelled in round-one result callback before commit; its first token remains',
        reference='four full synchronous greedy rounds completed first, shared read-only weights and independent caches',
        prompt_ids=prompts, max_new_tokens=4, plans=plans, call_order=call_order,
        comparisons=comparisons, requests={rid: asdict(r) for rid, r in mixed.items()},
        expected_output_ids=mixed_expected, mixed_matches=mixed_match, mixed_clean=mixed_clean,
        releases=released, unrecorded_last_use_guards=guards,
        all_eos=dict(plans=all_eos_calls, visible_tokens=discarded, passed=all_eos_match),
        error_slots_held_before_drain=error_held, host_error_cleaned=error_clean,
        reuse_matches=reuse_match, free_slots=pipeline.pool.free_count,
        remaining_prefixes=len(candidate._prefixes), remaining_linear_states=len(candidate.cache.linear_states),
        atol=1e-4, rtol=2e-4)
    report['passed'] = (mixed_match and mixed_clean and all_eos_match and error_clean and reuse_match
        and plans == [['A', 'B', 'C'], ['C', 'B', 'A'], ['B', 'C'], ['C']]
        and call_order == ['forward', 'forward', 'result', 'forward', 'result', 'forward', 'result', 'result']
        and len(comparisons) == 9 and all(c['passed'] and c['greedy_matches'] and c['metadata_matches']
                                       and c['visible_token'] in (None, c['expected_token']) for c in comparisons)
        and {r['request_id'] for r in released} == set(prompts)
        and all(r['last_use_recorded'] and r['last_use_ready'] for r in released)
        and {g['request_id'] for g in guards} == {'A', 'B'}
        and all(g['slots_held'] and g['prefix_held'] for g in guards))
    if candidate.device.type == 'npu':
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    if model_directory:
        report['checkpoint_config_sha256'] = hashlib.sha256((Path(model_directory) / 'config.json').read_bytes()).hexdigest()
    root = Path(__file__).parent
    sources = ('stopping_pipeline.py', 'verify_stopping_pipeline.py', 'decode_pipeline.py', 'model_runner.py',
               'engine.py', 'step_state.py', 'npu_events.py', 'deferred_runner.py', 'kv_cache.py',
               'hybrid_cache.py', 'tensor_cache.py', 'models/qwen3_5.py', 'models/qwen2.py')
    report['source_sha256'] = {'minideepinfer/' + name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='local Qwen3.5 weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/stopping-pipeline.json'))
    args = parser.parse_args()
    report = run(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    maximum = max(c['max_abs'] for c in report['comparisons'])
    print(f"EOS/取消退出：logits 最大差异={maximum}；异常后清理={report['host_error_cleaned']}；"
          f"缓存归还={report['free_slots']}/64；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
