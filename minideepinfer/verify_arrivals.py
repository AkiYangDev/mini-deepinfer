"""Check mixed device-token decode and callback arrivals with safe slot reuse."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform


def run(model_directory, device):
    import torch
    from .engine import Step, Status
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from .step_state import batch_remap
    from .stopping_pipeline import StoppingDecodePipeline

    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        a, b = (tuple(tokenizer.encode(text)) for text in ('The capital of France is', 'A KV cache'))
        if (len(a), len(b)) != (5, 3):
            raise ValueError('this controlled example requires prompt lengths 5/3')
        candidate = ModelRunner.from_pretrained(model_directory, device, torch.float32, 16)
        name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        candidate = ModelRunner(Qwen3_5ForCausalLM(config).to(select_device(device)).eval(), 16)
        a, b, name = (2, 3, 4, 5, 6), (7, 8, 9), 'tiny-random-Qwen3.5'
    reference, expected = {}, {}
    # Complete AB first; compare C with a separate cold cache, not A's old state.
    for prompts, count, slots in (({'A': a, 'B': b}, 4, {'A': tuple(range(8)), 'B': tuple(range(8, 14))}),
                                  ({'C': a}, 2, {'C': tuple(range(6))})):
        runner = ModelRunner(candidate.model, 16)
        for rid in prompts:
            reference[rid], expected[rid] = [], []
        try:
            for index in range(count):
                plans = []
                for rid, prompt in prompts.items():
                    ids = prompt if index == 0 else (expected[rid][-1],)
                    start = 0 if index == 0 else len(prompt) + index - 1
                    end = start + len(ids)
                    plans.append(Step(rid, 'prefill' if index == 0 else 'decode', ids,
                                      tuple(range(start, end)), slots[rid][start:end], slots[rid][:end]))
                for step, logits in zip(plans, runner.forward_batch(tuple(plans))):
                    token = int(logits[-1].argmax())
                    expected[step.request_id].append(token)
                    reference[step.request_id].append((step, token, logits.float().cpu()))
        finally:
            runner.synchronize()
            for rid in prompts:
                runner.release(rid)
    pipeline = StoppingDecodePipeline(candidate)
    def clean():
        return (not pipeline._active and not pipeline._states and not pipeline._waiting and not pipeline._issued
                and not pipeline._slots and not pipeline._retired and not candidate._prefixes
                and not candidate.cache.linear_states and pipeline.pool.free_count == 16)
    plans, checks, arrivals, mixed, release_checks, call_order = [], [], [], [], [], []
    old_slots = ()
    forward, release = candidate.execute_main_model, candidate.release
    def capture(inputs):
        nonlocal old_slots
        call_order.append('forward')
        plans.append([(s.request_id, s.phase) for s in inputs.steps])
        if not old_slots:
            old_slots = pipeline._slots['A']
        if any(s.request_id == 'C' and s.phase == 'prefill' for s in inputs.steps):
            previous, current = pipeline._states
            mixed.append(dict(remap=batch_remap(previous.request_ids, current.request_ids),
                slots=list(pipeline._slots['C']), old_a_slots=list(old_slots),
                reused_a_slots=set(pipeline._slots['C']).issubset(old_slots),
                a_prefix_gone='A' not in candidate._prefixes,
                fresh_linear_state=not any(rid in ('A', 'C') for rid, layer in candidate.cache.linear_states)))
        return forward(inputs)
    def released(rid):
        event = pipeline._retired.get(rid)
        release_checks.append(dict(request_id=rid, last_use_ready=event is not None and event.query()))
        release(rid)
    def observe(index, steps, tokens, logits):
        call_order.append('result')
        if index == 0:
            new = pipeline.submit('C', a, 2)
            arrivals.append(dict(index=index, waiting=new.status == Status.WAITING,
                                 no_slots='C' not in pipeline._slots))
        if index == 1:
            arrivals.append(dict(index=index, waiting=pipeline.requests['C'].status == Status.WAITING,
                                 no_slots='C' not in pipeline._slots))
        for step, visible, row in zip(steps, tokens, logits):
            local = 0 if step.phase == 'prefill' else step.positions[0] - len(pipeline.requests[step.request_id].prompt_ids) + 1
            wanted_step, wanted_token, wanted = reference[step.request_id][local]
            actual = row.float().cpu()
            checks.append(dict(round=index, request_id=step.request_id, local_round=local, visible_token=visible,
                max_abs=float((actual - wanted).abs().max()), passed=bool(torch.allclose(actual, wanted, atol=1e-4, rtol=2e-4)),
                greedy_matches=int(actual[-1].argmax()) == wanted_token,
                metadata_matches=replace(step, input_ids=wanted_step.input_ids) == wanted_step))
    candidate.execute_main_model, candidate.release = capture, released
    try:
        result = pipeline.run({'A': a, 'B': b}, 4, eos_token_ids={'A': expected['A'][0]}, on_result=observe)
    finally:
        candidate.execute_main_model, candidate.release = forward, release
    mixed_clean = clean()
    output_match = (result['A'].output_ids == expected['A'][:1] and result['B'].output_ids == expected['B']
                    and result['C'].output_ids == expected['C']
                    and [r.finish_reason for r in result.values()] == ['eos', 'length', 'length'])
    def tail(index, steps, tokens, logits):
        if index == 0:
            pipeline.submit('E', a, 2)
    tail_result = pipeline.run({'D': b}, 1, max_num_seqs=1, on_result=tail)
    tail_match = (tail_result['D'].output_ids == expected['B'][:1]
                  and tail_result['E'].output_ids == expected['C'] and clean())
    error_waiting, injected = False, False
    def fail(*args):
        nonlocal error_waiting
        pipeline.submit('C', a, 2)
        error_waiting = (pipeline.requests['C'].status == Status.WAITING and 'C' not in pipeline._slots
                         and len(pipeline._states) == 2)
        raise RuntimeError('injected callback error with a waiting arrival')
    try:
        pipeline.run({'A': a, 'B': b}, 4, on_result=fail)
    except RuntimeError as error:
        if str(error) != 'injected callback error with a waiting arrival':
            raise
        injected = True
    error_clean = injected and error_waiting and clean() and all(r.status == Status.FAILED for r in pipeline.requests.values())
    reused = pipeline.run({'B': b}, 1)
    reuse_match = clean() and reused['B'].output_ids == expected['B'][:1]
    report = dict(model=name, device=str(candidate.device), dtype='float32', python=platform.python_version(),
        torch=torch.__version__, scope='single-threaded callback arrivals, FIFO full-footprint admission; no performance measurement',
        eos_scope='A uses an observed first token as controlled EOS, not natural text termination',
        reference='AB synchronous four rounds, C separate cold-cache two rounds; shared read-only weights',
        prompt_ids={'A_C_E': a, 'B_D': b}, plans=plans, comparisons=checks, call_order=call_order,
        arrivals=arrivals, mixed_inputs=mixed, release_checks=release_checks,
        requests={rid: asdict(r) for rid, r in result.items()}, outputs_match=output_match, mixed_clean=mixed_clean,
        tail_arrival_matches=tail_match, waiting_error_held=error_waiting, host_error_cleaned=error_clean,
        reuse_matches=reuse_match, free_slots=pipeline.pool.free_count, remaining_prefixes=len(candidate._prefixes),
        remaining_linear_states=len(candidate.cache.linear_states), atol=1e-4, rtol=2e-4)
    report['passed'] = (output_match and mixed_clean and tail_match and error_clean and reuse_match
        and plans == [[('A', 'prefill'), ('B', 'prefill')], [('A', 'decode'), ('B', 'decode')],
                     [('B', 'decode')], [('B', 'decode'), ('C', 'prefill')], [('C', 'decode')]]
        and len(checks) == 8 and all(c['passed'] and c['greedy_matches'] and c['metadata_matches'] for c in checks)
        and len(mixed) == 1 and mixed[0]['remap'] == (0, -1)
        and all(mixed[0][k] for k in ('reused_a_slots', 'a_prefix_gone', 'fresh_linear_state'))
        and all(a['waiting'] and a['no_slots'] for a in arrivals)
        and {r['request_id'] for r in release_checks} == {'A', 'B', 'C'}
        and all(r['last_use_ready'] for r in release_checks))
    if candidate.device.type == 'npu':
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    if model_directory:
        report['checkpoint_config_sha256'] = hashlib.sha256((Path(model_directory) / 'config.json').read_bytes()).hexdigest()
    root = Path(__file__).parent
    sources = ('verify_arrivals.py', 'stopping_pipeline.py', 'decode_pipeline.py', 'model_runner.py', 'engine.py',
               'step_state.py', 'npu_events.py', 'deferred_runner.py', 'kv_cache.py', 'hybrid_cache.py',
               'tensor_cache.py', 'models/qwen3_5.py', 'models/qwen2.py')
    report['source_sha256'] = {'minideepinfer/' + name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='local Qwen3.5 weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/arrivals.json'))
    args = parser.parse_args()
    report = run(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    print(f"新请求加入：logits 最大差异={max(c['max_abs'] for c in report['comparisons'])}；"
          f"槽位复用={report['mixed_inputs'][0]['reused_a_slots']}；缓存归还={report['free_slots']}/16；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
