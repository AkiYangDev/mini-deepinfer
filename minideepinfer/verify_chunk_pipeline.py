"""Compare budgeted cross-round prompt chunks with full-prompt greedy execution."""

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
    from .stopping_pipeline import StoppingDecodePipeline

    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        a, b = (tuple(tokenizer.encode(text)) for text in ('The capital of France is', 'What is 1 plus 1?'))
        if (len(a), len(b)) != (5, 8):
            raise ValueError('this fixed example requires prompt lengths 5/8')
        candidate = ModelRunner.from_pretrained(model_directory, device, torch.float32, 24)
        name = Path(model_directory).name
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        candidate = ModelRunner(Qwen3_5ForCausalLM(config).to(select_device(device)).eval(), 24)
        a, b, name = tuple(range(2, 7)), tuple(range(7, 15)), 'tiny-random-Qwen3.5'
    refs, expected = {}, {}
    # Build reference Steps directly, without the shared planner or chunk path.
    for prompts, count, slots in (({'A': a, 'B': b}, 4, {'A': tuple(range(8)), 'B': tuple(range(8, 19))}),
                                  ({'C': a}, 2, {'C': tuple(range(6))})):
        runner = ModelRunner(candidate.model, 24)
        for rid in prompts:
            refs[rid], expected[rid] = [], []
        try:
            for index in range(count):
                plans = []
                for rid, prompt in prompts.items():
                    ids = prompt if index == 0 else (expected[rid][-1],)
                    start = 0 if index == 0 else len(prompt) + index - 1
                    end = start + len(ids)
                    plans.append(Step(rid, 'prefill' if index == 0 else 'decode', ids,
                                      tuple(range(start, end)), slots[rid][start:end], slots[rid][:end]))
                for step, row in zip(plans, runner.forward_batch(tuple(plans))):
                    token = int(row[-1].argmax())
                    expected[step.request_id].append(token)
                    refs[step.request_id].append((step, token, row.float().cpu()))
        finally:
            runner.synchronize()
            for rid in prompts:
                runner.release(rid)
    pipeline = StoppingDecodePipeline(candidate)
    def clean():
        return (not pipeline._active and not pipeline._states and not pipeline._slots and not pipeline._retired
                and not pipeline._issued and not pipeline._last_use and not pipeline._waiting
                and not candidate._prefixes and not candidate.cache.linear_states and pipeline.pool.free_count == 24)
    plans, checks, sentinels, clocks, sample_guards, reuse, releases = [], [], [], [], [], [], []
    original, original_release = candidate.execute_main_model, candidate.release
    old_slots = ()
    def forward(inputs):
        nonlocal old_slots
        state = pipeline._states[-1]
        plans.append([asdict(step) for step in inputs.steps])
        sample_guards.append(dict(round=state.index, sample_rows=state.sample_rows,
            indices_prepared_before_forward=(not state.sample_rows or len(state.sample_rows) == len(state.steps)
                                             or state.sample_indices is not None)))
        if not old_slots:
            old_slots = pipeline._slots['A']
        if any(s.request_id == 'C' and s.positions[0] == 0 for s in inputs.steps):
            reuse.append(dict(slots=list(pipeline._slots['C']), reused_a_slots=set(pipeline._slots['C']).issubset(old_slots),
                              fresh_state=not any(rid in ('A', 'C') for rid, layer in candidate.cache.linear_states),
                              a_prefix_gone='A' not in candidate._prefixes))
        return original(inputs)
    def release(rid):
        event = pipeline._retired.get(rid)
        releases.append(dict(request_id=rid, last_use_ready=event is not None and event.query()))
        original_release(rid)
    def observe(index, steps, tokens, logits):
        if index == 0:
            clocks.append(dict(cpu_cached=pipeline.requests['A'].num_cached_tokens,
                               cpu_outputs=len(pipeline.requests['A'].output_ids),
                               issued=asdict(pipeline._issued['A'])))
            pipeline.submit('C', a, 2)
        raw_tokens = pipeline._states[0].host_tokens.tolist()  # This event was synchronized.
        for step, token, raw, row in zip(steps, tokens, raw_tokens, logits):
            prompt = pipeline.requests[step.request_id].prompt_ids
            local = 0 if step.phase == 'prefill' else step.positions[0] - len(prompt) + 1
            full_step, wanted_token, wanted = refs[step.request_id][local]
            if step.phase == 'prefill':
                start, end = step.positions[0], step.positions[-1] + 1
                wanted = wanted[start:end]
                wanted_step = Step(step.request_id, 'prefill', prompt[start:end], tuple(range(start, end)),
                                   full_step.slot_mapping[start:end], full_step.context_slots[:end], end == len(prompt))
            else:
                wanted_step = full_step
            actual = row.float().cpu()
            checks.append(dict(round=index, request_id=step.request_id, positions=step.positions,
                partial=not step.is_last_prompt_chunk, max_abs=float((actual - wanted).abs().max()),
                passed=bool(torch.allclose(actual, wanted, atol=1e-4, rtol=2e-4)),
                metadata_matches=replace(step, input_ids=wanted_step.input_ids) == wanted_step,
                sample_matches=token == wanted_token if step.is_last_prompt_chunk else token is None))
            if not step.is_last_prompt_chunk:
                sentinels.append(dict(round=index, request_id=step.request_id, host_token=raw, visible_token=token))
    candidate.execute_main_model, candidate.release = forward, release
    try:
        requests = pipeline.run({'A': a, 'B': b}, 4, max_num_batched_tokens=4, chunked_prefill=True, on_result=observe)
    finally:
        candidate.execute_main_model, candidate.release = original, original_release
    normal_clean = clean()
    output_match = all(r.output_ids == expected[rid] and r.finish_reason == 'length' for rid, r in requests.items())
    held, injected = False, False
    def fail(*args):
        nonlocal held
        pipeline.cancel('A')
        held = (len(pipeline._states) == 2 and 'A' in pipeline._last_use
                and not pipeline._retired['A'].query() and 'A' in pipeline._slots)
        raise RuntimeError('injected host error while prompt chunks are pending')
    try:
        pipeline.run({'A': a, 'B': b}, 4, max_num_batched_tokens=4, chunked_prefill=True, on_result=fail)
    except RuntimeError as error:
        if str(error) != 'injected host error while prompt chunks are pending':
            raise
        injected = True
    error_clean = injected and held and clean() and pipeline.requests['A'].status == Status.CANCELLED
    reused = pipeline.run({'A': a}, 1)
    reuse_match = clean() and reused['A'].output_ids == expected['A'][:1]
    report = dict(model=name, device=str(candidate.device), dtype='float32', python=platform.python_version(),
        torch=torch.__version__, scope='decode-first budget four, chunked prompts, callback arrival; no performance measurement',
        reference='full-prompt AB four rounds and cold C two rounds, direct Steps without shared planner, same read-only weights',
        prompt_ids={'A_C': a, 'B': b}, max_num_batched_tokens=4, plans=plans, comparisons=checks,
        partial_sentinels=sentinels, progress_clocks=clocks, sampler_preparation=sample_guards,
        slot_reuse=reuse, releases=releases, requests={rid: asdict(r) for rid, r in requests.items()},
        outputs_match=output_match, normal_clean=normal_clean, error_held_before_drain=held,
        host_error_cleaned=error_clean, reuse_matches=reuse_match, free_slots=pipeline.pool.free_count,
        remaining_prefixes=len(candidate._prefixes), remaining_linear_states=len(candidate.cache.linear_states),
        atol=1e-4, rtol=2e-4)
    report['passed'] = (normal_clean and output_match and error_clean and reuse_match and len(plans) == 9
        and len(checks) == 14 and all(c['passed'] and c['sample_matches'] and c['metadata_matches'] for c in checks)
        and all(sum(len(s['input_ids']) for s in batch) <= 4 for batch in plans)
        and len(sentinels) == 4 and all(s['host_token'] == -1 and s['visible_token'] is None for s in sentinels)
        and all(g['indices_prepared_before_forward'] for g in sample_guards)
        and clocks == [dict(cpu_cached=0, cpu_outputs=0, issued=dict(cached_tokens=5, sampled_tokens=1))]
        and len(reuse) == 1 and all(reuse[0][k] for k in ('reused_a_slots', 'fresh_state', 'a_prefix_gone'))
        and all(r['last_use_ready'] for r in releases))
    if candidate.device.type == 'npu':
        import torch_npu
        report['torch_npu'] = torch_npu.__version__
    if model_directory:
        report['checkpoint_config_sha256'] = hashlib.sha256((Path(model_directory) / 'config.json').read_bytes()).hexdigest()
    root = Path(__file__).parent
    sources = ('verify_chunk_pipeline.py', 'planning.py', 'batch_engine.py', 'stopping_pipeline.py', 'decode_pipeline.py',
               'model_runner.py', 'engine.py', 'step_state.py', 'npu_events.py', 'deferred_runner.py', 'kv_cache.py',
               'hybrid_cache.py', 'tensor_cache.py', 'models/qwen3_5.py', 'models/qwen2.py')
    report['source_sha256'] = {'minideepinfer/' + n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='local Qwen3.5 weights; omit for a tiny random model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/chunk-pipeline.json'))
    args = parser.parse_args()
    report = run(args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    print(f"分块跨轮执行：logits 最大差异={max(c['max_abs'] for c in report['comparisons'])}；"
          f"中间块数={len(report['partial_sentinels'])}；缓存归还={report['free_slots']}/24；通过={report['passed']}")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
