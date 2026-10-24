"""Follow A, B and C through the existing pipeline using a tiny CPU model."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def run():
    import torch
    from .engine import Engine
    from .model_runner import ModelRunner
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from .stopping_pipeline import StoppingDecodePipeline

    torch.set_num_threads(1)
    torch.manual_seed(2036)
    config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                          ('linear_attention',) * 3 + ('full_attention',),
                          max_position_embeddings=64, partial_rotary_factor=0.5)
    model = Qwen3_5ForCausalLM(config).eval()
    prompts = {'A': tuple(range(2, 11)), 'B': (11, 12), 'C': (13, 14, 15, 16)}
    counts = {'A': 6, 'B': 2, 'C': 3}
    pipeline = StoppingDecodePipeline(ModelRunner(model, 20))
    observations, previous_owners, slot_history = [], {}, {}

    def observe(index, steps, tokens, logits):
        nonlocal previous_owners
        arrived = []
        if index in (0, 1):
            rid = 'B' if index == 0 else 'C'
            pipeline.submit(rid, prompts[rid], counts[rid])
            arrived.append(rid)
        owners = dict(pipeline._slots)
        reused = []
        for rid, slots in owners.items():
            for slot in slots:
                old = slot_history.get(slot)
                if old is not None and old != rid:
                    reused.append(dict(slot=slot, previous=old, current=rid))
                slot_history[slot] = rid
        observations.append(dict(
            round=index, issued_round=pipeline._states[-1].index, arrived=arrived,
            steps=[dict(request_id=s.request_id, phase=s.phase, positions=s.positions,
                        partial=not s.is_last_prompt_chunk, sampled_token=t)
                   for s, t in zip(steps, tokens)],
            progress={rid: dict(status=r.status.value, committed=r.num_cached_tokens,
                                issued=pipeline._issued[rid].cached_tokens if rid in pipeline._issued else 0,
                                slots=len(owners.get(rid, ())))
                      for rid, r in pipeline.requests.items()},
            owners=owners, released=[rid for rid in previous_owners if rid not in owners],
            reused=reused, free_slots=pipeline.pool.free_count))
        previous_owners = owners

    result = pipeline.run({'A': prompts['A']}, counts['A'], on_result=observe,
                          max_num_seqs=2, max_num_batched_tokens=4, chunked_prefill=True)
    clean = (pipeline.pool.free_count == 20 and not pipeline._states and not pipeline._slots
             and not pipeline.runner._prefixes and not pipeline.runner.cache.linear_states)
    # The original one-request Engine uses whole prompts and an independent cache.
    reference = {}
    for rid in prompts:
        engine = Engine(ModelRunner(model, 20), 20)
        request = engine.submit(prompts[rid], counts[rid])
        list(engine.run())
        reference[rid] = request.output_ids
    matches = all(r.output_ids == reference[rid] and r.finish_reason == 'length'
                  for rid, r in result.items())
    return dict(model='tiny-random-Qwen3.5', device='cpu', seed=2036, capacity=20,
                budget=4, max_num_seqs=2, observations=observations,
                requests={rid: asdict(r) for rid, r in result.items()},
                reference_outputs=reference, outputs_match=matches, clean=clean,
                drained_released=list(previous_owners), free_slots=pipeline.pool.free_count,
                passed=clean and matches)


def print_report(report):
    print('A: prompt=9, output=6; B: prompt=2, output=2; C: prompt=4, output=3')
    print('CPU random model; EOS disabled. B arrives at collect 0, C at collect 1.')
    print('Snapshots are BEFORE CPU commit; issued means submitted, not device-complete.\n')
    for record in report['observations']:
        print(f"collect {record['round']} | issued through {record['issued_round']}")
        for step in record['steps']:
            start, end = step['positions'][0], step['positions'][-1]
            action = 'partial prompt; no sample' if step['partial'] else 'sample available'
            print(f"  {step['request_id']} {step['phase']} {start}..{end}: {action}")
        if record['arrived']:
            print('  arrived: ' + ', '.join(record['arrived']))
        for rid, p in record['progress'].items():
            print(f"  {rid}: {p['status']}; cached CPU/issued={p['committed']}/{p['issued']}; slots={p['slots']}")
        if record['released']:
            print('  observed released: ' + ', '.join(record['released']))
        if record['reused']:
            by_owner = {}
            for item in record['reused']:
                by_owner.setdefault((item['previous'], item['current']), []).append(item['slot'])
            for (old, new), slots in by_owner.items():
                print(f'  reused {old} -> {new}: slots {slots}')
        print(f"  free slots: {record['free_slots']}/{report['capacity']}\n")
    print('drained | observed released: ' + ', '.join(report['drained_released']))
    for rid, request in report['requests'].items():
        print(f"  {rid}: {request['finish_reason']}; {len(request['output_ids'])} output tokens")
    print(f"whole-prompt reference outputs match: {report['outputs_match']}")
    print(f"free slots: {report['free_slots']}/{report['capacity']}; clean: {report['clean']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional JSON observation file')
    args = parser.parse_args(argv)
    report = run()
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
