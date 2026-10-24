"""Compare the generation frontend's sync, eager-pipeline, and staggered paths."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform


def compare(reference, candidate, model):
    import torch
    from .engine import Step
    from .model_runner import ModelRunner

    refs = {}
    for steps, packed in reference['captures']:
        for step, row in zip(steps, packed.split([len(s.input_ids) for s in steps])):
            refs[(step.request_id, step.positions[0])] = row
    comparisons, extra_forwards = [], 0
    for index, (steps, packed) in enumerate(candidate['captures']):
        for step, row in zip(steps, packed.split([len(s.input_ids) for s in steps])):
            key = (step.request_id, 0 if step.phase == 'prefill' else step.positions[0])
            extra = key not in refs
            if extra:
                request = reference['requests'][step.request_id]
                if (request['finish_reason'] != 'eos' or step.phase != 'decode'
                        or step.positions != (request['num_cached_tokens'],)):
                    raise ValueError('unreferenced step is not the single lookahead after EOS')
                ids = tuple(request['prompt_ids']) + tuple(request['output_ids'])
                slots = tuple(range(len(ids)))
                cold = ModelRunner(model, len(ids))
                try:
                    full = Step(step.request_id, 'prefill', ids, slots, slots, slots)
                    refs[key] = cold.forward_batch((full,))[0][-1:].detach()
                    extra_forwards += 1
                finally:
                    cold.synchronize()
                    cold.release(step.request_id)
            wanted = refs[key]
            if step.phase == 'prefill':
                wanted = wanted[step.positions[0]:step.positions[-1] + 1]
            comparisons.append(dict(round=index, request_id=step.request_id, phase=step.phase,
                                    positions=step.positions, discarded_after_eos=extra,
                                    max_abs=float((row - wanted).abs().max()),
                                    passed=bool(torch.allclose(row, wanted, atol=1e-4, rtol=2e-4))))
    matches = candidate['requests'].keys() == reference['requests'].keys() and all(
        candidate['requests'][rid]['output_ids'] == request['output_ids']
        and candidate['requests'][rid]['finish_reason'] == request['finish_reason']
        and candidate['requests'][rid]['num_cached_tokens'] == request['num_cached_tokens']
        for rid, request in reference['requests'].items())
    return comparisons, matches, extra_forwards


def run(model_directory=None, device='cpu', trace_rounds=False):
    import torch
    from .generate import parse_args, run_generation
    from .model_runner import ModelRunner, select_device
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from .pipeline_text import print_round

    torch.set_num_threads(1)
    if model_directory:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        loaded = ModelRunner.from_pretrained(model_directory, device, torch.float32, 24)
        model, name = loaded.model, Path(model_directory).name
        texts = ['The capital of France is', 'What is 1 plus 1?', 'The capital of France is']
    else:
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        model = Qwen3_5ForCausalLM(config).to(select_device(device)).eval()
        class Tokenizer:
            eos_token_id = None
            def encode(self, text):
                return list(map(int, text.split()))
        tokenizer, name = Tokenizer(), 'tiny-random-Qwen3.5'
        texts = ['2 3 4 5 6', '7 8 9 10 11 12 13 14', '2 3 4 5 6']

    def execute(texts, mode, *, raw=True, count=4, capacity=24, limit=2, budget=4):
        argv = ['--model', 'preloaded', '--device', device, '--max-new-tokens', str(count),
                '--cache-capacity', str(capacity), '--max-num-seqs', str(limit),
                '--max-batched-tokens', str(capacity if mode == 'sync' else budget)]
        if raw:
            argv.append('--raw')
        if mode != 'sync':
            argv.extend(('--pipeline', '--chunked-prefill'))
        if mode == 'staggered':
            argv.append('--stagger-prompts')
        for text in texts:
            argv.extend(('--prompt', text))
        args = parse_args(argv)
        runner = ModelRunner(model, capacity)
        captured, observations = [], []
        forward = runner.execute_main_model
        def capture(inputs):
            logits = forward(inputs)
            captured.append((inputs.steps, logits.detach()))  # Read only after the loop drains.
            return logits
        runner.execute_main_model = capture
        def observe(record):
            observations.append(record)
            if trace_rounds and mode == 'staggered':
                print_round(record)
        try:
            pairs, pool = run_generation(runner, tokenizer, args, on_round=observe)
        finally:
            runner.execute_main_model = forward
        clean = pool.free_count == capacity and not runner._prefixes and not runner.cache.linear_states
        return dict(requests={r.request_id: asdict(r) for r, _ in pairs}, captures=captured,
                    observations=observations, clean=clean, forwards=len(captured), free_slots=pool.free_count)

    reference = execute(texts, 'sync')
    cases = {}
    for mode in ('pipeline', 'staggered'):
        result = execute(texts, mode)
        comparisons, matches, extra = compare(reference, result, model)
        if extra:
            raise ValueError('raw example unexpectedly stopped at EOS')
        result.pop('captures')
        cases[mode] = dict(**result, comparisons=comparisons, outputs_match=matches)
    counts = {'sync': reference['forwards'], **{mode: r['forwards'] for mode, r in cases.items()}}
    if model_directory:
        chat = ['What is 1 plus 1? Answer in one short sentence.']
        chat_ref = execute(chat, 'sync', raw=False, count=32, capacity=256, limit=1, budget=16)
        chat_result = execute(chat, 'pipeline', raw=False, count=32, capacity=256, limit=1, budget=16)
        comparisons, matches, extra = compare(chat_ref, chat_result, model)
        counts.update(chat_sync=chat_ref['forwards'], chat_pipeline=chat_result['forwards'],
                      chat_extra_reference=extra)
        chat_result.pop('captures')
        cases['chat_eos'] = dict(**chat_result, comparisons=comparisons, outputs_match=matches,
                                decoded_text=tokenizer.decode(chat_result['requests']['req-0']['output_ids'],
                                                              skip_special_tokens=True),
                                stopped_at_eos=chat_result['requests']['req-0']['finish_reason'] == 'eos')
    report = dict(model=name, device=device, dtype='float32', python=platform.python_version(),
                  torch=torch.__version__, scope='shared CLI execution function with preloaded weights; no timing',
                  reference='sync frontend, whole prompts, independent cache; one shared read-only model',
                  texts=texts, reference_requests=reference['requests'], cases=cases, forward_calls=counts,
                  total_forward_calls=sum(counts.values()), atol=1e-4, rtol=2e-4,
                  passed=reference['clean'] and all(r['clean'] and r['outputs_match']
                         and all(c['passed'] for c in r['comparisons']) for r in cases.values()))
    if model_directory:
        if device.startswith('npu'):
            import torch_npu
            report['torch_npu'] = torch_npu.__version__
        report['passed'] = report['passed'] and cases['chat_eos']['stopped_at_eos']
        report['checkpoint_config_sha256'] = hashlib.sha256((Path(model_directory) / 'config.json').read_bytes()).hexdigest()
    sources = ['verify_pipeline_text.py', 'generate.py', 'pipeline_text.py', 'text_loop.py',
               'planning.py', 'batch_engine.py', 'stopping_pipeline.py', 'decode_pipeline.py',
               'model_runner.py', 'engine.py', 'step_state.py', 'npu_events.py', 'deferred_runner.py',
               'kv_cache.py', 'hybrid_cache.py', 'tensor_cache.py', 'models/qwen3_5.py', 'models/qwen2.py']
    report['source_sha256'] = {'minideepinfer/' + p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
                               for p in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=None, help='existing local Qwen3.5 checkpoint; otherwise use a tiny model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--trace-rounds', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/pipeline-text.json'))
    args = parser.parse_args()
    report = run(args.model, args.device, args.trace_rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    largest = max(c['max_abs'] for case in report['cases'].values() for c in case['comparisons'])
    print(f"text pipeline: max_abs={largest}; forwards={report['total_forward_calls']}; passed={report['passed']}")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
