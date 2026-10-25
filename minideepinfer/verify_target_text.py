"""Check single-model sampled text against independent full-prefix forwards."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def check_target(target, tokenizer, report, prompt='The capital of France is'):
    from .sampling import TemperatureSampler
    from .speculative import CachedTarget
    from .target_text import run_target_text

    ids = tokenizer.encode(prompt)
    capacity = len(ids)+5
    expected, cold_work = [], dict(forward_calls=0, input_tokens=0)
    sampler = TemperatureSampler(.7, 2026, top_k=20, top_p=.9)
    for _ in range(6):
        cold = CachedTarget(target, tuple(range(capacity)), capacity)
        try:
            expected.append(sampler.sample(cold.forward(tuple(ids)+tuple(expected))[-1]))
            cold_work['forward_calls'] += cold.forward_calls
            cold_work['input_tokens'] += cold.input_tokens
        finally:
            cold.synchronize()
            if cold.hybrid:
                cold.cache.release('spec')
    # Disable natural EOS for this fixed-length numerical check. A separate
    # configured-EOS case below verifies stopping on the first sampled token.
    text_tokenizer = SimpleNamespace(encode=tokenizer.encode, decode=tokenizer.decode, eos_token_id=None)
    report.update(prompt_ids=ids, temperature=.7, seed=2026, top_k=20, top_p=.9,
                  max_new_tokens=6, cache_capacity=capacity,
                  full_prefix_reference=dict(output_ids=expected, **cold_work), cases={})
    for name, limit, eos in (('sampled', None, None), ('cancelled', 2, None),
                             ('configured_eos', None, expected[0])):
        text_tokenizer.eos_token_id = eos
        groups = []
        result = run_target_text(target, text_tokenizer, prompt, raw=True, max_new_tokens=6,
                    cache_capacity=capacity, temperature=.7, top_k=20, top_p=.9,
                    on_commit=groups.append,
                    should_cancel=lambda: limit is not None and len(groups) >= limit)
        result.update(committed_groups=[list(tokens) for tokens in groups], configured_eos_token_id=eos)
        report['cases'][name] = result
        wanted = expected if name == 'sampled' else expected[:2] if name == 'cancelled' else expected[:1]
        reason = dict(sampled='length', cancelled='cancelled', configured_eos='eos')[name]
        if (result['output_ids'] != wanted or result['finish_reason'] != reason
                or result['free_slots'] != capacity or result['proposed_draft_tokens'] != 0
                or groups != [(token,) for token in wanted]
                or result['target_forward_calls'] != len(wanted)):
            raise RuntimeError('single-model output, stopping, callback or cleanup differs in '+name)
    report['total_forward_calls'] = cold_work['forward_calls']+sum(
        case['target_forward_calls'] for case in report['cases'].values())
    report['total_input_tokens'] = cold_work['input_tokens']+sum(
        case['target_input_tokens'] for case in report['cases'].values())
    report['passed'] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='existing local checkpoint; loaded once')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import platform
    import torch
    from transformers import AutoTokenizer
    from .model_runner import ModelRunner
    torch.set_num_threads(1)
    report = dict(passed=False, device=args.device, dtype='float32', torch=torch.__version__,
                  python=platform.python_version(), cpu_threads=1, target_model=Path(args.model).name,
                  weight_loads=0, source_sha256={},
                  scope='Single-model API, filtered sampling, cold-prefix comparison and commit callbacks. '
                        'Configured EOS is artificial. No CLI text-stream, timing or speedup measurement.')
    sources = ('verify_target_text.py', 'target_text.py', 'text_loop.py', 'speculative.py', 'sampling.py',
               'model_runner.py', 'models/qwen2.py', 'models/qwen3_5.py',
               'tensor_cache.py', 'hybrid_cache.py', 'engine.py', 'kv_cache.py')
    report['source_sha256'] = {'minideepinfer/'+name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                              for name in sources}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name+'.tmp')
        temporary.write_bytes((json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
        temporary.replace(args.output)
    phase = 'loading'
    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        target = ModelRunner.from_pretrained(args.model, args.device, torch.float32, 1).model
        report['weight_loads'] += 1
        if args.device.startswith('npu'):
            import torch_npu
            report.update(torch_npu=torch_npu.__version__, npu_name=torch.npu.get_device_name(0),
                          hf32=dict(matmul=torch.npu.matmul.allow_hf32, conv=torch.npu.conv.allow_hf32))
        phase = 'numerical_checks'
        check_target(target, tokenizer, report)
    except BaseException as error:
        report['failure'] = dict(phase=phase, type=type(error).__name__, message=str(error))
        save()
        raise
    save()
    print(json.dumps(dict(passed=report['passed'], forward_calls=report['total_forward_calls'],
                          cases={name: case['finish_reason'] for name, case in report['cases'].items()})))
    return report


if __name__ == '__main__':
    main()
