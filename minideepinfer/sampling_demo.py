"""CPU probability experiment and two untrained Qwen3.5 models; no checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def probability_experiment(trials=10000):
    from .sampling import SampledDraft, draw, verify_sampled
    p, q = [.25, .75], [.75, .25]
    after = ([.8, .2], [.1, .9])
    counts = [[0, 0], [0, 0]]
    rng = random.Random(2026)
    for _ in range(trials):
        candidate = draw(q, rng)
        decision, _ = verify_sampled(SampledDraft((candidate,), (q,)), (p, after[candidate]), 2, rng)
        first = decision.tokens[0]
        second = decision.tokens[1] if len(decision.tokens) == 2 else draw(after[first], rng)
        counts[first][second] += 1
    expected = [[p[a]*after[a][b] for b in range(2)] for a in range(2)]
    observed = [[n/trials for n in row] for row in counts]
    error = max(abs(observed[a][b]-expected[a][b]) for a in range(2) for b in range(2))
    return dict(p=p, q=q, next_p_given_first=after, trials=trials, seed=2026,
                counts=counts, expected_joint=expected, observed_joint=observed,
                max_abs_frequency_error=error, tolerance=.015, passed=error < .015,
                scope='fixed two-token toy distribution; empirical check, not a proof for arbitrary models')


def run(output=None, *, verbose=True):
    import torch
    from .model_draft import ModelDraft
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from .sampling import SampledDraft, TemperatureSampler
    from .spec_demo import compare_cache
    from .speculative import CachedTarget, generate

    report = dict(passed=False, device='cpu', dtype='float32', torch=torch.__version__,
                  temperature=.7, target_sampling_seed=2026, draft_sampling_seed=2027,
                  weight_seeds=[2026,2036], trained=False, tie_word_embeddings=False,
                  prompt_ids=[2,3,4], max_new_tokens=10, cache_capacity=12,
                  weight_loads=0, source_sha256={}, cache_checks=[], cases={},
                  scope='CPU FP64 sampling reference and untrained models; no NPU run, timing or speedup claim')
    sources = ('sampling.py','sampling_demo.py','model_draft.py','speculative.py','spec_demo.py',
               'models/qwen2.py','models/qwen3_5.py','model_runner.py','tensor_cache.py',
               'hybrid_cache.py','engine.py','kv_cache.py')
    report['source_sha256'] = {'minideepinfer/'+name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in sources}
    def save():
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.tmp')
            temporary.write_bytes((json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
            temporary.replace(path)
    phase = 'setup'
    save()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        target = Qwen3_5ForCausalLM(Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,
            ('linear_attention',)*3+('full_attention',),max_position_embeddings=64,
            partial_rotary_factor=.5,tie_word_embeddings=False)).eval()
        torch.manual_seed(2036)
        small = Qwen3_5ForCausalLM(Qwen3_5Config(97,16,24,2,2,1,8,1,2,4,4,
            ('linear_attention','full_attention'),max_position_embeddings=64,
            partial_rotary_factor=.5,tie_word_embeddings=False)).eval()
        phase = 'toy_distribution'
        report['probability_experiment'] = probability_experiment()
        if not report['probability_experiment']['passed']:
            raise RuntimeError('toy distribution frequency check failed')
        phase = 'target_only_reference'
        # Full-prefix recomputation is independent of speculative cache rollback.
        sampler, reference = TemperatureSampler(.7,2026), []
        for _ in range(10):
            cold = CachedTarget(target,tuple(range(12)),12)
            reference.append(sampler.sample(cold.forward((2,3,4)+tuple(reference))[-1]))
        report['full_prefix_reference'] = dict(output_ids=reference,forward_calls=10,input_tokens=sum(range(3,13)))
        def check(cache):
            row = compare_cache(cache)
            report['cache_checks'].append(row)
        phase = 'width_zero'
        ordinary = generate(target,(2,3,4),10,lambda *_: SampledDraft((),()),width=0,
                            sampler=TemperatureSampler(.7,2026),on_cache=check)
        report['cases']['width_zero'] = ordinary
        if ordinary['output_ids'] != reference:
            raise RuntimeError('target-only sampling differs from full-prefix reference')
        phase = 'two_models'
        draft = ModelDraft(small,12,sampler=TemperatureSampler(.7,2027),on_cache=check)
        try:
            sampled = generate(target,(2,3,4),10,draft,width=3,
                               sampler=TemperatureSampler(.7,2026),on_cache=check)
        finally:
            draft.close()
        sampled['draft_work'] = draft.stats()
        report['cases']['draft_width_three'] = sampled
        if (ordinary['free_slots'] != 12 or sampled['free_slots'] != 12
                or draft.stats()['free_slots'] != 12 or draft.stats()['linear_states_remaining']
                or draft.stats()['snapshot_retained']):
            raise RuntimeError('request cleanup failed')
        report['ordinary_matches_full_prefix'] = True
        report['passed'] = True
    except BaseException as error:
        report['failure'] = dict(phase=phase,type=type(error).__name__,message=str(error))
        save()
        raise
    save()
    if verbose:
        toy = report['probability_experiment']
        print(f"两 token 小词表：目标分布 {toy['expected_joint']}；{toy['trials']} 次观察 {toy['observed_joint']}。")
        print(f"普通采样与整段重算一致：{ordinary['output_ids']}。")
        print(f"双模型随机输出：{sampled['output_ids']}；不要求与同 seed 的普通采样逐 token 相同。")
        for row in sampled['rounds']:
            print(f"  提出 {row['draft']}，接受 {row['accepted_draft']}，提交 {row['committed']}；"
                  f"target 缓存 {row['tentative_cached']} → {row['retained_cached']}。")
        print(f"冷缓存检查 {len(report['cache_checks'])} 次通过；target/draft 各归还 12 个槽位。")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
