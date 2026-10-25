"""Read top-k/top-p probabilities and verify two tiny models on CPU; no weights."""

import argparse
import hashlib
import json
from pathlib import Path


def run(output=None):
    import torch
    from .models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from .sampling import TemperatureSampler
    from .verify_sampling import check_pair

    torch.set_num_threads(1)
    config = Qwen3_5Config(97,32,48,4,2,1,16,2,4,8,8,
        ('linear_attention',)*3+('full_attention',),max_position_embeddings=64,
        partial_rotary_factor=.5,tie_word_embeddings=False)
    torch.manual_seed(2026)
    target = Qwen3_5ForCausalLM(config).eval()
    torch.manual_seed(2036)
    small = Qwen3_5ForCausalLM(config).eval()
    report = dict(passed=False,device='cpu',dtype='float32',torch=torch.__version__,
                  trained=False,weight_loads=0,weight_seeds=[2026,2036],
                  scope='Fixed CPU filtering example and untrained models; no timing or speedup claim')
    values = torch.tensor([.45,.30,.15,.10],dtype=torch.float64).log()
    report['filter_example'] = dict(
        unfiltered=values.softmax(-1).tolist(),
        top_k_three=TemperatureSampler(top_k=3).probabilities(values).tolist(),
        top_k_three_then_p_point_seven=TemperatureSampler(top_k=3,top_p=.7).probabilities(values).tolist())
    sources = ('sampling_filters_demo.py','verify_sampling.py','sampling.py','model_draft.py',
               'speculative.py','spec_demo.py','models/qwen2.py','models/qwen3_5.py',
               'model_runner.py','tensor_cache.py','hybrid_cache.py','engine.py','kv_cache.py')
    report['source_sha256'] = {'minideepinfer/'+name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in sources}
    def save():
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    try:
        check_pair(target,small,(2,3,4),report,top_k=7,top_p=.8)
    except BaseException as error:
        report['failure'] = dict(type=type(error).__name__,message=str(error))
        save()
        raise
    save()
    print('四个 token：',report['filter_example'])
    print('两模型各自过滤：top-k=7，top-p=0.8。')
    for name,case in report['cases'].items():
        print(name,case['output_ids'],case['finish_reason'],f"free={case['free_slots']}/{report['capacity']}")
    print(f"概率对照 {len(report['probability_checks'])} 组、缓存对照 {len(report['cache_checks'])} 组通过。")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
