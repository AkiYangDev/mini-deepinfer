"""Check sampler-index spans and CPU/NPU overlap in paired Ascend traces."""

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from minideepinfer.analyze_prepare_trace import analyze


def analyze_one(trace, kernels):
    result = analyze(trace, kernels)
    raw = json.loads(trace.read_text(encoding='utf-8'), parse_float=Decimal)
    events = raw if isinstance(raw, list) else raw['traceEvents']
    def bounds(event):
        start = Decimal(str(event['ts']))
        return start, start + Decimal(str(event['dur']))
    launches = sorted([e for e in events if e.get('name') == 'mini.launch' and e.get('ph') == 'X'],
                      key=lambda e: bounds(e)[0])
    for step, launch in zip(result['steps'], launches):
        start, end = bounds(launch)
        selected = [e for e in events if e.get('ph') == 'X'
                    and bounds(e)[0] >= start and bounds(e)[1] <= end]
        forward = [e for e in selected if e.get('name') == 'mini.forward']
        indices = [e for e in selected if e.get('name') == 'mini.sampler_indices']
        if len(forward) != 1 or len(indices) > 1:
            raise ValueError('each launch needs one forward and at most one sampler-index span')
        step.update(sampler_index_us=None, index_before_forward=None,
                    index_after_forward=None, index_copy_us=[])
        if indices:
            s, e = bounds(indices[0])
            before, after = e <= bounds(forward[0])[0], s >= bounds(forward[0])[1]
            if not (before or after):
                raise ValueError('sampler-index span overlaps the CPU forward call')
            copies = [x for x in selected if x.get('name') == 'aten::copy_'
                      and bounds(x)[0] >= s and bounds(x)[1] <= e]
            if len(copies) != 1:
                raise ValueError('sampler-index span must contain one host copy call')
            step.update(sampler_index_us=float(indices[0]['dur']), index_before_forward=before,
                        index_after_forward=after, index_copy_us=[float(copies[0]['dur'])])
        s, e = step['tokenize_start_us'] - 250, step['tokenize_end_us'] + 250
        step['nearby_device_busy_us'] = [[max(a, s), min(b, e)]
            for a, b in result['device_busy_intervals_us'] if a < e and b > s]
    result.pop('device_busy_intervals_us')
    result['representation'] = 'profiled spans; busy intervals limited to +/-250 us around tokenization'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in ('late', 'early'):
        parser.add_argument('--' + variant + '-trace', type=Path, required=True)
        parser.add_argument('--' + variant + '-kernels', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/sampler-timeline.json'))
    args = parser.parse_args()
    report = dict(scope='paired profiled interval facts; not unprofiled performance',
        sampler_analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        variants={variant: analyze_one(getattr(args, variant + '_trace'), getattr(args, variant + '_kernels'))
                  for variant in ('late', 'early')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({variant: [{'step': step['step'], 'copy_us': step['index_copy_us'],
          'overlap_us': step['kernel_overlap_us']} for step in result['steps']]
          for variant, result in report['variants'].items()}, indent=2))


if __name__ == '__main__':
    main()
