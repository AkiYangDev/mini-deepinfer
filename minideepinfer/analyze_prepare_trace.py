"""Intersect CPU tokenization spans with device kernel intervals in an Ascend trace."""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def analyze(trace_path, kernels_path):
    raw = json.loads(trace_path.read_text(encoding='utf-8'), parse_float=Decimal)
    events = raw if isinstance(raw, list) else raw['traceEvents']
    def interval(event):
        start, duration = Decimal(str(event['ts'])), Decimal(str(event['dur']))
        if not start.is_finite() or not duration.is_finite() or duration < 0:
            raise ValueError('trace timestamps and durations must be finite and nonnegative')
        return start, start + duration
    markers = sorted((e for e in events if e.get('ph') == 'X' and
                      e.get('name') in ('mini.launch', 'mini.tokenize', 'mini.finish')),
                     key=lambda e: interval(e)[0])
    launches = [e for e in markers if e['name'] == 'mini.launch']
    finishes = [e for e in markers if e['name'] == 'mini.finish']
    tokenizations = [e for e in markers if e['name'] == 'mini.tokenize']
    if not launches or len(launches) != len(finishes):
        raise ValueError('missing or unmatched launch/finish markers')
    device_events = {}
    for event in events:
        if event.get('ph') == 'X' and (event.get('cat') == 'kernel' or event.get('name') == 'Computing'):
            start, end = interval(event)
            device_events.setdefault(start, set()).add(end)
    kernels = []
    with kernels_path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            start = Decimal(row['Start Time(us)'].strip())
            duration = Decimal(row['Duration(us)'].strip())
            if not start.is_finite() or not duration.is_finite() or duration < 0:
                raise ValueError('kernel timestamps and durations must be finite and nonnegative')
            kernels.append((start, start + duration))
    # Both exports print microseconds to three decimals. Start timestamps must
    # match exactly; durations can differ by one printed unit (one nanosecond).
    if not kernels or any(start not in device_events for start, end in kernels):
        raise ValueError('kernel table does not match the device events on this trace clock')
    duration_deltas = [min(abs(end - other) for other in device_events[start]) for start, end in kernels]
    if max(duration_deltas) > Decimal('0.001'):
        raise ValueError('kernel durations differ beyond the export precision')
    busy = merge(kernels)  # Union prevents duplicate/parallel kernels being counted twice.
    origin = interval(launches[0])[0]
    rows = []
    for launch, finish in zip(launches, finishes):
        launch_start, launch_end = interval(launch)
        finish_start, finish_end = interval(finish)
        if launch_end > finish_start:
            raise ValueError('launch/finish markers are not sequential')
        preparation = [e for e in tokenizations if interval(e)[0] >= launch_end
                       and interval(e)[1] <= finish_start]
        if len(preparation) != 1:
            raise ValueError('each captured step must prepare exactly one next text')
        start, end = interval(preparation[0])
        overlaps = merge((max(start, a), min(end, b)) for a, b in busy)
        rows.append(dict(step=len(rows), launch_us=float(launch_end - launch_start),
            tokenize_us=float(end - start), finish_us=float(finish_end - finish_start),
            kernel_overlap_us=float(sum((b - a for a, b in overlaps), Decimal(0))),
            launch_start_us=float(launch_start - origin), launch_end_us=float(launch_end - origin),
            tokenize_start_us=float(start - origin), tokenize_end_us=float(end - origin),
            finish_start_us=float(finish_start - origin), finish_end_us=float(finish_end - origin)))
    stop = max(interval(e)[1] for e in finishes)
    clipped = merge((max(a, origin), min(b, stop)) for a, b in busy)
    return dict(scope='profiled interval intersections, not unprofiled performance or causality',
        time_unit='us', kernel_rows=len(kernels), matching_device_events=len(kernels),
        max_export_duration_difference_us=float(max(duration_deltas)),
        steps=rows, window_us=float(stop - origin),
        kernel_busy_union_us=float(sum((b - a for a, b in clipped), Decimal(0))),
        device_busy_intervals_us=[[float(a - origin), float(b - origin)] for a, b in clipped],
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        kernel_table_sha256=hashlib.sha256(kernels_path.read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--kernels', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/prepare-timeline.json'))
    args = parser.parse_args()
    result = analyze(args.trace, args.kernels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'device_busy_intervals_us'}, indent=2))


if __name__ == '__main__':
    main()
