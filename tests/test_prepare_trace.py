"""Clock matching and interval unions for CPU/NPU overlap evidence."""

from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from minideepinfer.analyze_prepare_trace import analyze, merge
from examples.analyze_sampler_order import analyze_one


class PrepareTraceTests(unittest.TestCase):
    def fixtures(self, root, intervals=((12, 2), (13, 2))):
        base = Decimal('1792814075449276.440')
        def event(name, start, duration):
            return dict(name=name, ph='X', ts=str(base + start), dur=duration)
        events = [event('mini.launch', 0, 10), event('mini.tokenize', 11, 4),
                  event('mini.finish', 15, 5)]
        for start, duration in intervals:
            # Actual exporter may omit cat and duplicate Computing on two tracks.
            events.extend([event('Computing', start, duration)] * 2)
        trace, table = root / 'trace.json', root / 'kernels.csv'
        trace.write_text(json.dumps(events), encoding='utf-8')
        rows = ['Start Time(us),Duration(us)']
        rows += [f'{base + start}\t,{duration}' for start, duration in intervals]
        table.write_text('\n'.join(rows), encoding='utf-8')
        return trace, table

    def test_decimal_clock_and_union_do_not_double_count_parallel_activity(self):
        with TemporaryDirectory() as directory:
            result = analyze(*self.fixtures(Path(directory)))
        self.assertEqual(result['steps'][0]['kernel_overlap_us'], 3)
        self.assertEqual(result['kernel_busy_union_us'], 3)
        self.assertEqual(result['steps'][0]['tokenize_us'], 4)
        self.assertEqual(result['device_busy_intervals_us'], [[12, 15]])

    def test_zero_overlap_is_a_valid_observation(self):
        with TemporaryDirectory() as directory:
            result = analyze(*self.fixtures(Path(directory), ((2, 4),)))
        self.assertEqual(result['steps'][0]['kernel_overlap_us'], 0)
        self.assertEqual(merge([(0, 2), (1, 4), (8, 7)]), [(0, 4)])

    def test_missing_markers_or_wrong_device_clock_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace, table = self.fixtures(root)
            table.write_text('Start Time(us),Duration(us)\n12,2', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'trace clock'):
                analyze(trace, table)
            trace, table = self.fixtures(root)
            events = json.loads(trace.read_text(encoding='utf-8'))
            trace.write_text(json.dumps(events[1:]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'markers'):
                analyze(trace, table)

    def test_duration_rounding_is_limited_to_one_printed_nanosecond(self):
        with TemporaryDirectory() as directory:
            trace, table = self.fixtures(Path(directory), ((12, 2),))
            original = table.read_text(encoding='utf-8')
            table.write_text(original.replace(',2', ',2.001'), encoding='utf-8')
            self.assertEqual(analyze(trace, table)['max_export_duration_difference_us'], .001)
            table.write_text(original.replace(',2', ',2.002'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'export precision'):
                analyze(trace, table)

    def test_sampler_markers_identify_position_and_reject_ambiguous_ranges(self):
        for before in (False, True):
            with TemporaryDirectory() as directory:
                trace, table = self.fixtures(Path(directory), ((12, 2),))
                events = json.loads(trace.read_text(encoding='utf-8'))
                base = Decimal(events[0]['ts'])
                events.append(dict(name='mini.forward', ph='X', ts=str(base + 3), dur=3))
                start = base + (1 if before else 8)
                events.extend([dict(name='mini.sampler_indices', ph='X', ts=str(start), dur=1),
                               dict(name='aten::copy_', ph='X', ts=str(start), dur=.5)])
                trace.write_text(json.dumps(events), encoding='utf-8')
                step = analyze_one(trace, table)['steps'][0]
                self.assertEqual(step['index_before_forward'], before)
                self.assertEqual(step['index_after_forward'], not before)
                self.assertEqual(step['index_copy_us'], [.5])
                events.append(events[-2])  # Duplicate span must not silently choose a row.
                trace.write_text(json.dumps(events), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'at most one'):
                    analyze_one(trace, table)


if __name__ == '__main__':
    unittest.main()
