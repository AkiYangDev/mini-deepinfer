import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minideepinfer.benchmark_pipeline import metrics, run


class CommitMetricsTests(unittest.TestCase):
    def test_shared_arrival_token_latency_and_drain_have_distinct_endpoints(self):
        report = metrics({'A':[2_000_000, 5_000_000], 'B':[4_000_000]}, 1_000_000, 9_000_000)
        self.assertEqual(report['requests']['A']['ttft_ms'], 1)
        self.assertEqual(report['requests']['A']['e2e_ms'], 4)
        self.assertEqual(report['requests']['A']['tpot_ms'], 3)
        self.assertIsNone(report['requests']['B']['tpot_ms'])
        self.assertEqual(report['drain_after_last_commit_ms'], 4)
        self.assertEqual(report['output_tokens_per_s'], 375)
        for times in ({'A':[]}, {'A':[0]}, {'A':[10_000_000]}, {'A':[5_000_000, 2_000_000]}):
            with self.assertRaises(ValueError):
                metrics(times, 1_000_000, 9_000_000)

    def test_repetition_limit_rejects_costly_or_empty_runs(self):
        for count in (0, 4, True):
            with self.assertRaises(ValueError):
                run(repeats=count)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install torch for the paired comparison')
class PipelineBenchmarkTests(unittest.TestCase):
    def test_all_workloads_match_outputs_work_and_cleanup_in_both_pair_orders(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = run(repeats=2)
        self.assertTrue(report['passed'])
        self.assertEqual(report['failure_count'], 0)
        self.assertEqual(set(report['cases']), {'single','equal_batch','mixed_batch'})
        for case in report['cases'].values():
            self.assertEqual([t['mode'] for t in case['trials']], ['sync','pipeline','pipeline','sync'])
            reference = case['warmups'][0]['outputs']
            for trial in case['warmups']+case['trials']:
                self.assertEqual(trial['outputs'], reference)
                self.assertEqual(trial['extra_input_tokens'], 0)
                self.assertEqual(trial['free_slots'], 128)
                self.assertTrue(trial['passed'] and trial['clean'])
                self.assertTrue(all(r['output_tokens']==8 for r in trial['requests'].values()))
        self.assertTrue(report['cases']['single']['same_plans'])

    def test_admission_comparison_reuses_benchmark_and_preserves_actual_work(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report=run(repeats=1,compare_admission=True)
        self.assertEqual(report['comparison'],['pipeline','collect_before_admit'])
        self.assertTrue(report['passed'])
        self.assertTrue(report['cases']['single']['same_plans'])
        for name in ('equal_batch','mixed_batch'):
            original,changed=report['cases'][name]['trials']
            self.assertLessEqual(changed['forward_calls'],original['forward_calls'])
            self.assertEqual(original['processed_input_tokens'],changed['processed_input_tokens'])
            self.assertEqual(original['outputs'],changed['outputs'])
        original,changed=report['cases']['equal_batch']['trials']
        self.assertLess(changed['forward_calls'],original['forward_calls'])

    def test_first_bad_trial_is_preserved_and_stops_further_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'partial.json'
            bad = dict(mode='sync', passed=False, outputs={'req-0':[]}, clean=False)
            with patch('minideepinfer.benchmark_pipeline.run_trial', return_value=bad) as trial:
                with self.assertRaisesRegex(ValueError, 'cleanup mismatch'):
                    run(repeats=3, output=path)
            trial.assert_called_once()
            saved = json.loads(path.read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['failure_count'], 1)
            self.assertFalse(saved['passed'])
            self.assertEqual(len(saved['cases']['single']['warmups']), 1)


if __name__ == '__main__':
    unittest.main()
