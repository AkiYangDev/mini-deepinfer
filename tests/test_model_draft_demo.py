import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minideepinfer.model_draft_demo import commit_metrics


class CommitMetricsTests(unittest.TestCase):
    def test_grouped_commits_keep_zero_gaps_and_include_final_cleanup(self):
        result=commit_metrics([2_000_000,3_000_000,3_000_000],1_000_000,5_000_000)
        self.assertEqual(result['ttft_ms'],1)
        self.assertEqual(result['e2e_ms'],2)
        self.assertEqual(result['tpot_ms'],0.5)
        self.assertEqual(result['itl_ms'],[1,0])
        self.assertEqual(result['whole_ms'],4)
        self.assertEqual(result['output_tokens_per_second'],750)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'install torch for model draft demo')
class ModelDraftDemoTests(unittest.TestCase):
    def test_snapshot_comparison_keeps_model_work_and_swaps_order(self):
        from minideepinfer.model_draft_demo import run
        report=run(repeats=2,compare_draft_snapshots=True)
        self.assertTrue(report['same_work_except_draft_snapshots'])
        self.assertEqual([row['path'] for row in report['trials']],
            ['always_snapshot','conditional_snapshot','conditional_snapshot','always_snapshot'])
        self.assertEqual([row['width'] for row in report['checked_runs']],[1,1,3,3])
        for row in report['trials']:
            stats=row['draft_work']
            if row['path']=='always_snapshot':
                self.assertGreater(stats['snapshot_count'],0)
                self.assertGreater(stats['snapshot_bytes_total'],0)
            else:
                self.assertEqual(stats['snapshot_count'],0)
                self.assertEqual(stats['snapshot_bytes_total'],0)
        before,after=report['checked_runs'][2:]
        self.assertEqual(before['draft_work']['snapshot_count'],after['draft_work']['snapshot_count'])
        with self.assertRaisesRegex(ValueError,'requires width 1'):run(width=3,compare_draft_snapshots=True)

    def test_checks_and_paired_trials_account_for_both_models(self):
        from minideepinfer.model_draft_demo import run
        report=run(repeats=1)
        self.assertTrue(report['passed'])
        self.assertEqual([row['width'] for row in report['checked_runs']],[1,3])
        self.assertEqual(len(report['warmups']),2)
        self.assertEqual([row['path'] for row in report['trials']],['baseline','dual'])
        self.assertTrue(all(row['passed'] for row in report['checks']))
        self.assertEqual(report['total_forward_calls'],sum(row['forward_calls'] for roles in report['model_work'].values() for row in roles.values()))
        for trial in report['trials']:
            self.assertEqual(trial['output_ids'],report['baseline']['output_ids'])
            self.assertGreaterEqual(trial['metrics']['whole_ms'],trial['metrics']['e2e_ms'])
            if trial['path']=='dual':
                self.assertEqual(trial['draft_work']['free_slots'],12)
                self.assertGreater(trial['draft_work']['forward_calls'],0)
                self.assertEqual(trial['acceptance']['strict_rate'],trial['acceptance']['accepted']/trial['acceptance']['proposed'])

    def test_failed_reference_is_saved_and_stops_before_timing(self):
        from minideepinfer.model_draft_demo import run
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'failed.json'
            with patch('minideepinfer.spec_demo.compare_cache',return_value=dict(passed=False,reference_forward_calls=1,reference_input_tokens=4)) as check:
                with self.assertRaisesRegex(RuntimeError,'cold full-prefix'):
                    run(output,repeats=1)
            report=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(check.call_count,1)
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['failure_count'],1)
            self.assertFalse(report['passed'])
            self.assertEqual(len(report['checks']),1)
            self.assertEqual(report['trials'],[])
            self.assertEqual(len(report['source_sha256']),11)

    def test_different_model_work_stops_comparison_before_warmup(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.model_draft_demo import run
        stats=ModelDraft.stats
        calls=[]
        def changed(draft):
            result=stats(draft);calls.append(result)
            if len(calls)==2:result['input_tokens']+=1
            return result
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'failed.json'
            with patch.object(ModelDraft,'stats',changed):
                with self.assertRaisesRegex(RuntimeError,'model work'):
                    run(output,repeats=1,compare_draft_snapshots=True)
            report=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'failed')
            self.assertEqual(len(report['checked_runs']),2)
            self.assertEqual(report['warmups'],[])
            self.assertEqual(report['trials'],[])
