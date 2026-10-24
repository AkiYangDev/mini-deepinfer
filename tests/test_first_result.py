import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minideepinfer.benchmark_first_result import run


class FirstResultTests(unittest.TestCase):
    def test_repetition_limit_is_checked_before_loading(self):
        for count in (0,4,True):
            with self.assertRaises(ValueError):
                run(repeats=count)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'install torch for pipeline comparison')
    def test_fixed_members_keep_identical_plans_work_and_outputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = run(repeats=2)
        self.assertTrue(report['passed'])
        self.assertEqual(report['total_forward_calls'],98)
        for case in report['cases'].values():
            self.assertEqual([t['first_result'] for t in case['trials']],['delayed','early','early','delayed'])
            for trial in case['warmups']+case['trials']:
                self.assertTrue(trial['outputs_match'] and trial['plans_match'] and trial['clean'])
                self.assertEqual(trial['forward_calls'],8)
                self.assertEqual(trial['extra_input_tokens'],0)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'install torch for pipeline comparison')
    def test_plan_mismatch_is_preserved_before_stopping(self):
        from minideepinfer.benchmark_first_result import run_trial
        calls = 0
        def changed(*args,**kwargs):
            nonlocal calls
            trial = run_trial(*args,**kwargs)
            calls += 1
            if calls == 4:
                trial['plans'] = []
            return trial
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'minideepinfer.benchmark_first_result.run_trial',changed):
                with self.assertRaisesRegex(ValueError,'plan, work'):
                    run(repeats=3,output=path)
            report = json.loads(path.read_text())
            self.assertEqual(calls,4)
            self.assertEqual(report['status'],'failed')
            self.assertFalse(report['cases']['single']['warmups'][-1]['plans_match'])
