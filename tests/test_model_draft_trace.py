import contextlib
import importlib.util
import io
import unittest
from unittest.mock import patch

from minideepinfer.model_draft_demo import run
from minideepinfer.model_draft_trace import print_trace


class TraceInterfaceTests(unittest.TestCase):
    def test_incompatible_modes_fail_before_model_import(self):
        with patch.dict('sys.modules',{'torch':None}):
            for options in (dict(trace_rounds=True,repeats=1),dict(trace_rounds='yes'),
                    dict(cpu_example='rejection',device='npu:0'),dict(cpu_example='rejection',repeats=1),
                    dict(cpu_example='rejection',target_directory='target',draft_directory='draft'),dict(cpu_example='unknown')):
                with self.subTest(options=options),self.assertRaises(ValueError):run(**options)

    def test_final_output_is_not_described_as_another_pending_input(self):
        row=dict(draft_state=dict(prefix_ids=[2,3],limit=0),input_ids=[3],target=[4],
                 accepted_draft=0,committed=[4],final_role='target',tentative_cached=2,
                 retained_cached=2,invalidated_slots=[],replay_tokens=0,pending_token=4,finish_reason='length')
        report=dict(path='dual',width=3,prompt_ids=[2],capacity=2,initial_output=[3],rounds=[row],
                    target_free_slots=2,draft_cleanup=dict(free_slots=2,linear_states_remaining=0,snapshot_retained=False))
        output=io.StringIO()
        with contextlib.redirect_stdout(output):print_trace(report)
        text=output.getvalue()
        self.assertIn('请求结束，不再输入模型',text)
        self.assertNotIn('末尾待计算 token=4',text)
        self.assertIn('归还槽位 2/2',text)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'install torch for model draft observations')
class DraftTraceTests(unittest.TestCase):
    def test_natural_rejection_trace_preserves_all_outputs_state_and_model_work(self):
        plain=run(width=3,cpu_example='rejection')
        traced=run(width=3,cpu_example='rejection',trace_rounds=True)
        traces=traced.pop('trace_runs');traced.pop('trace_scope')
        self.assertEqual(traced,plain)
        self.assertEqual(len(traces),1)
        trace=traces[0]
        self.assertEqual(trace['initial_output'],plain['baseline']['output_ids'][:1])
        self.assertEqual(trace['width'],3)
        result=plain['checked_runs'][1]
        self.assertEqual(result['accepted_draft_tokens'],0)
        self.assertGreater(result['proposed_draft_tokens'],0)
        self.assertEqual(result['draft_work']['recovered_branches'],7)
        for observed,actual in zip(trace['rounds'],result['rounds']):
            state=observed['draft_state']
            self.assertEqual({key:value for key,value in observed.items() if key!='draft_state'},actual)
            self.assertEqual(state['candidates'],actual['draft'])
            if state['limit']:
                self.assertEqual(state['aligned_ids'],state['prefix_ids'])
                self.assertEqual(state['after_ids'][len(state['prefix_ids']):],state['candidates'][:-1])
                self.assertEqual(state['retained_before_sync']+state['sync_input_tokens'],len(state['aligned_ids']))
        second=trace['rounds'][1]['draft_state']
        self.assertEqual((len(second['before_ids']),second['retained_before_sync'],len(second['aligned_ids'])),(6,4,5))
        self.assertTrue(second['recovered_branch'] and second['hybrid'])
        self.assertEqual(trace['rounds'][-1]['draft_state']['limit'],0)
        self.assertIsNone(trace['rounds'][-1]['draft_state']['aligned_ids'])
        self.assertEqual(trace['draft_cleanup'],result['draft_work'])
        self.assertEqual(trace['target_free_slots'],12)

    def test_acceptance_and_single_candidate_snapshot_comparison_remain_observable(self):
        report=run(width=3,trace_rounds=True)
        trace=report['trace_runs'][0]
        self.assertTrue(any(row['final_role']=='bonus' for row in trace['rounds']))
        self.assertFalse(any(row['draft_state'].get('recovered_branch') for row in trace['rounds']))
        report=run(trace_rounds=True,compare_draft_snapshots=True)
        self.assertTrue(report['same_work_except_draft_snapshots'])
        self.assertEqual([trace['path'] for trace in report['trace_runs']],['always_snapshot','conditional_snapshot'])
        before,after=report['trace_runs']
        self.assertEqual(before['rounds'],after['rounds'])
        self.assertGreater(before['draft_cleanup']['snapshot_count'],0)
        self.assertEqual(after['draft_cleanup']['snapshot_count'],0)
