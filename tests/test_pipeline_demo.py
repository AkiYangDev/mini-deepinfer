"""Keep the walkthrough's resource handoff real, rather than scripted output."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install torch for the CPU walkthrough')
class PipelineDemoTests(unittest.TestCase):
    def test_waiting_request_reuses_finished_requests_slots_and_matches_whole_prompts(self):
        from minideepinfer.pipeline_demo import run

        report = run()
        self.assertTrue(report['passed'] and report['outputs_match'] and report['clean'])
        self.assertEqual(report['free_slots'], 20)
        self.assertEqual({rid: len(r['output_ids']) for rid, r in report['requests'].items()},
                         {'A': 6, 'B': 2, 'C': 3})
        observations = report['observations']
        self.assertEqual(observations[0]['progress']['A']['committed'], 0)
        self.assertEqual(observations[0]['progress']['A']['issued'], 8)
        self.assertEqual(observations[1]['progress']['C']['status'], 'waiting')
        self.assertEqual(observations[1]['progress']['C']['slots'], 0)
        b_slots = set(observations[1]['owners']['B'])
        c_first = next(o for o in observations if 'C' in o['owners'])
        self.assertEqual(c_first['progress']['B']['status'], 'finished')
        self.assertNotIn('B', c_first['owners'])
        self.assertTrue(b_slots <= set(c_first['owners']['C']))
        self.assertEqual({s['slot'] for s in c_first['reused']}, b_slots)
        for o in observations:
            slots = [s for owned in o['owners'].values() for s in owned]
            self.assertEqual(len(slots), len(set(slots)))
            self.assertEqual(len(slots) + o['free_slots'], 20)
            self.assertLessEqual(len(o['owners']), 2)
            self.assertLessEqual(sum(len(s['positions']) for s in o['steps']), 4)
            for s in o['steps']:
                if s['partial']:
                    self.assertIsNone(s['sampled_token'])
        self.assertEqual(report['drained_released'], ['C'])


if __name__ == '__main__':
    unittest.main()
