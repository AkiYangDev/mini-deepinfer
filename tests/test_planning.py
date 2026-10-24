import unittest

from minideepinfer import Request, Step
from minideepinfer.planning import IssuedProgress, plan_step


class PlanningTests(unittest.TestCase):
    def test_partial_final_and_decode_share_one_step_shape(self):
        request = Request('A', (2, 3, 4, 5, 6), 3)
        slots = tuple(range(7))
        partial = plan_step(request, slots, 0, 3)
        final = plan_step(request, slots, 3, 3)
        decode = plan_step(request, slots, 5, 2, 19)
        self.assertEqual(partial, Step('A', 'prefill', (2, 3, 4), (0, 1, 2), slots[:3], slots[:3], False))
        self.assertEqual(final.input_ids, (5, 6))
        self.assertTrue(final.is_last_prompt_chunk)
        self.assertEqual(decode, Step('A', 'decode', (19,), (5,), (5,), slots[:6]))

    def test_issued_progress_is_separate_from_cpu_commit(self):
        request = Request('A', (2, 3, 4), 2)
        first = plan_step(request, (0, 1, 2, 3), 0, 2)
        old = IssuedProgress()
        issued = old.advance(first)
        self.assertEqual(old, IssuedProgress(0, 0))
        self.assertEqual(issued, IssuedProgress(2, 0))
        issued = issued.advance(plan_step(request, (0, 1, 2, 3), 2, 2))
        self.assertEqual(issued, IssuedProgress(3, 1))
        self.assertEqual((request.num_cached_tokens, request.output_ids), (0, []))

    def test_invalid_position_budget_and_discontinuous_issue_are_rejected(self):
        request = Request('A', (2, 3), 1)
        for start, budget in ((-1, 1), (0, 0), (True, 1), (2, 1)):
            with self.assertRaises(ValueError):
                plan_step(request, (0, 1), start, budget)
        with self.assertRaises(ValueError):
            IssuedProgress(-1, 0)
        with self.assertRaises(ValueError):
            IssuedProgress(1, 0).advance(plan_step(request, (0, 1), 0, 2))
