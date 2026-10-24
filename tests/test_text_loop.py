"""Arrival order and preparation errors must not change the active batch."""

import unittest

from minideepinfer import BatchEngine, Status
from minideepinfer.text_loop import encode_prompt, run_text_requests
from test_deferred_engine import ControlledRunner


class Tokenizer:
    eos_token_id = None

    def __init__(self, runner=None):
        self.runner = runner
        self.calls = []

    def encode(self, text):
        self.calls.append((text, self.runner.pending is not None if self.runner else False))
        if text == 'broken':
            raise ValueError('bad incoming text')
        return [int(value) for value in text.split()]

    def apply_chat_template(self, messages, **kwargs):
        self.template_args = (messages, kwargs)
        return self.encode(messages[0]['content'])


class TextLoopTests(unittest.TestCase):
    def make(self):
        runner = ControlledRunner()
        engine = BatchEngine(runner, 16, 2, 3, chunked_prefill=True, deferred=True)
        return runner, engine, Tokenizer(runner)

    def test_two_orders_have_same_plans_tokens_and_reclamation(self):
        histories = []
        for during in (False, True):
            runner, engine, tokenizer = self.make()
            texts = ['1 2 3 4 5', '6 7', '8', '9 10 11']
            pairs = run_text_requests(engine, tokenizer, iter(texts), raw=True,
                                      max_new_tokens=3, prepare_during_step=during)
            self.assertEqual([text for request, text in pairs], texts)
            for request, _ in pairs:
                self.assertEqual(request.status, Status.FINISHED)
                for i, token in enumerate(request.output_ids):
                    self.assertEqual(token, sum(request.prompt_ids + tuple(request.output_ids[:i])) % 17)
            self.assertEqual(engine.cache.free_count, 16)
            self.assertEqual([pending for text, pending in tokenizer.calls], [False] + [during] * 3)
            histories.append((runner.math.batches, [r.output_ids for r, _ in pairs]))
        self.assertEqual(*histories)

    def test_empty_and_one_token_eos_inputs_drain_without_extra_work(self):
        for during in (False, True):
            runner, engine, tokenizer = self.make()
            self.assertEqual(run_text_requests(engine, tokenizer, [], prepare_during_step=during), [])
            self.assertEqual(runner.math.batches, [])
            tokenizer.eos_token_id = 3
            pairs = run_text_requests(engine, tokenizer, ['3', '3'], raw=True,
                                      max_new_tokens=3, prepare_during_step=during)
            self.assertTrue(all(r.finish_reason == 'eos' and r.output_ids == [3] for r, _ in pairs))
            self.assertEqual(engine.cache.free_count, 16)

    def test_bad_new_text_or_admission_does_not_abandon_launched_request(self):
        for bad_text in ('broken', '', '1 ' * 20):
            for during in (False, True):
                with self.subTest(text=bad_text, during=during):
                    runner, engine, tokenizer = self.make()
                    with self.assertRaises(ValueError):
                        run_text_requests(engine, tokenizer, ['2', bad_text], raw=True,
                                          max_new_tokens=3, prepare_during_step=during)
                    valid = engine._requests['req-0']
                    self.assertEqual(valid.output_ids, [2])
                    self.assertEqual(valid.num_cached_tokens, 1)
                    self.assertEqual(valid.status, Status.RUNNING)
                    self.assertIsNone(runner.pending)
                    list(engine.run())
                    self.assertEqual(valid.output_ids, [2, 4, 8])
                    self.assertEqual(engine.cache.free_count, 16)

    def test_lazy_input_can_observe_pending_state_and_cancel_current_request(self):
        runner, engine, tokenizer = self.make()
        def incoming():
            yield '1 2 3 4 5'
            self.assertIsNotNone(runner.pending)
            self.assertEqual(engine._requests['req-0'].num_cached_tokens, 0)
            engine.cancel('req-0')
            yield '7 8'
        pairs = run_text_requests(engine, tokenizer, incoming(), raw=True,
                                  max_new_tokens=2, prepare_during_step=True)
        self.assertEqual(pairs[0][0].status, Status.CANCELLED)
        self.assertEqual(pairs[0][0].output_ids, [])
        self.assertEqual(pairs[1][0].output_ids, [15, 13])
        self.assertEqual(engine.cache.free_count, 16)

    def test_finish_failure_retains_device_ownership_even_when_input_preparation_failed(self):
        runner, engine, tokenizer = self.make()
        runner.fail_wait = True
        with self.assertRaisesRegex(RuntimeError, 'wait failed') as caught:
            run_text_requests(engine, tokenizer, ['2', 'broken'], raw=True,
                              max_new_tokens=3, prepare_during_step=True)
        self.assertIsInstance(caught.exception.__context__, ValueError)
        self.assertIsNotNone(runner.pending)
        self.assertEqual(engine.cache.free_count, 13)
        self.assertEqual(engine._requests['req-0'].status, Status.FAILED)
        runner.fail_wait = False
        engine.abort_step()
        self.assertEqual(engine.cache.free_count, 16)

    def test_chat_template_options_and_split_mode_requirement(self):
        tokenizer = Tokenizer()
        self.assertEqual(encode_prompt(tokenizer, '2 3', thinking=True), [2, 3])
        messages, kwargs = tokenizer.template_args
        self.assertEqual(messages, [{'role': 'user', 'content': '2 3'}])
        self.assertEqual(kwargs, dict(tokenize=True, add_generation_prompt=True,
                                     return_dict=False, enable_thinking=True))
        with self.assertRaisesRegex(ValueError, 'deferred BatchEngine'):
            run_text_requests(object(), tokenizer, [], prepare_during_step=True)


if __name__ == '__main__':
    unittest.main()
