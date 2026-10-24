"""The CLI text path must keep the device pipeline's commit and ownership rules."""

import contextlib
import importlib.util
import io
import unittest
from unittest.mock import patch

from minideepinfer.generate import main, parse_args, run_generation
from minideepinfer.pipeline_text import print_round, run_pipeline_text

AVAILABLE = importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
    from minideepinfer.stopping_pipeline import StoppingDecodePipeline


class TextTokenizer:
    eos_token_id = None

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [int(t) for t in text.split()]

    def apply_chat_template(self, messages, **options):
        self.options = options
        return self.encode(messages[0]['content'])

    def decode(self, tokens, **options):
        return ' '.join(map(str, tokens))


class GenerationArgumentsTests(unittest.TestCase):
    def test_invalid_pipeline_options_fail_before_model_import_or_loading(self):
        for options in (['--pipeline', '--deferred'], ['--pipeline', '--prepare-during-step'],
                        ['--trace-rounds'], ['--stagger-prompts'],
                        ['--pipeline', '--dtype', 'bfloat16']):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(['--model', 'unused', *options])
                self.assertEqual(error.exception.code, 2)

    def test_default_and_existing_deferred_options_keep_their_meaning(self):
        plain = parse_args(['--model', 'unused'])
        self.assertFalse(plain.pipeline or plain.deferred)
        self.assertTrue(plain.prompt)
        split = parse_args(['--model', 'unused', '--prepare-during-step'])
        self.assertTrue(split.deferred)
        self.assertFalse(split.pipeline)


@unittest.skipUnless(AVAILABLE, 'install torch for text pipeline checks')
class PipelineTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(2036)
        config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                              ('linear_attention',) * 3 + ('full_attention',),
                              max_position_embeddings=64, partial_rotary_factor=0.5)
        cls.model = Qwen3_5ForCausalLM(config).eval()

    def args(self, *extra, texts=None, count=4, capacity=24):
        argv = ['--model', 'unused', '--device', 'cpu', '--raw', '--max-new-tokens', str(count),
                '--cache-capacity', str(capacity), '--max-num-seqs', '2', '--max-batched-tokens', '4',
                '--chunked-prefill']
        for text in texts or ['2 3 4 5 6', '7 8 9 10 11 12 13 14', '2 3 4 5 6']:
            argv.extend(('--prompt', text))
        return parse_args([*argv, *extra])

    def runner(self, capacity=24):
        return ModelRunner(self.model, capacity)

    def assert_clean(self, runner, pool):
        self.assertFalse(runner._prefixes or runner.cache.linear_states)
        self.assertEqual(pool.free_count, pool.capacity)

    def test_eager_and_staggered_text_paths_match_sync_outputs_and_observe_releases(self):
        histories = []
        for mode in ((), ('--pipeline',), ('--pipeline', '--stagger-prompts')):
            records, tokenizer, runner = [], TextTokenizer(), self.runner()
            pairs, pool = run_generation(runner, tokenizer, self.args(*mode), on_round=records.append)
            self.assertEqual(tokenizer.calls, [text for request, text in pairs])
            self.assertTrue(all(r.finish_reason == 'length' for r, _ in pairs))
            histories.append([r.output_ids for r, _ in pairs])
            self.assert_clean(runner, pool)
            if mode:
                self.assertEqual(records[-1]['event'], 'drained')
                released = [rid for event in records for rid in event['released']]
                self.assertEqual(sorted(released), ['req-0', 'req-1', 'req-2'])
                self.assertTrue(any(s['partial'] and s['sampled_token'] is None
                                    for event in records[:-1] for s in event['steps']))
                if '--stagger-prompts' in mode:
                    self.assertEqual(records[0]['arrived'], ['req-1'])
                    self.assertIn('req-1', records[0]['waiting'])
                    self.assertNotIn('req-1', records[0]['owners'])
                    self.assertEqual(records[1]['arrived'], ['req-2'])
                rendered = io.StringIO()
                with contextlib.redirect_stdout(rendered):
                    for record in records:
                        print_round(record)
                self.assertIn('before CPU commit', rendered.getvalue())
                self.assertIn('observed released:', rendered.getvalue())
                self.assertIn('pipeline drained', rendered.getvalue())
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(histories[0], histories[2])

    def test_tail_callback_submits_every_text_even_when_each_generates_one_token(self):
        runner, tokenizer, records = self.runner(8), TextTokenizer(), []
        args = self.args('--pipeline', '--stagger-prompts', texts=['2 3', '4', '5 6'], count=1, capacity=8)
        pairs, pool = run_generation(runner, tokenizer, args, on_round=records.append)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(len(r.output_ids) == 1 for r, _ in pairs))
        self.assertEqual([rid for record in records[:-1] for rid in record['arrived']], ['req-1', 'req-2'])
        self.assert_clean(runner, pool)

    def test_chat_template_and_eos_apply_to_initial_and_later_prompts(self):
        first, _ = run_generation(self.runner(), TextTokenizer(), self.args(texts=['2 3'], count=1))
        tokenizer = TextTokenizer()
        tokenizer.eos_token_id = first[0][0].output_ids[0]
        args = self.args('--pipeline', '--stagger-prompts', texts=['2 3', '2 3'])
        args.raw, args.thinking = False, True
        runner = self.runner()
        pairs, pool = run_generation(runner, tokenizer, args)
        self.assertTrue(all(r.finish_reason == 'eos' and len(r.output_ids) == 1 for r, _ in pairs))
        self.assertTrue(tokenizer.options['enable_thinking'])
        self.assert_clean(runner, pool)

    def test_text_or_observer_failure_drains_before_runner_reuse(self):
        for broken_text in ('not-an-int', '2 ' * 30, ''):
            runner = self.runner()
            with self.subTest(text=broken_text), self.assertRaises(ValueError):
                run_generation(runner, TextTokenizer(), self.args('--pipeline', '--stagger-prompts',
                               texts=['2 3 4 5 6', broken_text]))
            self.assertFalse(runner._prefixes or runner.cache.linear_states)
            pairs, pool = run_generation(runner, TextTokenizer(), self.args('--pipeline', texts=['2 3'], count=1))
            self.assertEqual(len(pairs[0][0].output_ids), 1)
            self.assert_clean(runner, pool)
        runner = self.runner()
        def fail(record):
            raise RuntimeError('observer failed')
        with self.assertRaisesRegex(RuntimeError, 'observer failed'):
            run_generation(runner, TextTokenizer(), self.args('--pipeline'), on_round=fail)
        self.assertFalse(runner._prefixes or runner.cache.linear_states)

    def test_empty_input_does_not_start_a_pipeline(self):
        pipeline = StoppingDecodePipeline(self.runner())
        self.assertEqual(run_pipeline_text(pipeline, TextTokenizer(), []), [])
        self.assertFalse(pipeline._active)
        self.assert_clean(pipeline.runner, pipeline.pool)

    def test_eos_lookahead_logits_have_an_independent_full_prompt_reference(self):
        from dataclasses import asdict
        from minideepinfer.verify_pipeline_text import compare
        initial, _ = run_generation(self.runner(), TextTokenizer(), self.args(texts=['2 3'], count=1))
        tokenizer = TextTokenizer()
        tokenizer.eos_token_id = initial[0][0].output_ids[0]
        results = []
        for mode in ((), ('--pipeline',)):
            runner, captured = self.runner(), []
            forward = runner.execute_main_model
            def capture(inputs):
                logits = forward(inputs)
                captured.append((inputs.steps, logits.detach()))
                return logits
            runner.execute_main_model = capture
            try:
                pairs, pool = run_generation(runner, tokenizer, self.args(*mode, texts=['2 3']))
            finally:
                runner.execute_main_model = forward
            results.append(dict(requests={r.request_id: asdict(r) for r, _ in pairs}, captures=captured))
            self.assert_clean(runner, pool)
        comparisons, matches, extra = compare(*results, self.model)
        self.assertTrue(matches and all(row['passed'] for row in comparisons))
        self.assertEqual(extra, 1)
        self.assertTrue(comparisons[-1]['discarded_after_eos'])

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'install transformers for CLI dispatch')
    def test_main_uses_the_pipeline_and_prints_the_same_final_text(self):
        tokenizer, runner = TextTokenizer(), self.runner()
        argv = ['--model', 'unused', '--device', 'cpu', '--raw', '--pipeline', '--trace-rounds',
                '--stagger-prompts', '--max-new-tokens', '1', '--cache-capacity', '24',
                '--max-num-seqs', '2', '--prompt', '2 3', '--prompt', '4']
        output = io.StringIO()
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), patch(
                'minideepinfer.model_runner.ModelRunner.from_pretrained', return_value=runner), \
                contextlib.redirect_stdout(output):
            main(argv)
        self.assertIn('req-1: 4', output.getvalue())
        self.assertIn('pipeline drained', output.getvalue())
        self.assertIn('free KV slots 24/24', output.getvalue())
        self.assertFalse(runner._prefixes or runner.cache.linear_states)


if __name__ == '__main__':
    unittest.main()
