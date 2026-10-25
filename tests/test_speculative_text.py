import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minideepinfer.generate import main, parse_args
from minideepinfer.speculative_text import run_speculative_text


class Tokenizer:
    eos_token_id = None

    def encode(self, text):
        return [int(t) for t in text.split()]

    def apply_chat_template(self, messages, **options):
        self.options = options
        return [1]+self.encode(messages[0]['content'])+[2]

    def decode(self, tokens, **options):
        return ' '.join(map(str, tokens))


class SpeculativeArgumentsTests(unittest.TestCase):
    def test_bad_combinations_stop_before_loading(self):
        bad = (['--speculative'], ['--draft-model', 'draft'], ['--draft-width', '3'],
               ['--output', 'out.json'], ['--check-greedy'], ['--stream'], ['--cancel-after-tokens', '2'],
               ['--speculative', '--draft-model', 'draft', '--pipeline'],
               ['--speculative', '--draft-model', 'draft', '--deferred'])
        extra = (['--dtype', 'bfloat16'], ['--prepare-during-step'], ['--chunked-prefill'],
                 ['--max-num-seqs', '1'], ['--max-batched-tokens', '8'], ['--stagger-prompts'],
                 ['--prompt', 'a', '--prompt', 'b'], ['--draft-width', '0'], ['--draft-width', '9'],
                 ['--max-new-tokens', '0'], ['--cache-capacity', '0'], ['--cancel-after-tokens', '0'])
        for options in (*bad, *(['--speculative', '--draft-model', 'draft', *o] for o in extra)):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), \
                    patch('minideepinfer.model_draft.check_tokenizer_pair') as contract:
                with self.assertRaises(SystemExit) as raised:
                    main(['--model', 'unused', *options])
                self.assertEqual(raised.exception.code, 2)
                contract.assert_not_called()
        args = parse_args(['--model', 'target', '--speculative', '--draft-model', 'draft'])
        self.assertEqual(args.draft_width, 1)
        self.assertFalse(args.check_greedy)

    def test_tokenizer_mismatch_is_rejected_before_torch_or_checkpoint_loading(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'torch' or name.startswith('transformers'):
                raise AssertionError('weights dependencies must not be imported yet')
            return original(name, *args, **kwargs)
        with patch('minideepinfer.model_draft.check_tokenizer_pair', side_effect=ValueError('tokenizer files differ')), \
                patch('builtins.__import__', guarded):
            with self.assertRaisesRegex(ValueError, 'tokenizer files differ'):
                main(['--model', 'target', '--speculative', '--draft-model', 'draft'])


AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(AVAILABLE, 'install torch for speculative text checks')
class SpeculativeTextTests(unittest.TestCase):
    def models(self, hybrid=True):
        import torch
        from minideepinfer.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
        from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
        torch.set_num_threads(1)
        if hybrid:
            config = Qwen3_5Config(97, 16, 24, 2, 2, 1, 8, 1, 2, 4, 4,
                                   ('linear_attention', 'full_attention'), max_position_embeddings=64,
                                   partial_rotary_factor=0.5, tie_word_embeddings=False)
            implementation = Qwen3_5ForCausalLM
        else:
            config = Qwen2Config(97, 16, 24, 2, 2, 1, max_position_embeddings=64)
            implementation = Qwen2ForCausalLM
        torch.manual_seed(2026)
        target = implementation(config).eval()
        torch.manual_seed(2036)
        return target, implementation(config).eval()

    def assert_clean(self, result):
        self.assertEqual(result['free_slots'], result['cache_capacity'])
        self.assertEqual(result['draft_work']['free_slots'], result['cache_capacity'])
        self.assertTrue(result['draft_work']['closed'])
        self.assertEqual(result['draft_work']['linear_states_remaining'], 0)
        self.assertFalse(result['draft_work']['snapshot_retained'])

    @contextlib.contextmanager
    def cli_fixture(self, target, small, tokenizer, *options):
        from minideepinfer.model_runner import ModelRunner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/'target', root/'draft']
            for path in paths:
                path.mkdir()
                (path/'config.json').write_text(json.dumps(dict(vocab_size=97)))
                (path/'tokenizer.json').write_text('{}')
                (path/'tokenizer_config.json').write_text('{}')
            output, stdout = root/'report.json', io.StringIO()
            argv = ['--model', str(paths[0]), '--speculative', '--draft-model', str(paths[1]),
                    '--device', 'cpu', '--raw', '--prompt', '3 4', '--max-new-tokens', '8',
                    '--cache-capacity', '32', '--output', str(output), *options]
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                    patch.object(ModelRunner, 'from_pretrained', side_effect=[ModelRunner(target, 1), ModelRunner(small, 1)]), \
                    contextlib.redirect_stdout(stdout):
                yield argv, stdout, output

    def test_pre_cancel_has_no_model_work_and_no_empty_greedy_run(self):
        from threading import Event
        from minideepinfer.model_draft_trace import print_trace
        target, small = self.models()
        stop = Event()
        stop.set()
        with patch.object(target, 'forward') as large, patch.object(small, 'forward') as little:
            result = run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                         max_new_tokens=8, cache_capacity=16, trace_rounds=True,
                         check_greedy=True, should_cancel=stop.is_set)
            large.assert_not_called()
            little.assert_not_called()
        self.assertEqual(result['finish_reason'], 'cancelled')
        self.assertEqual(result['output_ids'], [])
        self.assertEqual(result['target_forward_calls'], 0)
        self.assertEqual(result['direct_target_tokens'], 0)
        self.assertEqual(result['greedy_check'], dict(status='not_run', reason='no_committed_tokens'))
        self.assert_clean(result)
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            print_trace(result['trace'])
        self.assertIn('计算前已取消', rendered.getvalue())
        self.assertNotIn('待下一轮消费', rendered.getvalue())

    def test_cancel_after_prefill_never_starts_draft_and_checks_only_the_prefix(self):
        from threading import Event
        target, small = self.models()
        stop, groups = Event(), []
        def committed(tokens):
            groups.append(tokens)
            stop.set()
        result = run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                     max_new_tokens=8, cache_capacity=16, trace_rounds=True,
                     check_greedy=True, on_commit=committed, should_cancel=stop.is_set)
        self.assertEqual(result['finish_reason'], 'cancelled')
        self.assertEqual(len(result['output_ids']), 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(result['target_forward_calls'], 1)
        self.assertEqual(result['draft_work']['forward_calls'], 0)
        self.assertEqual(result['greedy_check']['scope'], 'committed_prefix')
        self.assertEqual(result['greedy_check']['finish_reason'], 'length')
        self.assertEqual(result['greedy_check']['output_ids'], result['output_ids'])
        self.assert_clean(result)

    def test_cancel_keeps_the_entire_accepted_group_and_never_starts_another_round(self):
        import copy
        from threading import Event
        for hybrid in (False, True):
            target, _ = self.models(hybrid)
            small = copy.deepcopy(target)
            stop, groups = Event(), []
            def committed(tokens):
                groups.append(tokens)
                if sum(map(len, groups)) >= 2:
                    stop.set()
            result = run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                         max_new_tokens=10, cache_capacity=16, width=3, trace_rounds=True,
                         check_greedy=True, on_commit=committed, should_cancel=stop.is_set)
            self.assertEqual(list(map(len, groups)), [1, 4])
            self.assertEqual(result['output_ids'], [t for group in groups for t in group])
            self.assertEqual(result['accepted_draft_tokens'], 3)
            self.assertEqual(result['target_forward_calls'], 2)
            self.assertEqual(result['draft_work']['calls'], 1)
            self.assertEqual(result['rounds'][0]['finish_reason'], 'cancelled')
            self.assertEqual(result['trace']['rounds'][0]['finish_reason'], 'cancelled')
            self.assertEqual(result['greedy_check']['status'], 'passed')
            self.assert_clean(result)

    def test_cancel_after_rejection_exposes_only_the_rolled_back_valid_cache(self):
        from threading import Event
        from minideepinfer.spec_demo import compare_cache
        from minideepinfer.speculative import CachedTarget
        target, small = self.models()
        stop, groups, checks = Event(), [], []
        rollback = CachedTarget.rollback
        def checked(cache, retained):
            invalidated = rollback(cache, retained)
            checks.append(compare_cache(cache))
            return invalidated
        def committed(tokens):
            groups.append(tokens)
            if len(groups) == 2:
                stop.set()
        with patch.object(CachedTarget, 'rollback', checked):
            result = run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                         max_new_tokens=8, cache_capacity=16, width=3, check_greedy=True,
                         on_commit=committed, should_cancel=stop.is_set)
        self.assertEqual(result['finish_reason'], 'cancelled')
        self.assertTrue(result['rounds'][0]['invalidated_slots'])
        self.assertGreater(result['replay_forward_calls'], 0)
        self.assertTrue(checks and all(row['passed'] for row in checks))
        self.assertEqual(result['greedy_check']['status'], 'passed')
        self.assert_clean(result)

    def test_natural_eos_and_length_completion_win_over_a_simultaneous_cancel(self):
        import copy
        from threading import Event
        target, _ = self.models()
        small = copy.deepcopy(target)
        plain = run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                                      max_new_tokens=5, cache_capacity=16)
        self.assertNotEqual(plain['output_ids'][0], plain['output_ids'][1])
        for eos, count, boundary, expected in ((None, 1, 1, 'length'),
                (plain['output_ids'][0], 8, 1, 'eos'), (None, 2, 2, 'length'),
                (plain['output_ids'][1], 8, 2, 'eos')):
            tokenizer = Tokenizer()
            tokenizer.eos_token_id = eos
            stop, groups = Event(), []
            def committed(tokens):
                groups.append(tokens)
                if len(groups) >= boundary:
                    stop.set()
            result = run_speculative_text(target, small, tokenizer, '3 4', raw=True,
                         max_new_tokens=count, cache_capacity=16, width=3, check_greedy=True,
                         on_commit=committed, should_cancel=stop.is_set)
            self.assertTrue(stop.is_set())
            self.assertEqual(result['finish_reason'], expected)
            self.assertEqual(result['greedy_check']['status'], 'passed')
            self.assertNotIn('scope', result['greedy_check'])
            self.assert_clean(result)

    def test_cancel_predicate_error_is_an_error_and_still_cleans_both_caches(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.speculative import CachedTarget
        target, small = self.models()
        drafts, caches, polls = [], [], []
        initialize = CachedTarget.__init__
        def created(*args, **kwargs):
            draft = ModelDraft(*args, **kwargs)
            drafts.append(draft)
            return draft
        def capture(cache, *args, **kwargs):
            initialize(cache, *args, **kwargs)
            caches.append(cache)
        def failed():
            polls.append(1)
            if len(polls) == 3:
                raise RuntimeError('cancel source failed')
            return False
        with patch('minideepinfer.model_draft.ModelDraft', created), patch.object(CachedTarget, '__init__', capture):
            with self.assertRaisesRegex(RuntimeError, 'cancel source failed'):
                run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                         max_new_tokens=8, cache_capacity=16, should_cancel=failed)
        self.assertEqual(len(caches), 2)
        self.assertTrue(all(not cache.cache.linear_states and cache._snapshot is None for cache in caches))
        self.assertTrue(drafts[0].closed)
        self.assertEqual(drafts[0].pool.free_count, drafts[0].pool.capacity)

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'install transformers for streaming CLI')
    def test_streaming_cli_delivers_text_before_return_and_saves_full_or_cancelled_output(self):
        from transformers import TextStreamer
        target, small = self.models()
        for options in ([], ['--cancel-after-tokens', '2']):
            active, events = [False], []
            class ObservedStreamer(TextStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    events.append((text, stream_end, active[0]))
                    super().on_finalized_text(text, stream_end)
            def running(*args, **kwargs):
                active[0] = True
                try:
                    return run_speculative_text(*args, **kwargs)
                finally:
                    active[0] = False
            with self.cli_fixture(target, small, Tokenizer(), '--stream', '--check-greedy', *options) as (argv, stdout, output), \
                    patch('transformers.TextStreamer', ObservedStreamer), \
                    patch('minideepinfer.speculative_text.run_speculative_text', running):
                result = main(argv)
                saved = json.loads(output.read_text(encoding='utf-8'))
                rendered = stdout.getvalue()
            self.assertEqual(saved, result)
            self.assertEqual(result['status'], 'cancelled' if options else 'completed')
            self.assertTrue(any(text and not ended and during for text, ended, during in events))
            self.assertEqual(sum(ended for _, ended, _ in events), 1)
            self.assertEqual(''.join(text for text, _, _ in events), result['text'])
            self.assertTrue(rendered.startswith(result['text']+'\n\n['))
            self.assertEqual(result['greedy_check']['status'], 'passed')
            self.assert_clean(result)

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'install transformers for streaming CLI')
    def test_streamer_failure_flushes_once_and_does_not_write_a_success_report(self):
        from transformers import TextStreamer
        from minideepinfer.model_draft import ModelDraft
        target, small = self.models()
        drafts, ended, puts = [], [], []
        class BrokenStreamer(TextStreamer):
            def put(self, tokens):
                super().put(tokens)
                puts.append(1)
                if len(puts) == 2:
                    raise OSError('output consumer failed')
            def end(self):
                ended.append(1)
                super().end()
        def created(*args, **kwargs):
            draft = ModelDraft(*args, **kwargs)
            drafts.append(draft)
            return draft
        with self.cli_fixture(target, small, Tokenizer(), '--stream') as (argv, stdout, output), \
                patch('transformers.TextStreamer', BrokenStreamer), patch('minideepinfer.model_draft.ModelDraft', created):
            with self.assertRaisesRegex(OSError, 'output consumer failed'):
                main(argv)
            self.assertFalse(output.exists())
        self.assertEqual(ended, [1])
        self.assertTrue(drafts[0].closed)
        self.assertEqual(drafts[0].pool.free_count, drafts[0].pool.capacity)

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'install transformers for Unicode streaming')
    def test_streaming_cli_buffers_a_character_split_across_tokens(self):
        target, small = self.models()
        plain = run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                                     max_new_tokens=3, cache_capacity=16)
        ids = plain['output_ids']
        self.assertEqual(len(set(ids)), 3)
        byte_for_token = dict(zip(ids, '你'.encode('utf-8')))
        class ByteTokenizer(Tokenizer):
            def decode(self, tokens, **options):
                return bytes(byte_for_token[t] for t in tokens).decode('utf-8', errors='replace')
        with self.cli_fixture(target, small, ByteTokenizer(), '--stream', '--max-new-tokens', '3') as (argv, stdout, output):
            result = main(argv)
            rendered = stdout.getvalue()
        self.assertEqual(result['text'], '你')
        self.assertTrue(rendered.startswith('你\n\n['))
        self.assertNotIn('\ufffd', rendered)
        self.assert_clean(result)

    def test_raw_chat_and_length_boundaries_match_independent_greedy(self):
        for hybrid in (False, True):
            target, small = self.models(hybrid)
            for raw in (False, True):
                for width, count in ((1, 1), (3, 2), (3, 8)):
                    tokenizer = Tokenizer()
                    with self.subTest(hybrid=hybrid, raw=raw, width=width, count=count):
                        result = run_speculative_text(target, small, tokenizer, '3 4', raw=raw,
                                      thinking=True, width=width, max_new_tokens=count,
                                      cache_capacity=128, check_greedy=True)
                        self.assertEqual(result['greedy_check']['status'], 'passed')
                        self.assertEqual(result['prompt_ids'], [3, 4] if raw else [1, 3, 4, 2])
                        self.assertEqual(result['text'], ' '.join(map(str, result['output_ids'])))
                        self.assertEqual(result['finish_reason'], 'length')
                        self.assertEqual(len(result['output_ids']), count)
                        self.assert_clean(result)
                        if count == 1:
                            self.assertEqual(result['draft_work']['forward_calls'], 0)
                        if not raw:
                            self.assertTrue(tokenizer.options['enable_thinking'])

    def test_eos_stops_at_prefill_and_in_later_verification(self):
        target, small = self.models()
        plain = run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                                     max_new_tokens=10, cache_capacity=16, width=3)
        self.assertEqual(plain['greedy_check']['status'], 'not_run')
        for token in (plain['output_ids'][0], plain['output_ids'][3]):
            tokenizer = Tokenizer()
            tokenizer.eos_token_id = token
            result = run_speculative_text(target, small, tokenizer, '3 4', raw=True,
                        max_new_tokens=10, cache_capacity=16, width=3, check_greedy=True)
            self.assertEqual(result['output_ids'], plain['output_ids'][:plain['output_ids'].index(token)+1])
            self.assertEqual(result['finish_reason'], 'eos')
            self.assertEqual(result['greedy_check']['status'], 'passed')
            self.assert_clean(result)

    def test_trace_and_commit_observer_preserve_the_same_generation_work(self):
        target, small = self.models()
        options = dict(raw=True, max_new_tokens=8, cache_capacity=16, width=3)
        plain = run_speculative_text(target, small, Tokenizer(), '3 4', **options)
        groups = []
        observed = run_speculative_text(target, small, Tokenizer(), '3 4', trace_rounds=True,
                                        on_commit=groups.append, **options)
        trace = observed.pop('trace')
        self.assertEqual(observed, plain)
        self.assertEqual([t for group in groups for t in group], plain['output_ids'])
        self.assertTrue(trace['rounds'])
        self.assertEqual(trace['draft_cleanup'], plain['draft_work'])
        self.assert_clean(plain)

    def test_invalid_request_does_not_start_either_model(self):
        target, small = self.models()
        for options in (dict(prompt=''), dict(prompt='97'), dict(prompt='-1'), dict(width=0),
                        dict(max_new_tokens=0), dict(cache_capacity=2), dict(max_new_tokens=65)):
            arguments = dict(prompt='3 4', raw=True, max_new_tokens=6, cache_capacity=128)
            arguments.update(options)
            with self.subTest(options=options), patch.object(target, 'forward') as large, patch.object(small, 'forward') as little:
                with self.assertRaises(ValueError):
                    run_speculative_text(target, small, Tokenizer(), **arguments)
                large.assert_not_called()
                little.assert_not_called()

    def test_commit_error_or_interrupt_still_closes_both_request_caches(self):
        from minideepinfer.model_draft import ModelDraft
        from minideepinfer.speculative import CachedTarget
        target, small = self.models()
        for error in (RuntimeError('consumer failed'), KeyboardInterrupt()):
            drafts, caches = [], []
            initialize = CachedTarget.__init__
            def created(*args, **kwargs):
                draft = ModelDraft(*args, **kwargs)
                drafts.append(draft)
                return draft
            def capture(cache, *args, **kwargs):
                initialize(cache, *args, **kwargs)
                caches.append(cache)
            calls = []
            def failed(tokens):
                calls.append(tokens)
                if len(calls) == 2:
                    raise error
            with patch('minideepinfer.model_draft.ModelDraft', created), patch.object(CachedTarget, '__init__', capture):
                with self.assertRaises(type(error)) as raised:
                    run_speculative_text(target, small, Tokenizer(), '3 4', raw=True,
                                          max_new_tokens=6, cache_capacity=16, on_commit=failed)
                self.assertIs(raised.exception, error)
            self.assertEqual(len(caches), 2)
            self.assertTrue(all(not cache.cache.linear_states and cache._snapshot is None for cache in caches))
            self.assertTrue(drafts[0].closed)
            self.assertEqual(drafts[0].pool.free_count, drafts[0].pool.capacity)

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'install transformers for CLI dispatch')
    def test_cli_loads_each_model_once_and_saves_actual_text_and_trace(self):
        from minideepinfer.model_runner import ModelRunner
        target, small = self.models()
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/'target', root/'draft']
            for path in paths:
                path.mkdir()
                (path/'config.json').write_text(json.dumps(dict(vocab_size=97)))
                (path/'tokenizer.json').write_text('{}')
                (path/'tokenizer_config.json').write_text('{}')
            output = root/'report.json'
            argv = ['--model', str(paths[0]), '--speculative', '--draft-model', str(paths[1]),
                    '--device', 'cpu', '--prompt', '3 4', '--thinking', '--draft-width', '3',
                    '--max-new-tokens', '6', '--cache-capacity', '32', '--trace-rounds',
                    '--check-greedy', '--output', str(output)]
            stdout = io.StringIO()
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer) as token_load, \
                    patch.object(ModelRunner, 'from_pretrained', side_effect=[ModelRunner(target, 1), ModelRunner(small, 1)]) as load, \
                    contextlib.redirect_stdout(stdout):
                result = main(argv)
            self.assertEqual(token_load.call_count, 1)
            self.assertEqual([call.args[0] for call in load.call_args_list], list(map(str, paths)))
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), result)
            self.assertEqual(result['greedy_check']['status'], 'passed')
            self.assertTrue(result['tokenizer_contract']['compatible'])
            self.assertIn(result['text'], stdout.getvalue())
            self.assertIn('accepted draft', stdout.getvalue())
            self.assertIn('greedy check passed', stdout.getvalue())
            self.assertEqual(len(result['source_sha256']), 13)
            self.assert_clean(result)


if __name__ == '__main__':
    unittest.main()
