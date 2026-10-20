"""Packed batches must agree with independent per-request Transformers calls."""

from dataclasses import asdict
import contextlib
import importlib.util
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))
if AVAILABLE:
    import torch
    from transformers import Qwen2Config as ReferenceConfig, Qwen2ForCausalLM as ReferenceModel
    from minideepinfer import BatchEngine, Status, Step
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
    from minideepinfer.verify_batch import packed_reference_cache
    from minideepinfer import verify


@unittest.skipUnless(AVAILABLE, "install the model extra for numerical tests")
class BatchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(2027)
        cls.config = Qwen2Config(97, 32, 64, 2, 4, 2, max_position_embeddings=64)
        config = ReferenceConfig(**asdict(cls.config), tie_word_embeddings=False,
                                 use_sliding_window=False, attention_dropout=0.0)
        config._attn_implementation = "eager"
        cls.reference = ReferenceModel(config).eval()

    def runner(self):
        model = Qwen2ForCausalLM(self.config)
        model.load_state_dict(self.reference.state_dict(), strict=True)
        return ModelRunner(model, 32)

    @torch.inference_mode() if AVAILABLE else lambda f: f
    def reference_logits(self, tokens):
        return self.reference(torch.tensor([tokens]), use_cache=False).logits[0]

    def test_mixed_phases_reordering_and_dirty_slot_reuse_match_reference(self):
        runner = self.runner()
        batches = [
            (Step("a", "prefill", (2, 3, 4), (0, 1, 2), (7, 1, 9), (7, 1, 9)),
             Step("b", "prefill", (5,), (0,), (0,), (0,))),
            (Step("b", "decode", (8,), (1,), (4,), (0, 4)),
             Step("a", "decode", (9,), (3,), (3,), (7, 1, 9, 3))),
            (Step("a", "decode", (13,), (4,), (5,), (7, 1, 9, 3, 5)),
             Step("c", "prefill", (11, 12), (0, 1), (0, 4), (0, 4))),
        ]
        prefixes, calls = {}, []
        with runner.model.register_forward_hook(lambda *args: calls.append(1)):
            for index, batch in enumerate(batches):
                if index == 2:
                    runner.release("b")
                for step, logits in zip(batch, runner.forward_batch(batch)):
                    prefixes.setdefault(step.request_id, []).extend(step.input_ids)
                    expected = self.reference_logits(prefixes[step.request_id])[-len(step.input_ids):]
                    torch.testing.assert_close(logits, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(len(calls), len(batches))

    def test_other_requests_and_future_tokens_cannot_change_a_causal_prefix(self):
        a = self.runner().forward_batch((
            Step("a", "prefill", (2, 3, 7, 9), (0, 1, 2, 3), (1, 3, 5, 7), (1, 3, 5, 7)),
            Step("b", "prefill", (4, 5), (0, 1), (0, 2), (0, 2))))
        b = self.runner().forward_batch((
            Step("a", "prefill", (2, 3, 11, 12), (0, 1, 2, 3), (1, 3, 5, 7), (1, 3, 5, 7)),
            Step("b", "prefill", (70, 71, 72), (0, 1, 2), (0, 2, 4), (0, 2, 4))))
        torch.testing.assert_close(a[0][:2], b[0][:2], atol=2e-6, rtol=2e-5)

    @torch.inference_mode() if AVAILABLE else lambda f: f
    def test_hf_packed_cache_adapter_matches_independent_sequences(self):
        cache = packed_reference_cache()
        inputs = torch.tensor([[2, 3, 5]])  # A has two tokens, B one.
        mask = torch.tensor([[0, -1e9, -1e9], [0, 0, -1e9], [-1e9, -1e9, 0]])
        first = self.reference(inputs, attention_mask=mask[None, None],
                               position_ids=torch.tensor([[0, 1, 0]]),
                               past_key_values=cache, use_cache=True).logits[0]
        torch.testing.assert_close(first[:2], self.reference_logits([2, 3]), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(first[2:], self.reference_logits([5]), atol=2e-6, rtol=2e-5)
        cache.permutation = torch.tensor([0, 1, 3, 2, 4])
        mask = torch.tensor([[0, 0, 0, -1e9, -1e9], [-1e9, -1e9, -1e9, 0, 0]])
        second = self.reference(torch.tensor([[7, 8]]), attention_mask=mask[None, None],
                                position_ids=torch.tensor([[2, 1]]),
                                past_key_values=cache, use_cache=True).logits[0]
        torch.testing.assert_close(second[0], self.reference_logits([2, 3, 7])[-1], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(second[1], self.reference_logits([5, 8])[-1], atol=2e-6, rtol=2e-5)

    def test_dynamic_engine_greedy_outputs_match_independent_full_recomputation(self):
        runner = self.runner()
        engine = BatchEngine(runner, 32, max_num_seqs=2, max_num_batched_tokens=8)
        a, b = engine.submit([2, 3], 1), engine.submit([4, 5, 6], 4)
        engine.step()
        c = engine.submit([7], 3)
        engine.step()
        engine.cancel(c.request_id)
        d = engine.submit([8, 9], 3)
        list(engine.run())
        for request in (a, b, c, d):
            prefix = list(request.prompt_ids)
            for actual in request.output_ids:
                expected = int(self.reference_logits(prefix)[-1].argmax())
                self.assertEqual(actual, expected, request.request_id)
                prefix.append(expected)
        self.assertEqual(c.status, Status.CANCELLED)
        self.assertEqual(engine.cache.free_count, 32)
        self.assertEqual(runner._prefixes, {})

    def test_chunked_prompt_matches_reference_and_rejects_premature_decode(self):
        runner = self.runner()
        partial = Step("a", "prefill", (2, 3), (0, 1), (7, 1), (7, 1), False)
        self.assertEqual(runner.run_batch((partial,)), (None,))
        with self.assertRaisesRegex(ValueError, "final prompt chunk"):
            runner.run_batch((Step("a", "decode", (9,), (2,), (4,), (7, 1, 4)),))
        continuation = Step("a", "prefill", (5, 6), (2, 3), (9, 4), (7, 1, 9, 4), True)
        actual = runner.forward_batch((continuation,))[0]
        torch.testing.assert_close(actual, self.reference_logits([2, 3, 5, 6])[-2:], atol=2e-6, rtol=2e-5)
        with self.assertRaises(ValueError):
            runner.forward_batch((Step("a", "prefill", (7,), (4,), (5,), (7, 1, 9, 4, 5)),))

    def test_chunked_mixed_batch_greedy_matches_full_recompute_and_reclaims_state(self):
        runner = self.runner()
        engine = BatchEngine(runner, 32, 2, 3, chunked_prefill=True)
        requests = [engine.submit([2], 4), engine.submit([3, 4, 5, 6, 7, 8, 9], 3)]
        outputs = list(engine.run())
        self.assertTrue(any(o.token_id is None for o in outputs))
        for request in requests:
            prefix = list(request.prompt_ids)
            for token in request.output_ids:
                self.assertEqual(token, int(self.reference_logits(prefix)[-1].argmax()))
                prefix.append(token)
        self.assertEqual(engine.cache.free_count, 32)
        self.assertEqual(runner._prefixes, {})
        partial = engine.submit([11, 12, 13, 14], 2)
        engine.step()
        self.assertFalse(runner._prefixes[partial.request_id].prompt_complete)
        engine.cancel(partial.request_id)
        self.assertEqual(runner._prefixes, {})
        replacement = engine.submit([15, 16], 1)
        list(engine.run())
        self.assertEqual(replacement.output_ids, [int(self.reference_logits([15, 16])[-1].argmax())])
        self.assertEqual(engine.cache.free_count, 32)

    def test_verification_cli_checks_whole_and_chunked_prefill_against_reference(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs:
                                    [2, 3, 5, 7, 11] if text.startswith("The capital") else [13, 17, 19, 23])
        for size in (None, 1, 2, 20):
            arguments = ["verify", "--model", "unused", "--device", "cpu", "--dtype", "float32"]
            if size is not None:
                arguments += ["--prefill-chunk-size", str(size)]
            output = io.StringIO()
            with self.subTest(size=size), patch("sys.argv", arguments), contextlib.redirect_stdout(output), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=self.reference), \
                    patch.object(ModelRunner, "from_pretrained", return_value=self.runner()):
                verify.main()
            report = json.loads(output.getvalue())
            self.assertTrue(report["passed"])
            self.assertEqual(report["steps"][-1]["context_tokens"], 9)
            prefill = [r for r in report["steps"] if r["phase"] == "prefill"]
            self.assertEqual(sum(len(r["input_ids"]) for r in prefill), 5)
            self.assertEqual(sum(r["is_last_prompt_chunk"] for r in prefill), 1)

    def test_aliases_unwritten_prefixes_and_live_restarts_are_rejected_before_execution(self):
        runner = self.runner()
        a = Step("a", "prefill", (2,), (0,), (0,), (0,))
        b = Step("b", "prefill", (3,), (0,), (1,), (1,))
        runner.forward_batch((a, b))
        bad_batches = [(), (a, a), (a,),
                       (Step("c", "prefill", (4,), (0,), (1,), (1,)),),
                       (Step("missing", "decode", (4,), (1,), (3,), (2, 3)),),
                       (Step("a", "decode", (4,), (1,), (2,), (0, 2)),
                        Step("b", "decode", (5,), (1,), (2,), (1, 2)))]
        calls = []
        with runner.model.register_forward_hook(lambda *args: calls.append(1)):
            for batch in bad_batches:
                with self.subTest(batch=batch), self.assertRaises(ValueError):
                    runner.forward_batch(batch)
        self.assertEqual(calls, [])
        runner.release("a")
        with self.assertRaises(ValueError):
            runner.forward_batch((Step("a", "decode", (5,), (1,), (2,), (0, 2)),))

    def test_model_failure_aborts_batch_without_leaking_prefix_metadata(self):
        runner = self.runner()
        engine = BatchEngine(runner, 32, max_num_seqs=2)
        a, b = engine.submit([2], 3), engine.submit([3], 3)
        waiting = engine.submit([4], 1)
        engine.step()

        def fail_after_layer(*args):
            raise RuntimeError("after first layer KV write")

        with runner.model.model.layers[0].register_forward_hook(fail_after_layer):
            with self.assertRaisesRegex(RuntimeError, "after first layer"):
                engine.step()
        self.assertEqual([len(a.output_ids), len(b.output_ids)], [1, 1])
        self.assertEqual([a.status, b.status], [Status.FAILED, Status.FAILED])
        self.assertEqual(runner._prefixes, {})
        self.assertEqual(engine.cache.free_count, 32)
        list(engine.run())
        self.assertEqual(waiting.status, Status.FINISHED)


if __name__ == "__main__":
    unittest.main()
