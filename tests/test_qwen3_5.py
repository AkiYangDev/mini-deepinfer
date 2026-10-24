"""Hybrid-state correctness against independent Transformers Qwen3.5 math."""

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = False
if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"):
    import transformers
    AVAILABLE = hasattr(transformers, "Qwen3_5ForCausalLM")

if AVAILABLE:
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM as ReferenceModel
    from minideepinfer import BatchEngine, Engine, Step, Status
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM


@unittest.skipUnless(AVAILABLE, "Qwen3.5 numerical tests require Transformers with Qwen3.5 support")
class Qwen35Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(2035)
        cls.config = Qwen3_5Config(97, 32, 48, 4, 2, 1, 16, 2, 4, 8, 8,
                                  ("linear_attention",) * 3 + ("full_attention",),
                                  max_position_embeddings=128, partial_rotary_factor=0.5)
        values = asdict(cls.config)
        values["layer_types"] = list(values["layer_types"])
        values["rope_parameters"] = {"rope_type": "default", "rope_theta": values.pop("rope_theta"),
                                     "partial_rotary_factor": values.pop("partial_rotary_factor"),
                                     "mrope_section": [1, 1, 2]}
        cls.reference_config = Qwen3_5TextConfig(**values)
        cls.reference_config._attn_implementation = "eager"
        cls.reference = ReferenceModel(cls.reference_config).eval()

    def model(self):
        model = Qwen3_5ForCausalLM(self.config)
        model.load_state_dict(self.reference.state_dict(), strict=True)
        return model.eval()

    def runner(self):
        return ModelRunner(self.model(), 64)

    @torch.inference_mode() if AVAILABLE else lambda f: f
    def expected(self, ids):
        return self.reference(torch.tensor([ids]), use_cache=False).logits[0]

    def test_whole_chunked_and_cached_reference_with_noncontiguous_slots(self):
        slots = (9, 2, 11, 4, 0, 7, 5, 13)
        prompt, continuation = [2, 3, 5, 7, 11], [13, 17, 19]
        for size in (1, 2, 5):
            runner, prefix, past = self.runner(), [], None
            work = [("prefill", prompt[i:i + size], i + size >= len(prompt))
                    for i in range(0, len(prompt), size)]
            work += [("decode", [token], True) for token in continuation]
            with torch.inference_mode():
                for phase, tokens, last in work:
                    start = len(prefix)
                    prefix.extend(tokens)
                    end = len(prefix)
                    step = Step("a", phase, tuple(tokens), tuple(range(start, end)),
                                slots[start:end], slots[:end], last)
                    actual = runner.forward_batch((step,))[0]
                    ref = self.reference(torch.tensor([tokens]), past_key_values=past, use_cache=True)
                    past = ref.past_key_values
                    with self.subTest(size=size, length=end):
                        torch.testing.assert_close(actual, ref.logits[0], atol=3e-6, rtol=3e-5)
                        torch.testing.assert_close(actual, self.expected(prefix)[-len(tokens):], atol=3e-6, rtol=3e-5)
            self.assertEqual(runner.cache.keys.shape[0], 1)
            self.assertEqual(len(runner.cache.linear_states), 3)
            runner.release("a")
            self.assertEqual(runner.cache.linear_states, {})

    def test_mixed_batch_reorders_states_and_reuses_released_slots(self):
        runner, prefixes = self.runner(), {}
        batches = [
            (Step("a", "prefill", (2, 3), (0, 1), (7, 1), (7, 1), False),
             Step("b", "prefill", (5,), (0,), (0,), (0,))),
            (Step("b", "decode", (8,), (1,), (4,), (0, 4)),
             Step("a", "prefill", (9, 10), (2, 3), (9, 3), (7, 1, 9, 3))),
            (Step("a", "decode", (13,), (4,), (5,), (7, 1, 9, 3, 5)),
             Step("c", "prefill", (11, 12), (0, 1), (0, 4), (0, 4))),
        ]
        for i, batch in enumerate(batches):
            if i == 2:
                runner.release("b")
            for step, logits in zip(batch, runner.forward_batch(batch)):
                prefixes.setdefault(step.request_id, []).extend(step.input_ids)
                torch.testing.assert_close(logits, self.expected(prefixes[step.request_id])[-len(step.input_ids):],
                                           atol=3e-6, rtol=3e-5)
        self.assertEqual({key[0] for key in runner.cache.linear_states}, {"a", "c"})

    def test_bfloat16_normalization_and_recurrent_scan_match_pinned_reference(self):
        reference = ReferenceModel(self.reference_config).eval().to(torch.bfloat16)
        reference.load_state_dict(self.reference.state_dict())
        model = self.model().to(torch.bfloat16)
        runner = ModelRunner(model, 64)
        with torch.inference_mode():
            for ids in ((2, 3), (5, 7, 11)):
                runner.release("a")
                n = len(ids)
                actual = runner.forward(Step("a", "prefill", ids, tuple(range(n)), tuple(range(n)), tuple(range(n))))
                expected = reference(torch.tensor([ids]), use_cache=False).logits[0]
                torch.testing.assert_close(actual, expected, atol=5e-4, rtol=1e-3)

    def test_future_and_other_requests_cannot_change_prefix(self):
        outputs = []
        for future, other in (((7, 9), (4, 5)), ((11, 12), (70, 71, 72))):
            ids = (2, 3) + future
            steps = (Step("a", "prefill", ids, tuple(range(4)), (1, 3, 5, 7), (1, 3, 5, 7)),
                     Step("b", "prefill", other, tuple(range(len(other))),
                          (0, 2, 4)[:len(other)], (0, 2, 4)[:len(other)]))
            outputs.append(self.runner().forward_batch(steps)[0][:2])
        torch.testing.assert_close(*outputs, atol=3e-6, rtol=3e-5)

    def test_finish_cancel_and_failed_batch_release_both_state_kinds(self):
        runner = self.runner()
        engine = BatchEngine(runner, 64, max_num_seqs=2, max_num_batched_tokens=2, chunked_prefill=True)
        partial = engine.submit([2, 3, 5, 7, 11], 3)
        engine.step()
        self.assertTrue(runner.cache.linear_states)
        engine.cancel(partial.request_id)
        self.assertEqual(runner.cache.linear_states, {})
        a, b = engine.submit([13, 17], 3), engine.submit([19], 2)
        engine.step()

        def fail(*args):
            raise RuntimeError("after recurrent state write")

        with runner.model.model.layers[0].register_forward_hook(fail), self.assertRaisesRegex(RuntimeError, "recurrent"):
            engine.step()
        self.assertEqual(a.status, Status.FAILED)
        self.assertEqual(b.status, Status.FAILED)
        self.assertEqual(runner.cache.linear_states, {})
        self.assertEqual(runner._prefixes, {})
        self.assertEqual(engine.cache.free_count, 64)
        replacement = engine.submit([23, 29], 2)
        list(engine.run())
        self.assertEqual(replacement.status, Status.FINISHED)
        self.assertEqual(runner.cache.linear_states, {})
        self.assertEqual(engine.cache.free_count, 64)

    def test_single_engine_reclaims_recurrent_state_on_finish_and_cancel(self):
        runner = self.runner()
        engine = Engine(runner, 64)
        a = engine.submit([2, 3], 2)
        list(engine.run())
        self.assertEqual(a.output_ids[0], int(self.expected([2, 3])[-1].argmax()))
        self.assertEqual(runner.cache.linear_states, {})
        b = engine.submit([5], 4)
        engine.step()
        self.assertTrue(runner.cache.linear_states)
        engine.cancel(b.request_id)
        self.assertEqual(runner.cache.linear_states, {})

    def test_multimodal_checkpoint_loads_only_text_and_preserves_tied_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"model_type": "qwen3_5", "tie_word_embeddings": True,
                      "text_config": self.reference_config.to_dict()}
            (root / "config.json").write_text(json.dumps(config))
            state = {name.replace("model.", "model.language_model.", 1): value.clone().contiguous()
                     for name, value in self.reference.state_dict().items() if name != "lm_head.weight"}
            state["model.visual.ignored.weight"] = torch.ones(1)
            state["mtp.ignored.weight"] = torch.ones(1)
            filename = "part.safetensors"
            save_file(state, root / filename)
            weight_map = {name: filename for name in state}
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": weight_map}))
            runner = ModelRunner.from_pretrained(root, "cpu", torch.float32, 16)
            self.assertIs(runner.model.lm_head.weight, runner.model.model.embed_tokens.weight)
            actual = runner.forward(Step("a", "prefill", (2, 3), (0, 1), (0, 1), (0, 1)))
            torch.testing.assert_close(actual, self.expected([2, 3]), atol=3e-6, rtol=3e-5)
            state.pop("model.language_model.layers.0.linear_attn.A_log")
            save_file(state, root / filename)
            with self.assertRaisesRegex(ValueError, "missing checkpoint"):
                Qwen3_5ForCausalLM.from_pretrained(root)
            index.write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "outside"):
                Qwen3_5ForCausalLM.from_pretrained(root)

    def test_configuration_rejects_unsupported_variants(self):
        for changes in ({"layer_types": ("full_attention",)}, {"partial_rotary_factor": 0.01},
                        {"linear_num_value_heads": 3}, {"linear_conv_kernel_dim": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)
        for model_type in ("qwen3_5_moe", "qwen3"):
            with self.assertRaises(ValueError):
                Qwen3_5Config.from_dict({"model_type": model_type})


if __name__ == "__main__":
    unittest.main()
