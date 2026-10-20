"""Numerical tests use tiny random weights; no checkpoint download or NPU."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))
if AVAILABLE:
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen2Config as ReferenceConfig, Qwen2ForCausalLM as ReferenceModel
    from minideepinfer import Engine, Status, Step
    from minideepinfer.model_runner import ModelRunner
    from minideepinfer.models.qwen2 import Qwen2Config, Qwen2ForCausalLM


@unittest.skipUnless(AVAILABLE, "install the model extra for numerical tests")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        cls.config = Qwen2Config(97, 32, 64, 2, 4, 2, max_position_embeddings=64)
        reference_config = ReferenceConfig(**asdict(cls.config), tie_word_embeddings=False,
                                           use_sliding_window=False, attention_dropout=0.0)
        reference_config._attn_implementation = "eager"
        cls.reference = ReferenceModel(reference_config).eval()

    def new_model(self):
        model = Qwen2ForCausalLM(self.config)
        model.load_state_dict(self.reference.state_dict(), strict=True)
        return model.eval()

    @torch.inference_mode() if AVAILABLE else lambda f: f
    def reference_logits(self, tokens):
        return self.reference(torch.tensor([tokens]), use_cache=False).logits[0]

    def test_prefill_and_decode_match_reference_with_noncontiguous_slots(self):
        runner = ModelRunner(self.new_model(), 12)
        prompt, forced = [1, 5, 9], [4, 7, 11]
        slots = (8, 1, 10, 3, 0, 6)
        step = Step("a", "prefill", tuple(prompt), (0, 1, 2), slots[:3], slots[:3])
        actual = runner.forward(step)
        torch.testing.assert_close(actual, self.reference_logits(prompt), atol=2e-6, rtol=2e-5)
        for index, token in enumerate(forced, start=len(prompt)):
            prompt.append(token)
            step = Step("a", "decode", (token,), (index,), (slots[index],), slots[:index + 1])
            actual = runner.forward(step)[0]
            torch.testing.assert_close(actual, self.reference_logits(prompt)[-1], atol=2e-6, rtol=2e-5)

    def test_cache_matches_full_recomputation_after_dirty_slot_reuse(self):
        cached = ModelRunner(self.new_model(), 10)
        for request_id, prompt in (("a", [2, 3]), ("b", [16, 8, 4])):
            prefix = prompt[:]
            n = len(prefix)
            cached.forward(Step(request_id, "prefill", tuple(prefix), tuple(range(n)),
                                tuple(range(n)), tuple(range(n))))
            for token in (6, 12):
                prefix.append(token)
                end = len(prefix)
                actual = cached.forward(Step(request_id, "decode", (token,), (end - 1,),
                                             (end - 1,), tuple(range(end))))
                fresh = ModelRunner(self.new_model(), 10)
                expected = fresh.forward(Step("fresh", "prefill", tuple(prefix), tuple(range(end)),
                                              tuple(range(end)), tuple(range(end))))
                torch.testing.assert_close(actual[-1], expected[-1], atol=2e-6, rtol=2e-5)

    def test_engine_greedy_generation_matches_reference_and_reclaims_slots(self):
        engine = Engine(ModelRunner(self.new_model(), 8), 8)
        for prompt in ([1, 3, 5], [8]):
            prefix, expected = list(prompt), []
            for _ in range(4):
                token = int(self.reference_logits(prefix)[-1].argmax())
                prefix.append(token)
                expected.append(token)
            request = engine.submit(prompt, max_new_tokens=4)
            list(engine.run())
            self.assertEqual(request.output_ids, expected)
            self.assertEqual(request.status, Status.FINISHED)
            self.assertEqual(engine.cache.free_count, 8)

    def test_causal_prefix_is_unchanged_by_future_tokens(self):
        runner = ModelRunner(self.new_model(), 8)
        short = runner.forward(Step("short", "prefill", (2, 3), (0, 1), (0, 1), (0, 1)))
        long = runner.forward(Step("long", "prefill", (2, 3, 17, 19), (0, 1, 2, 3),
                                   (0, 1, 2, 3), (0, 1, 2, 3)))
        torch.testing.assert_close(short, long[:2], atol=2e-6, rtol=2e-5)

    def test_rejects_unwritten_prefix_wrong_request_and_slot_aliasing(self):
        runner = ModelRunner(self.new_model(), 8)
        invalid = [Step("a", "decode", (2,), (1,), (1,), (0, 1)),
                   Step("a", "prefill", (2, 3), (0, 1), (0, 0), (0, 0)),
                   Step("a", "prefill", (97,), (0,), (0,), (0,))]
        for step in invalid:
            with self.subTest(step=step), self.assertRaises(ValueError):
                runner.forward(step)
        runner.forward(Step("a", "prefill", (2,), (0,), (0,), (0,)))
        with self.assertRaises(ValueError):
            runner.forward(Step("b", "decode", (3,), (1,), (1,), (0, 1)))

    def test_local_safetensors_loading_and_missing_weight_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(asdict(self.config)))
            state = {k: v.contiguous() for k, v in self.reference.state_dict().items()}
            save_file(state, root / "model.safetensors")
            model = Qwen2ForCausalLM.from_pretrained(root, dtype=torch.float32)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, state[name], atol=0, rtol=0)
            state.pop("lm_head.weight")
            save_file(state, root / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "missing checkpoint"):
                Qwen2ForCausalLM.from_pretrained(root)

    def test_sharded_loading_and_index_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(asdict(self.config)))
            state = self.reference.state_dict()
            names = sorted(state)
            first, second = names[::2], names[1::2]
            mapping = {name: "part1.safetensors" if name in first else "part2.safetensors" for name in names}
            for filename, selected in (("part1.safetensors", first), ("part2.safetensors", second)):
                save_file({name: state[name].contiguous() for name in selected}, root / filename)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": mapping}))
            model = Qwen2ForCausalLM.from_pretrained(root, dtype=torch.float32)
            torch.testing.assert_close(model.lm_head.weight, state["lm_head.weight"], atol=0, rtol=0)
            mapping[names[0]] = "../outside.safetensors"
            index.write_text(json.dumps({"weight_map": mapping}))
            with self.assertRaisesRegex(ValueError, "outside the model"):
                Qwen2ForCausalLM.from_pretrained(root)

    def test_rejects_unsupported_architecture_options(self):
        for extra in ({"model_type": "deepseek_v2"}, {"use_sliding_window": True},
                      {"rope_scaling": {"rope_type": "linear", "factor": 2}},
                      {"tie_word_embeddings": True}, {"head_dim": 16}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                Qwen2Config.from_dict(asdict(self.config) | extra)


if __name__ == "__main__":
    unittest.main()
