"""Optional validation adapter; never used by the engine or generation path.

Transformers 5.12.1 defaults to chunked prefill and BF16 recurrent storage.
Use its independent recurrent equations with FP32 storage for a like-for-like
comparison with this project's token scan. Native compatibility is a separate
check; this adapter must be selected explicitly in the verification CLI.
"""

from importlib.metadata import version

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule


class FP32RecurrentCache(DynamicCache):
    def update_recurrent_state(self, state, layer_idx, **kwargs):
        if state.dtype != torch.float32:
            raise ValueError("the reference recurrent kernel must produce FP32 state")
        layer = self.layers[layer_idx]
        if not layer.is_recurrent_states_initialized:
            layer.recurrent_states = torch.zeros_like(state)
            layer.is_recurrent_states_initialized = True
        layer.recurrent_states.copy_(state)
        return layer.recurrent_states


class RecurrentFP32Reference:
    """Keep HF projections/norms/attention/MLP; select its recurrent scan/cache.

    Changes are scoped to this reference model's linear layers during a call.
    The wrapper is for synchronous verification, not concurrent model serving.
    """

    def __init__(self, model):
        if version("transformers") != "5.12.1":
            raise ValueError("recurrent-fp32 reference requires Transformers 5.12.1")
        if getattr(model.config, "model_type", None) != "qwen3_5_text":
            raise ValueError("recurrent-fp32 reference requires a Qwen3.5 text model")
        self.model = model

    def __call__(self, *args, **kwargs):
        if kwargs.get("use_cache", False):
            past = kwargs.get("past_key_values")
            if past is not None and not isinstance(past, FP32RecurrentCache):
                raise ValueError("cannot mix native and FP32 reference caches")
            if past is None:
                kwargs["past_key_values"] = FP32RecurrentCache(config=self.model.config)

        def recurrent(query, key, value, **options):
            options.pop("cu_seqlens", None)
            return torch_recurrent_gated_delta_rule(query, key, value, **options)

        saved = []
        try:
            for layer in self.model.model.layers:
                if hasattr(layer, "linear_attn"):
                    mixer = layer.linear_attn
                    saved.append((mixer, mixer.chunk_gated_delta_rule))
                    mixer.chunk_gated_delta_rule = recurrent
            return self.model(*args, **kwargs)
        finally:
            for mixer, original in saved:
                mixer.chunk_gated_delta_rule = original
