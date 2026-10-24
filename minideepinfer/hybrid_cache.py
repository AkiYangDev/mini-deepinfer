"""Qwen3.5: token-addressed KV plus request-owned convolution/recurrent state."""

from types import SimpleNamespace

from .tensor_cache import TensorKVCache


class HybridCache(TensorKVCache):
    def __init__(self, config, capacity, device, dtype):
        self.full_layers = {layer: index for index, layer in enumerate(
            i for i, kind in enumerate(config.layer_types) if kind == "full_attention")}
        compact = SimpleNamespace(num_hidden_layers=len(self.full_layers),
                                  num_key_value_heads=config.num_key_value_heads,
                                  head_dim=config.head_dim)
        super().__init__(compact, capacity, device, dtype)
        # Each entry is (last K-1 projected inputs, FP32 delta-rule matrix).
        # Allocate on first use; release with the request, including partial prompts.
        self.linear_states = {}

    def write(self, layer, slots, keys, values):
        super().write(self.full_layers[layer], slots, keys, values)

    def read(self, layer, slots):
        return super().read(self.full_layers[layer], slots)

    def release(self, request_id):
        for key in [key for key in self.linear_states if key[0] == request_id]:
            del self.linear_states[key]
