"""Reference tensor storage addressed by the engine's logical token slots."""

import torch


class TensorKVCache:
    def __init__(self, config, capacity, device, dtype):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be a positive integer")
        self.capacity = capacity
        shape = (config.num_hidden_layers, capacity, config.num_key_value_heads, config.head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)

    def write(self, layer, slots, keys, values):
        # RoPE has already been applied to keys; values are unrotated.
        self.keys[layer].index_copy_(0, slots, keys)
        self.values[layer].index_copy_(0, slots, values)

    def read(self, layer, slots):
        # Gather is intentionally explicit. This is not a paged attention kernel.
        return (self.keys[layer].index_select(0, slots),
                self.values[layer].index_select(0, slots))
