# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Normalization/rotary equations adapted from their Apache-2.0 Qwen2 reference.
# Source and simplifications are documented in docs/REFERENCES.md.
"""Qwen2 forward, with explicit attention, packed requests, and external KV."""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self):
        dimensions = (self.vocab_size, self.hidden_size, self.intermediate_size,
                      self.num_hidden_layers, self.num_attention_heads,
                      self.num_key_value_heads, self.max_position_embeddings)
        if any(type(size) is not int or size < 1 for size in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if self.hidden_size % self.num_attention_heads or self.head_dim % 2:
            raise ValueError("hidden_size must divide into even-sized attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be a multiple of KV heads")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("normalization epsilon and RoPE base must be positive")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, data):
        if data.get("model_type", "qwen2") != "qwen2":
            raise ValueError("only the Qwen2 architecture is implemented")
        unsupported = (
            data.get("use_sliding_window", False), data.get("rope_scaling"),
            data.get("tie_word_embeddings", False), data.get("use_mrope", False),
            data.get("hidden_act", "silu") != "silu",
            data.get("attention_bias", True) is not True,
        )
        if any(unsupported):
            raise ValueError("requires full attention, default RoPE, SiLU, QKV bias, and untied embeddings")
        config = cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
        if data.get("head_dim", config.head_dim) != config.head_dim:
            raise ValueError("custom head_dim is not implemented")
        # Newer config writers may store RoPE parameters in a separate object.
        rope = data.get("rope_parameters") or {}
        if rope and (rope.get("rope_type", "default") != "default"
                     or rope.get("rope_theta", config.rope_theta) != config.rope_theta):
            raise ValueError("unsupported RoPE parameters")
        return config


class RMSNorm(nn.Module):
    def __init__(self, width, eps, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return value.to(x.dtype) * self.weight


def rotary(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-right, left), dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.config = config
        width, kv_width = config.hidden_size, config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(width, width, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(width, kv_width, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(width, kv_width, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(width, width, bias=False, device=device, dtype=dtype)

    def forward(self, x, inputs, cache, layer, cos, sin):
        c = self.config
        q = self.q_proj(x).view(-1, c.num_attention_heads, c.head_dim)
        k = self.k_proj(x).view(-1, c.num_key_value_heads, c.head_dim)
        v = self.v_proj(x).view(-1, c.num_key_value_heads, c.head_dim)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        cache.write(layer, inputs.slot_mapping, k, v)
        k, v = cache.read(layer, inputs.context_slots)
        groups = c.num_attention_heads // c.num_key_value_heads
        k = k.repeat_interleave(groups, dim=1).transpose(0, 1)
        v = v.repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = (q.transpose(0, 1) @ k.transpose(1, 2)) * (c.head_dim ** -0.5)
        # Query positions are absolute: decode must see all preceding KV.
        blocked = inputs.attention_mask
        if blocked is None:
            key_positions = torch.arange(k.shape[1], device=x.device)
            blocked = key_positions.unsqueeze(0) > inputs.positions.unsqueeze(1)
        scores = scores.masked_fill(blocked.unsqueeze(0), torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        output = (probabilities @ v).transpose(0, 1).reshape(-1, c.hidden_size)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.self_attn = Attention(config, device, dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.mlp = MLP(config, device, dtype)

    def forward(self, x, inputs, cache, layer, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), inputs, cache, layer, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(DecoderLayer(config, device, dtype) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.config = config
        # Names match the checkpoint; model execution is implemented here.
        self.model = Qwen2Model(config, device, dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, inputs, cache):
        x = self.model.embed_tokens(inputs.input_ids)
        indices = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (self.config.rope_theta ** (indices / self.config.head_dim))
        frequencies = inputs.positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = angles.cos().to(x.dtype).unsqueeze(1), angles.sin().to(x.dtype).unsqueeze(1)
        for layer, decoder in enumerate(self.model.layers):
            x = decoder(x, inputs, cache, layer, cos, sin)
        return self.lm_head(self.model.norm(x))

    @classmethod
    def from_pretrained(cls, directory, dtype=torch.bfloat16):
        """Load local safetensors only; no remote code or pickle checkpoints."""
        from safetensors import safe_open

        root = Path(directory).resolve()
        config = Qwen2Config.from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
        index = root / "model.safetensors.index.json"
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index.exists() else None
        filenames = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
        paths = [(root / name).resolve() for name in filenames]
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError("checkpoint index points outside the model directory")
        model = cls(config, device="meta", dtype=dtype)
        model.to_empty(device="cpu")
        parameters = dict(model.named_parameters())
        if weight_map is not None and set(weight_map) != set(parameters):
            raise ValueError("checkpoint index does not match model parameter names")
        loaded = set()
        with torch.no_grad():
            for path in paths:
                with safe_open(path, framework="pt", device="cpu") as shard:
                    for name in shard.keys():
                        if name not in parameters or name in loaded:
                            raise ValueError(f"unexpected or duplicate parameter: {name}")
                        if weight_map is not None and (root / weight_map[name]).resolve() != path:
                            raise ValueError(f"parameter is in the wrong shard: {name}")
                        value = shard.get_tensor(name)
                        if value.shape != parameters[name].shape:
                            raise ValueError(f"shape mismatch for {name}: {tuple(value.shape)}")
                        parameters[name].copy_(value)
                        loaded.add(name)
        if loaded != set(parameters):
            raise ValueError(f"missing checkpoint parameters: {sorted(set(parameters) - loaded)}")
        return model.eval()
