# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Equations adapted from Transformers' Qwen3.5 reference; see docs/REFERENCES.md.
"""Text-only Qwen3.5 with an explicit, token-by-token Gated DeltaNet scan.

Projections/MLPs use packed tokens. DeltaNet scans each request separately;
this favors readable state ownership over a fused, high-throughput kernel.
"""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .qwen2 import MLP, rotary


@dataclass(frozen=True)
class Qwen3_5Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    layer_types: tuple[str, ...]
    linear_conv_kernel_dim: int = 4
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    tie_word_embeddings: bool = True

    def __post_init__(self):
        sizes = (self.vocab_size, self.hidden_size, self.intermediate_size, self.num_hidden_layers,
                 self.num_attention_heads, self.num_key_value_heads, self.head_dim,
                 self.linear_num_key_heads, self.linear_num_value_heads, self.linear_key_head_dim,
                 self.linear_value_head_dim, self.linear_conv_kernel_dim, self.max_position_embeddings)
        if any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError("model dimensions must be positive integers")
        if (self.num_attention_heads % self.num_key_value_heads
                or self.linear_num_value_heads % self.linear_num_key_heads):
            raise ValueError("query/value heads must be multiples of their key heads")
        if (len(self.layer_types) != self.num_hidden_layers
                or set(self.layer_types) - {"linear_attention", "full_attention"}):
            raise ValueError("layer_types must describe every decoder layer")
        dim = int(self.head_dim * self.partial_rotary_factor)
        if not 0 < self.partial_rotary_factor <= 1 or dim < 2 or dim % 2:
            raise ValueError("partial RoPE must have a positive even dimension")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0 or type(self.tie_word_embeddings) is not bool:
            raise ValueError("invalid normalization, RoPE, or embedding configuration")

    @classmethod
    def from_dict(cls, data):
        if data.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError("requires a dense Qwen3.5 checkpoint")
        text = data.get("text_config", data)
        rope = text.get("rope_parameters") or {}
        if (text.get("attention_bias", False) or text.get("hidden_act", "silu") != "silu"
                or text.get("attn_output_gate", True) is not True
                or rope.get("rope_type", "default") != "default"):
            raise ValueError("requires bias-free gated attention, SiLU, and default RoPE")
        values = {f.name: text[f.name] for f in fields(cls) if f.name in text}
        values["layer_types"] = tuple(text["layer_types"])
        values["rope_theta"] = rope.get("rope_theta", 10000000.0)
        values["partial_rotary_factor"] = rope.get("partial_rotary_factor", 0.25)
        values["tie_word_embeddings"] = data.get("tie_word_embeddings", text.get("tie_word_embeddings", True))
        return cls(**values)


class RMSNorm(nn.Module):
    def __init__(self, width, eps, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * (1.0 + self.weight.float())).to(x.dtype)


class GatedAttention(nn.Module):
    def __init__(self, c, device=None, dtype=None):
        super().__init__()
        self.config = c
        width, kv_width = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, width * 2, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(c.hidden_size, kv_width, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(c.hidden_size, kv_width, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(width, c.hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps, device, dtype)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps, device, dtype)

    def forward(self, x, inputs, cache, layer, cos, sin):
        c = self.config
        q, gate = self.q_proj(x).view(-1, c.num_attention_heads, c.head_dim * 2).chunk(2, -1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(-1, c.num_key_value_heads, c.head_dim))
        v = self.v_proj(x).view(-1, c.num_key_value_heads, c.head_dim)
        dim = cos.shape[-1]
        q = torch.cat((rotary(q[..., :dim], cos, sin), q[..., dim:]), -1)
        k = torch.cat((rotary(k[..., :dim], cos, sin), k[..., dim:]), -1)
        cache.write(layer, inputs.slot_mapping, k, v)
        k, v = cache.read(layer, inputs.context_slots)
        groups = c.num_attention_heads // c.num_key_value_heads
        k, v = (t.repeat_interleave(groups, 1).transpose(0, 1) for t in (k, v))
        scores = (q.transpose(0, 1) @ k.transpose(-1, -2)) * (c.head_dim ** -0.5)
        blocked = inputs.attention_mask
        if blocked is None:
            blocked = torch.arange(k.shape[1], device=x.device)[None, :] > inputs.positions[:, None]
        scores = scores.masked_fill(blocked[None, :, :], torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, -1, dtype=torch.float32).to(x.dtype)
        output = (probs @ v).transpose(0, 1).reshape(x.shape[0], -1)
        return self.o_proj(output * gate.reshape(x.shape[0], -1).sigmoid())


class GatedDeltaNet(nn.Module):
    def __init__(self, c, device=None, dtype=None):
        super().__init__()
        self.config = c
        kd = c.linear_num_key_heads * c.linear_key_head_dim
        vd = c.linear_num_value_heads * c.linear_value_head_dim
        channels = 2 * kd + vd
        self.in_proj_qkv = nn.Linear(c.hidden_size, channels, bias=False, device=device, dtype=dtype)
        self.in_proj_z = nn.Linear(c.hidden_size, vd, bias=False, device=device, dtype=dtype)
        self.in_proj_a = nn.Linear(c.hidden_size, c.linear_num_value_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_b = nn.Linear(c.hidden_size, c.linear_num_value_heads, bias=False, device=device, dtype=dtype)
        self.conv1d = nn.Conv1d(channels, channels, c.linear_conv_kernel_dim, groups=channels,
                                bias=False, device=device, dtype=dtype)
        self.dt_bias = nn.Parameter(torch.ones(c.linear_num_value_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.zeros(c.linear_num_value_heads, device=device, dtype=dtype))
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(c.linear_value_head_dim, device=device, dtype=dtype))
        self.out_proj = nn.Linear(vd, c.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, inputs, cache, layer):
        c = self.config
        projected = self.in_proj_qkv(x)
        z = self.in_proj_z(x).view(-1, c.linear_num_value_heads, c.linear_value_head_dim)
        beta = self.in_proj_b(x).sigmoid().float()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        results, offset = [], 0
        for step in inputs.steps:
            end = offset + len(step.input_ids)
            key = (step.request_id, layer)
            history, state = cache.linear_states.get(key, (None, None))
            raw = projected[offset:end].transpose(0, 1)
            if history is None:
                history = raw.new_zeros(raw.shape[0], c.linear_conv_kernel_dim - 1)
            combined = torch.cat((history, raw), -1)
            # Depthwise causal convolution, written as K shifted products. This
            # also avoids requiring a platform-specific causal-conv extension.
            mixed = sum(combined[:, i:i + end - offset].float() * self.conv1d.weight[:, 0, i:i + 1].float()
                        for i in range(c.linear_conv_kernel_dim)).to(raw.dtype)
            mixed = F.silu(mixed.transpose(0, 1))
            kd = c.linear_num_key_heads * c.linear_key_head_dim
            q, k, v = mixed.split((kd, kd, c.linear_num_value_heads * c.linear_value_head_dim), -1)
            q, k = (t.view(-1, c.linear_num_key_heads, c.linear_key_head_dim) for t in (q, k))
            # Match the pinned Transformers 5.12.1 reference: normalize in the
            # projection dtype, then promote the recurrent scan to FP32.
            q, k = (t * torch.rsqrt(t.square().sum(-1, keepdim=True) + 1e-6) for t in (q, k))
            groups = c.linear_num_value_heads // c.linear_num_key_heads
            q, k = (t.repeat_interleave(groups, 1).float() for t in (q, k))
            q = q * (c.linear_key_head_dim ** -0.5)
            v = v.view(-1, c.linear_num_value_heads, c.linear_value_head_dim).float()
            if state is None:
                state = v.new_zeros(c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim)
            outputs = []
            for index in range(end - offset):
                state = state * decay[offset + index].exp()[:, None, None]
                prediction = (state * k[index, :, :, None]).sum(-2)
                correction = (v[index] - prediction) * beta[offset + index, :, None]
                state = state + k[index, :, :, None] * correction[:, None, :]
                outputs.append((state * q[index, :, :, None]).sum(-2))
            keep = c.linear_conv_kernel_dim - 1
            cache.linear_states[key] = (combined[:, -keep:].clone() if keep else combined[:, :0].clone(), state)
            results.append(torch.stack(outputs).to(x.dtype))
            offset = end
        output = torch.cat(results)
        normalized = output.float() * torch.rsqrt(output.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)
        normalized = self.norm.weight * normalized.to(x.dtype)
        gated = (normalized * F.silu(z.float())).to(x.dtype)
        return self.out_proj(gated.reshape(x.shape[0], -1))


class DecoderLayer(nn.Module):
    def __init__(self, c, layer, device=None, dtype=None):
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps, device, dtype)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps, device, dtype)
        self.mlp = MLP(c, device, dtype)
        self.linear = c.layer_types[layer] == "linear_attention"
        if self.linear:
            self.linear_attn = GatedDeltaNet(c, device, dtype)
        else:
            self.self_attn = GatedAttention(c, device, dtype)

    def forward(self, x, inputs, cache, layer, cos, sin):
        normed = self.input_layernorm(x)
        output = (self.linear_attn(normed, inputs, cache, layer) if self.linear else
                  self.self_attn(normed, inputs, cache, layer, cos, sin))
        x = x + output
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.model.layers = nn.ModuleList(DecoderLayer(config, i, device, dtype) for i in range(config.num_hidden_layers))
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, inputs, cache):
        if not inputs.steps:
            raise ValueError("Qwen3.5 requires request boundaries for recurrent state")
        x = self.model.embed_tokens(inputs.input_ids)
        dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        indices = torch.arange(0, dim, 2, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (self.config.rope_theta ** (indices / dim))
        frequencies = inputs.positions.float()[:, None] * inv_freq[None, :]
        angles = torch.cat((frequencies, frequencies), -1)
        cos, sin = (t.to(x.dtype)[:, None, :] for t in (angles.cos(), angles.sin()))
        for layer, decoder in enumerate(self.model.layers):
            x = decoder(x, inputs, cache, layer, cos, sin)
        return self.lm_head(self.model.norm(x))

    @classmethod
    def from_pretrained(cls, directory, dtype=torch.bfloat16):
        """Load text weights only; deliberately omit vision and MTP modules."""
        from safetensors import safe_open

        root = Path(directory).resolve()
        data = json.loads((root / "config.json").read_text(encoding="utf-8"))
        config = Qwen3_5Config.from_dict(data)
        index = root / "model.safetensors.index.json"
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index.exists() else None
        names = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
        paths = [(root / name).resolve() for name in names]
        if any(not p.is_relative_to(root) for p in paths):
            raise ValueError("checkpoint index points outside the model directory")
        model = cls(config, device="meta", dtype=dtype)
        model.to_empty(device="cpu")
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        parameters, loaded = dict(model.named_parameters()), set()
        multimodal = data.get("model_type") == "qwen3_5"
        with torch.no_grad():
            for path in paths:
                with safe_open(path, framework="pt", device="cpu") as shard:
                    for source in shard.keys():
                        if weight_map is not None and (root / weight_map[source]).resolve() != path:
                            raise ValueError(f"parameter is in the wrong shard: {source}")
                        if multimodal and source.startswith(("model.visual.", "mtp.")):
                            continue
                        name = source.replace("model.language_model.", "model.", 1) if multimodal else source
                        if name == "lm_head.weight" and config.tie_word_embeddings:
                            raise ValueError("tied checkpoints must omit the duplicate lm_head weight")
                        if name not in parameters or name in loaded:
                            raise ValueError(f"unexpected or duplicate parameter: {source}")
                        value = shard.get_tensor(source)
                        if value.shape != parameters[name].shape:
                            raise ValueError(f"shape mismatch for {source}")
                        parameters[name].copy_(value)
                        loaded.add(name)
        if loaded != set(parameters):
            raise ValueError(f"missing checkpoint parameters: {sorted(set(parameters) - loaded)}")
        return model.eval()
