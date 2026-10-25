"""CPU FP64 probability reference for temperature and speculative sampling.

Algorithm 1: https://proceedings.mlr.press/v202/leviathan23a.html
Dense distributions deliberately stay on the host; this is not an NPU kernel.
"""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class SampledDraft:
    tokens: tuple[int, ...]
    probabilities: tuple  # One full q distribution at each candidate's prefix.


def distribution(values):
    """Reject malformed probabilities instead of silently repairing a proposal."""
    import torch
    values = torch.as_tensor(values, dtype=torch.float64, device='cpu').detach()
    if (values.ndim != 1 or values.numel() == 0 or not bool(torch.isfinite(values).all())
            or bool((values < 0).any()) or abs(float(values.sum())-1.0) > 1e-10):
        raise ValueError('probabilities must be a finite, nonnegative, normalized vector')
    return values


def draw(probabilities, rng):
    """Inverse CDF with one request-local uniform draw, including sparse support."""
    import torch
    probabilities = distribution(probabilities)
    uniform = rng.random()
    if not 0 <= uniform < 1:
        raise ValueError('random draws must lie in [0, 1)')
    cumulative = probabilities.cumsum(0)
    # Normalize the CDF endpoint to avoid selecting past it after roundoff.
    total = float(cumulative[-1])
    threshold = min(uniform*total, math.nextafter(total, -math.inf))
    index = int(torch.searchsorted(cumulative, threshold, right=True))
    return index


class TemperatureSampler:
    def __init__(self, temperature=1.0, seed=2026, *, top_k=0, top_p=1.0):
        if (type(temperature) not in (int, float) or not math.isfinite(temperature)
                or temperature <= 0):
            raise ValueError('temperature must be positive and finite')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer; zero disables it')
        if type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError('top_p must be finite and in (0, 1]')
        self.temperature, self.seed = float(temperature), seed
        self.top_k, self.top_p = top_k, float(top_p)
        self.rng = random.Random(seed)

    def probabilities(self, logits):
        import torch
        # Transfer FP32 first; the device never needs to represent FP64 tensors.
        logits = logits.detach().cpu().to(dtype=torch.float64)
        if logits.ndim != 1 or logits.numel() == 0 or not bool(torch.isfinite(logits).all()):
            raise ValueError('sampling requires finite one-dimensional logits')
        # Subtract first: a tiny positive temperature must not produce inf-inf.
        scaled = (logits-logits.max())/self.temperature
        if not self.top_k and self.top_p == 1:
            return distribution(scaled.softmax(-1))  # Preserve the original path.
        # Stable order breaks equal-logit ties by ascending vocabulary ID.
        order = torch.argsort(scaled, descending=True, stable=True)
        if self.top_k:
            order = order[:self.top_k]
        weights = scaled[order].softmax(-1)
        if self.top_p < 1:
            # Keep the first token that reaches/exceeds p, even for tiny p.
            before = torch.cat((weights.new_zeros(1), weights.cumsum(0)[:-1]))
            keep = before < self.top_p
            order, weights = order[keep], weights[keep]
        probabilities = torch.zeros_like(scaled)
        probabilities[order] = weights/weights.sum()
        return distribution(probabilities)

    def sample(self, logits):
        return draw(self.probabilities(logits), self.rng)


def verify_sampled(proposal, target, remaining, rng, eos_token_id=None):
    """Accept x with min(1,p[x]/q[x]); reject into normalized positive p-q.

    target has one p row per candidate plus a final bonus row. Each row is
    conditioned on the same prefix as the matching draft row. Return the usual
    commit decision and small host observations, never the full vocabulary.
    """
    import torch
    from .speculative import Decision

    if not isinstance(proposal, SampledDraft):
        raise ValueError('sampled verification requires candidate probabilities')
    tokens, qs = proposal.tokens, proposal.probabilities
    ps = tuple(target)
    if (type(remaining) is not int or remaining < 1 or len(tokens) >= remaining
            or len(qs) != len(tokens) or len(ps) != len(tokens)+1):
        raise ValueError('verification needs draft+1 target rows and room for a final token')
    ps = tuple(distribution(p) for p in ps)
    qs = tuple(distribution(q) for q in qs)
    size = ps[0].numel()
    if (any(p.numel() != size for p in ps+qs)
            or any(type(t) is not int or not 0 <= t < size for t in tokens)
            or eos_token_id is not None and (type(eos_token_id) is not int or not 0 <= eos_token_id < size)):
        raise ValueError('probability vocabularies and token IDs must match')
    if any(float(q[t]) <= 0 for t, q in zip(tokens, qs)):
        raise ValueError('a sampled candidate must have positive draft probability')
    checks = []
    accepted = 0
    for token, p, q in zip(tokens, ps, qs):
        threshold = min(1.0, float(p[token])/float(q[token]))
        uniform = rng.random()
        if not 0 <= uniform < 1:
            raise ValueError('random draws must lie in [0, 1)')
        keep = uniform < threshold
        checks.append(dict(token=token, p=float(p[token]), q=float(q[token]),
                           acceptance_probability=threshold, uniform=uniform, accepted=keep))
        if not keep:
            residual = (p-q).clamp_min(0)
            mass = float(residual.sum())
            if mass <= 0:
                raise RuntimeError('a rejected candidate must leave positive residual mass')
            final = draw(residual/mass, rng)
            role = 'correction'
            break
        accepted += 1
        if token == eos_token_id:
            return Decision(tokens[:accepted], accepted, None, 'eos'), checks
    else:
        final = draw(ps[-1], rng)
        role = 'bonus' if tokens else 'target'
    committed = tokens[:accepted]+(final,)
    reason = 'eos' if final == eos_token_id else 'length' if len(committed) == remaining else None
    return Decision(committed, accepted, role, reason), checks
