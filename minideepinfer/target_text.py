"""Single-model text input for the existing synchronous generation loop."""


def run_target_text(target, tokenizer, prompt, *, max_new_tokens=32,
                    cache_capacity=512, raw=False, thinking=False,
                    on_commit=None, should_cancel=None,
                    temperature=None, seed=2026, top_k=0, top_p=1.0):
    """Generate with one model and a request-local sampler and cache.

    Width zero makes every output a direct target draw. No draft model or
    draft cache is constructed; stopping and cleanup use the same loop as
    speculative generation.
    """
    import torch
    from .engine import validate_request
    from .sampling import SampledDraft, TemperatureSampler
    from .speculative import generate
    from .text_loop import encode_prompt

    settings = TemperatureSampler(1.0 if temperature is None else temperature, seed,
                                  top_k=top_k, top_p=top_p)
    if temperature is None and (top_k or top_p != 1):
        raise ValueError('top_k and top_p require a positive temperature')
    if type(cache_capacity) is not int or cache_capacity < 1:
        raise ValueError('cache capacity must be a positive integer')
    weight = next(target.parameters())
    if weight.dtype != torch.float32 or weight.device.type not in ('cpu', 'npu'):
        raise ValueError('target must use FP32 on CPU or NPU')
    ids = validate_request(encode_prompt(tokenizer, prompt, raw=raw, thinking=thinking),
                           max_new_tokens, tokenizer.eos_token_id, cache_capacity)
    capacity = len(ids)+max_new_tokens-1
    sampler = None if temperature is None else settings

    def no_draft(prefix, limit):
        return () if sampler is None else SampledDraft((), ())

    result = generate(target, ids, max_new_tokens, no_draft, width=0,
                      eos_token_id=tokenizer.eos_token_id, cache_capacity=capacity,
                      sampler=sampler, on_commit=on_commit, should_cancel=should_cancel)
    result.update(prompt_ids=list(ids), text=tokenizer.decode(result['output_ids'], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False),
                  cache_capacity=capacity, cache_budget=cache_capacity, draft_width=0)
    return result
