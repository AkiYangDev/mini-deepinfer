"""Text input for the existing single-request greedy draft/target loop."""


def run_speculative_text(target, small, tokenizer, prompt, *, max_new_tokens=32,
                         cache_capacity=512, width=1, raw=False, thinking=False,
                         trace_rounds=False, check_greedy=False, on_commit=None, should_cancel=None):
    """Own both request caches; reuse loaded models without sharing their state.

    The optional greedy check runs afterwards with a fresh target cache. It is
    a correctness check, not a latency comparison or part of normal generation.
    """
    import torch
    from .engine import Engine, validate_request
    from .model_draft import ModelDraft
    from .model_draft_trace import DraftTrace
    from .model_runner import ModelRunner
    from .speculative import generate
    from .text_loop import encode_prompt

    if type(width) is not int or not 1 <= width <= 8:
        raise ValueError('draft width must be an integer from one to eight')
    if type(cache_capacity) is not int or cache_capacity < 1:
        raise ValueError('cache capacity must be a positive integer')
    if should_cancel is not None and not callable(should_cancel):
        raise ValueError('should_cancel must be a callable or None')
    weights = [next(model.parameters()) for model in (target, small)]
    if (any(weight.dtype != torch.float32 for weight in weights)
            or weights[0].device != weights[1].device
            or weights[0].device.type not in ('cpu', 'npu')):
        raise ValueError('draft and target must use FP32 on the same CPU or NPU device')
    if target.config.vocab_size != small.config.vocab_size:
        raise ValueError('draft and target vocabularies differ')
    ids = validate_request(encode_prompt(tokenizer, prompt, raw=raw, thinking=thinking),
                           max_new_tokens, tokenizer.eos_token_id, cache_capacity)
    needed = len(ids)+max_new_tokens-1
    if (any(token >= target.config.vocab_size for token in ids)
            or tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= target.config.vocab_size):
        raise ValueError('prompt or EOS is outside the model vocabulary')
    if needed > min(target.config.max_position_embeddings, small.config.max_position_embeddings):
        raise ValueError('request exceeds a model position limit')
    # Capacity is a request budget, not a reason to allocate beyond its inputs.
    capacity = needed
    draft = ModelDraft(small, capacity)
    trace = DraftTrace(draft, ids) if trace_rounds else None
    if trace is not None:
        draft.on_cache = trace.aligned
        trace.report.update(path='text generation', width=width)

    def committed(tokens):
        if trace is not None:
            trace.committed(tokens)
        if on_commit is not None:
            on_commit(tokens)

    try:
        result = generate(target, ids, max_new_tokens, trace.propose if trace else draft,
                          width=width, eos_token_id=tokenizer.eos_token_id,
                          cache_capacity=capacity, on_commit=committed,
                          on_round=trace.observed if trace else None, should_cancel=should_cancel)
    finally:
        draft.close()
    result.update(prompt_ids=list(ids), text=tokenizer.decode(result['output_ids'], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False),
                  cache_capacity=capacity, cache_budget=cache_capacity, draft_width=width,
                  draft_work=draft.stats(), greedy_check=dict(status='not_run'))
    if trace is not None:
        trace.closed()
        trace.report['target_free_slots'] = result['free_slots']
        trace.report['finish_reason'] = result['finish_reason']
        result['trace'] = trace.report
    if check_greedy:
        cancelled = result['finish_reason'] == 'cancelled'
        count = len(result['output_ids']) if cancelled else max_new_tokens
        if count == 0:
            result['greedy_check']['reason'] = 'no_committed_tokens'
            return result
        runner = ModelRunner(target, capacity)
        engine = Engine(runner, capacity)
        request = engine.submit(ids, count, tokenizer.eos_token_id)
        for _ in engine.run():
            pass
        expected_reason = 'length' if cancelled else result['finish_reason']
        matched = request.output_ids == result['output_ids'] and request.finish_reason == expected_reason
        result['greedy_check'] = dict(status='passed' if matched else 'failed',
                                      output_ids=request.output_ids, finish_reason=request.finish_reason,
                                      free_slots=engine.cache.free_count)
        if cancelled:
            result['greedy_check']['scope'] = 'committed_prefix'
        if not matched or engine.cache.free_count != capacity:
            raise RuntimeError('speculative output or cleanup differs from independent greedy')
    return result
