"""Text input and result-callback observations for the existing pipeline."""

from .text_loop import encode_prompt


def run_pipeline_text(pipeline, tokenizer, prompts, *, max_new_tokens=32, raw=False,
                      thinking=False, max_num_seqs=4, max_num_batched_tokens=None,
                      chunked_prefill=False, stagger_prompts=False, on_round=None):
    """Optionally admit one new text per collected result, on the same CPU thread.

    Observations are before CPU commit. A release is reported when its absence
    is first observed, not at an invented device timestamp. No device tensor is
    read by this adapter; the pipeline supplies already copied host tokens.
    """
    texts = list(prompts)
    if not texts:
        return []
    incoming = iter(enumerate(texts))
    initial = {}
    for index, text in incoming:
        initial[f'req-{index}'] = tuple(encode_prompt(tokenizer, text, raw=raw, thinking=thinking))
        if stagger_prompts:
            break
    previous_owners = {}

    def observe(index, steps, tokens, logits):
        nonlocal previous_owners
        arrived = []
        if stagger_prompts:
            item = next(incoming, None)
            if item is not None:
                number, text = item
                rid = f'req-{number}'
                ids = encode_prompt(tokenizer, text, raw=raw, thinking=thinking)
                pipeline.submit(rid, ids, max_new_tokens, tokenizer.eos_token_id)
                arrived.append(rid)
        if on_round is not None:
            owners = dict(pipeline._slots)
            record = dict(event='result', round=index, issued_round=pipeline._states[-1].index,
                          steps=[dict(request_id=s.request_id, phase=s.phase, positions=s.positions,
                                      input_tokens=len(s.input_ids), partial=not s.is_last_prompt_chunk,
                                      sampled_token=t) for s, t in zip(steps, tokens)],
                          arrived=arrived, released=[rid for rid in previous_owners if rid not in owners],
                          owners=owners, waiting=[rid for rid, r in pipeline.requests.items()
                                                 if r.status.value == 'waiting'],
                          free_slots=pipeline.pool.free_count, capacity=pipeline.pool.capacity)
            previous_owners = owners
            on_round(record)

    result = pipeline.run(initial, max_new_tokens,
                          eos_token_ids={rid: tokenizer.eos_token_id for rid in initial},
                          max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                          chunked_prefill=chunked_prefill, on_result=observe)
    if on_round is not None:
        on_round(dict(event='drained', released=list(previous_owners),
                      free_slots=pipeline.pool.free_count, capacity=pipeline.pool.capacity))
    return [(result[f'req-{index}'], text) for index, text in enumerate(texts)]


def print_round(record):
    """Print logical ownership at observation points, without timing claims."""
    if record['event'] == 'result':
        print(f"collect round {record['round']} [before CPU commit; "
              f"issued through round {record['issued_round']}]")
        for step in record['steps']:
            positions = step['positions']
            kind = ' partial' if step['partial'] else ''
            token = '-' if step['sampled_token'] is None else step['sampled_token']
            print(f"  {step['request_id']} {step['phase']} {positions[0]}..{positions[-1]}"
                  f"{kind}; sample={token}")
        if record['arrived']:
            print('  arrived: ' + ', '.join(record['arrived']))
        print('  waiting: ' + (', '.join(record['waiting']) or '-'))
    else:
        print('pipeline drained')
    if record['released']:
        print('  observed released: ' + ', '.join(record['released']))
    print(f"  free slots: {record['free_slots']}/{record['capacity']}")
