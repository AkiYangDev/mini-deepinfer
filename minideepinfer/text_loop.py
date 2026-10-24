"""Prepare one text arrival per iteration, before or after collecting the batch."""


def encode_prompt(tokenizer, prompt, *, raw=False, thinking=False):
    if raw:
        return tokenizer.encode(prompt)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=False, enable_thinking=thinking,
    )


def run_text_requests(engine, tokenizer, prompts, *, max_new_tokens=32,
                      raw=False, thinking=False, prepare_during_step=False):
    """Consume texts lazily; return (request, original text) pairs after draining.

    Both orders admit the next arrival on the next iteration. If preparing it
    raises, collect the already launched batch before propagating the error.
    The caller still owns the engine and can finish its other valid requests.
    """
    if prepare_during_step and not getattr(engine, 'deferred', False):
        raise ValueError('preparing during a step requires a deferred BatchEngine')
    incoming = iter(prompts)
    requests = []

    def submit_next():
        try:
            text = next(incoming)
        except StopIteration:
            return False
        tokens = encode_prompt(tokenizer, text, raw=raw, thinking=thinking)
        request = engine.submit(tokens, max_new_tokens, tokenizer.eos_token_id)
        requests.append((request, text))
        return True

    if not submit_next():
        return requests
    while True:
        if prepare_during_step:
            plans = engine.launch_step()
            try:
                added = submit_next()
            finally:
                if plans:
                    engine.finish_step()
            active = bool(plans)
        else:
            active = bool(engine.step())
            added = submit_next()
        if not active and not added:
            return requests
