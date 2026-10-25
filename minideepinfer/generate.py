"""Offline text generation with local Qwen3.5 or Qwen2 weights; greedy by default."""

import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint directory")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--prompt", action="append", help="repeat to batch multiple prompts")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-capacity", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=None, help="active requests in a batch (default: 4)")
    parser.add_argument("--max-batched-tokens", type=int, default=None, help="whole-prompt + decode budget")
    parser.add_argument("--raw", action="store_true", help="skip the checkpoint's chat template")
    parser.add_argument("--thinking", action="store_true", help="enable thinking if the chat template supports it")
    parser.add_argument("--chunked-prefill", action="store_true", help="split prompts by the batch token budget")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--deferred", action="store_true", help="launch and collect each batch separately")
    modes.add_argument("--pipeline", action="store_true", help="use cross-round device-token execution (FP32)")
    modes.add_argument("--speculative", action="store_true", help="use a separate greedy draft model (one prompt, FP32)")
    parser.add_argument("--draft-model", help="local draft checkpoint; requires --speculative")
    parser.add_argument("--draft-width", type=int, default=None, help="1..8 candidates per speculative round (default: 1)")
    parser.add_argument("--temperature", type=float, help="positive temperature; speculative FP32 reference (default: greedy)")
    parser.add_argument("--seed", type=int, help="request-local sampling seed (default: 2026); requires --temperature")
    parser.add_argument("--top-k", type=int, help="keep at most k tokens; 0 disables; requires --temperature")
    parser.add_argument("--top-p", type=float, help="nucleus threshold in (0,1]; applied after top-k; requires --temperature")
    parser.add_argument("--check-greedy", action="store_true", help="with --speculative, compare against a fresh target-only run")
    parser.add_argument("--output", help="with --speculative, save a generation or cancellation report as JSON")
    parser.add_argument("--stream", action="store_true", help="with --speculative, print committed text during generation")
    parser.add_argument("--cancel-after-tokens", type=int,
                        help="with --speculative, request cancellation after a commit reaches this output count")
    parser.add_argument("--prepare-during-step", action="store_true",
                        help="tokenize one new prompt before collecting a batch; enables --deferred")
    parser.add_argument("--stagger-prompts", action="store_true",
                        help="with --pipeline, submit one more prompt per collected result")
    parser.add_argument("--trace-rounds", action="store_true",
                        help="with --pipeline or --speculative, print result rounds and cache state")
    args = parser.parse_args(argv)
    if args.pipeline and (args.prepare_during_step or args.dtype != 'float32'):
        parser.error('--pipeline requires float32 and cannot use --prepare-during-step')
    if args.stagger_prompts and not args.pipeline:
        parser.error('--stagger-prompts requires --pipeline')
    if args.trace_rounds and not (args.pipeline or args.speculative):
        parser.error('--trace-rounds requires --pipeline or --speculative')
    args.deferred = args.deferred or args.prepare_during_step
    args.prompt = args.prompt or ["Explain what a KV cache does in two sentences."]
    if args.speculative:
        if not args.draft_model:
            parser.error('--speculative requires --draft-model')
        if (args.dtype != 'float32' or len(args.prompt) != 1 or args.prepare_during_step
                or args.stagger_prompts or args.chunked_prefill or args.max_batched_tokens is not None
                or args.max_num_seqs is not None):
            parser.error('--speculative requires one prompt and float32; batch/prefill/deferred options are unsupported')
        if args.max_new_tokens < 1 or args.cache_capacity < 1 or args.draft_width is not None and not 1 <= args.draft_width <= 8:
            parser.error('choose positive token/cache limits and a draft width from 1 to 8')
        args.draft_width = 1 if args.draft_width is None else args.draft_width
        if args.cancel_after_tokens is not None and args.cancel_after_tokens < 1:
            parser.error('--cancel-after-tokens must be positive')
        if args.temperature is not None:
            from .sampling import TemperatureSampler
            try:
                TemperatureSampler(args.temperature, 2026 if args.seed is None else args.seed,
                                   top_k=0 if args.top_k is None else args.top_k,
                                   top_p=1.0 if args.top_p is None else args.top_p)
            except ValueError as error:
                parser.error(str(error))
            if (args.device != 'cpu' and not (args.device.startswith('npu:') and args.device[4:].isdigit())
                    or args.check_greedy):
                parser.error('--temperature requires CPU or NPU and cannot use --check-greedy')
        elif args.seed is not None or args.top_k is not None or args.top_p is not None:
            parser.error('--seed, --top-k and --top-p require --temperature')
    elif (args.draft_model is not None or args.draft_width is not None or args.check_greedy or args.output is not None
          or args.stream or args.cancel_after_tokens is not None or args.temperature is not None or args.seed is not None
          or args.top_k is not None or args.top_p is not None):
        parser.error('draft, stream, cancellation, greedy-check and report options require --speculative')
    args.max_num_seqs = 4 if args.max_num_seqs is None else args.max_num_seqs
    return args


def run_generation(runner, tokenizer, args, *, on_round=None):
    """Use the same frontend with a loaded runner in the CLI and numerical checks."""
    if args.speculative:
        raise ValueError('use run_speculative_text with both loaded models')
    from . import BatchEngine, Engine
    from .text_loop import encode_prompt, run_text_requests

    prompts = args.prompt
    if args.pipeline:
        from .pipeline_text import run_pipeline_text
        from .stopping_pipeline import StoppingDecodePipeline
        pipeline = StoppingDecodePipeline(runner)
        pairs = run_pipeline_text(pipeline, tokenizer, prompts, max_new_tokens=args.max_new_tokens,
                                  raw=args.raw, thinking=args.thinking, max_num_seqs=args.max_num_seqs,
                                  max_num_batched_tokens=args.max_batched_tokens,
                                  chunked_prefill=args.chunked_prefill, stagger_prompts=args.stagger_prompts,
                                  on_round=on_round)
        return pairs, pipeline.pool
    if args.deferred:
        from .deferred_runner import DeferredRunner
        runner = DeferredRunner(runner)
    engine = (Engine(runner, args.cache_capacity) if len(prompts) == 1 and not args.chunked_prefill and not args.deferred else
              BatchEngine(runner, args.cache_capacity, args.max_num_seqs,
                          args.max_batched_tokens, args.chunked_prefill, deferred=args.deferred))
    if args.prepare_during_step:
        pairs = run_text_requests(engine, tokenizer, prompts, max_new_tokens=args.max_new_tokens,
                                  raw=args.raw, thinking=args.thinking, prepare_during_step=True)
    else:
        pairs = [(engine.submit(encode_prompt(tokenizer, prompt, raw=args.raw, thinking=args.thinking),
                                args.max_new_tokens, tokenizer.eos_token_id), prompt) for prompt in prompts]
        for _ in engine.run():
            pass
    return pairs, engine.cache


def main(argv=None):
    args = parse_args(argv)
    contract = None
    if args.speculative:
        from .model_draft import check_tokenizer_pair
        contract = check_tokenizer_pair(args.draft_model, args.model)
    import torch
    from transformers import AutoTokenizer
    from .model_runner import ModelRunner
    from .pipeline_text import print_round

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.speculative:
        import hashlib
        import json
        import platform
        from pathlib import Path
        from .model_draft_trace import print_trace
        from .speculative_text import run_speculative_text

        # Only the loaded models survive here; the Runner's unused caches do not.
        target = ModelRunner.from_pretrained(args.model, args.device, torch.float32, 1).model
        small = ModelRunner.from_pretrained(args.draft_model, args.device, torch.float32, 1).model
        from threading import Event
        cancel = Event()
        committed_count = 0
        streamer = None
        if args.stream:
            from transformers import TextStreamer
            streamer = TextStreamer(tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        def committed(tokens):
            nonlocal committed_count
            committed_count += len(tokens)
            if streamer is not None:
                streamer.put(torch.tensor(tokens, dtype=torch.long))
            if args.cancel_after_tokens is not None and committed_count >= args.cancel_after_tokens:
                cancel.set()
        try:
            result = run_speculative_text(target, small, tokenizer, args.prompt[0],
                    max_new_tokens=args.max_new_tokens, cache_capacity=args.cache_capacity,
                    width=args.draft_width, raw=args.raw, thinking=args.thinking,
                    trace_rounds=args.trace_rounds, check_greedy=args.check_greedy,
                    on_commit=committed, should_cancel=cancel.is_set,
                    temperature=args.temperature, seed=2026 if args.seed is None else args.seed,
                    top_k=0 if args.top_k is None else args.top_k,
                    top_p=1.0 if args.top_p is None else args.top_p)
        finally:
            if streamer is not None:
                streamer.end()
        result.update(status='cancelled' if result['finish_reason']=='cancelled' else 'completed',
                      tokenizer_contract=contract, raw=args.raw, thinking=args.thinking,
                      device=str(next(target.parameters()).device), dtype='float32',
                      target_model=Path(args.model).name, draft_model=Path(args.draft_model).name,
                      python=platform.python_version(), torch=torch.__version__, cpu_threads=torch.get_num_threads(),
                      scope='single-request eager greedy; independent draft cache; no timing or speedup claim')
        if args.temperature is not None:
            result['scope'] = 'single-request FP32 temperature sampling with CPU FP64 probabilities; independent draft cache; no distribution certification or speedup claim'
        if args.device.startswith('npu'):
            import torch_npu
            result.update(torch_npu=torch_npu.__version__,
                          hf32=dict(matmul=torch.npu.matmul.allow_hf32, conv=torch.npu.conv.allow_hf32))
        if args.stream or args.cancel_after_tokens is not None:
            result.update(stream=args.stream, cancel_after_tokens=args.cancel_after_tokens)
        if result['greedy_check']['status'] == 'passed':
            result['passed'] = True
        sources = ('generate.py', 'speculative_text.py', 'text_loop.py', 'model_draft.py',
                   'model_draft_trace.py', 'speculative.py', 'sampling.py', 'model_runner.py', 'models/qwen2.py',
                   'models/qwen3_5.py', 'tensor_cache.py', 'hybrid_cache.py', 'engine.py', 'kv_cache.py')
        result['source_sha256'] = {'minideepinfer/'+name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                  for name in sources}
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.tmp')
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            temporary.replace(path)
        if not args.stream:
            print(result['text'])
        print(f"\n[{result['finish_reason']}; {len(result['output_ids'])} output tokens; "
              f"accepted draft {result['accepted_draft_tokens']}/{result['proposed_draft_tokens']}; "
              f"target/draft free slots {result['free_slots']}/{result['draft_work']['free_slots']} "
              f"of {result['cache_capacity']} each; greedy check {result['greedy_check']['status']}]")
        if args.trace_rounds:
            print_trace(result['trace'])
        return result
    runner = ModelRunner.from_pretrained(
        args.model, args.device, getattr(torch, args.dtype), args.cache_capacity
    )
    pairs, pool = run_generation(runner, tokenizer, args, on_round=print_round if args.trace_rounds else None)
    for request, prompt in pairs:
        if len(pairs) > 1:
            print(f"\n{request.request_id}: {prompt}")
        print(tokenizer.decode(request.output_ids, skip_special_tokens=True))
        print(f"\n[{request.finish_reason}; {len(request.output_ids)} output tokens; "
              f"free KV slots {pool.free_count}/{pool.capacity}]")


if __name__ == "__main__":
    main()
