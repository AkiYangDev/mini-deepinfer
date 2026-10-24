"""Offline greedy text generation with local Qwen3.5 or Qwen2 weights."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint directory")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--prompt", action="append", help="repeat to batch multiple prompts")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-capacity", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4, help="active requests in a batch")
    parser.add_argument("--max-batched-tokens", type=int, default=None, help="whole-prompt + decode budget")
    parser.add_argument("--raw", action="store_true", help="skip the checkpoint's chat template")
    parser.add_argument("--thinking", action="store_true", help="enable thinking if the chat template supports it")
    parser.add_argument("--chunked-prefill", action="store_true", help="split prompts by the batch token budget")
    parser.add_argument("--deferred", action="store_true", help="launch and collect each batch separately")
    parser.add_argument("--prepare-during-step", action="store_true",
                        help="tokenize one new prompt before collecting a batch; enables --deferred")
    args = parser.parse_args()
    args.deferred = args.deferred or args.prepare_during_step

    import torch
    from transformers import AutoTokenizer
    from . import BatchEngine, Engine
    from .model_runner import ModelRunner
    from .text_loop import encode_prompt, run_text_requests

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = args.prompt or ["Explain what a KV cache does in two sentences."]
    runner = ModelRunner.from_pretrained(
        args.model, args.device, getattr(torch, args.dtype), args.cache_capacity
    )
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
    for request, prompt in pairs:
        if len(pairs) > 1:
            print(f"\n{request.request_id}: {prompt}")
        print(tokenizer.decode(request.output_ids, skip_special_tokens=True))
        print(f"\n[{request.finish_reason}; {len(request.output_ids)} output tokens; "
              f"free KV slots {engine.cache.free_count}/{engine.cache.capacity}]")


if __name__ == "__main__":
    main()
