"""Offline generation with local DeepSeek-R1-Distill-Qwen-1.5B weights."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local checkpoint directory")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--prompt", default="Explain what a KV cache does in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-capacity", type=int, default=512)
    parser.add_argument("--raw", action="store_true", help="skip the checkpoint's chat template")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from .engine import Engine
    from .model_runner import ModelRunner

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.raw:
        tokens = tokenizer.encode(args.prompt)
    else:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
    runner = ModelRunner.from_pretrained(
        args.model, args.device, getattr(torch, args.dtype), args.cache_capacity
    )
    engine = Engine(runner, args.cache_capacity)
    request = engine.submit(tokens, args.max_new_tokens, tokenizer.eos_token_id)
    for _ in engine.run():
        pass
    print(tokenizer.decode(request.output_ids, skip_special_tokens=True))
    print(f"\n[{request.finish_reason}; {len(request.output_ids)} output tokens; "
          f"free KV slots {engine.cache.free_count}/{engine.cache.capacity}]")


if __name__ == "__main__":
    main()
