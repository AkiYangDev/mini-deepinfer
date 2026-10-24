"""Reproduce Qwen3.5 BF16 greedy equality; not a latency/throughput benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from minideepinfer import BatchEngine
from minideepinfer.model_runner import ModelRunner
from minideepinfer.reference_qwen3_5 import RecurrentFP32Reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prompts = ['用两句话解释 KV Cache 的作用。', 'What is 1 plus 1? Answer in one short sentence.']
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
           add_generation_prompt=True, enable_thinking=False, return_dict=False) for text in prompts]
    maximum, capacity = 128, 512
    loaded = ModelRunner.from_pretrained(args.model, args.device, torch.bfloat16, capacity)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
            attn_implementation='eager', local_files_only=True).to(loaded.device).eval()
    reference = RecurrentFP32Reference(model)
    runs = []
    with torch.inference_mode():
        for chunked in (False, True):
            runner = ModelRunner(loaded.model, capacity)
            engine = BatchEngine(runner, capacity, max_num_seqs=2,
                                 max_num_batched_tokens=16 if chunked else 128,
                                 chunked_prefill=chunked)
            requests = [engine.submit(tokens, maximum, tokenizer.eos_token_id) for tokens in ids]
            list(engine.run())
            runs.append({'chunked': chunked, 'max_num_batched_tokens': engine.max_num_batched_tokens,
                         'outputs': [r.output_ids for r in requests],
                         'finish_reasons': [r.finish_reason for r in requests],
                         'free_slots': engine.cache.free_count,
                         'remaining_linear_states': len(runner.cache.linear_states),
                         'remaining_prefixes': len(runner._prefixes)})
            runner.synchronize()
        expected, reasons = [], []
        for tokens in ids:
            outputs, past = [], None
            inputs = tokens
            for _ in range(maximum):
                result = reference(torch.tensor([inputs], device=loaded.device),
                                   past_key_values=past, use_cache=True)
                past = result.past_key_values
                token = int(result.logits[0, -1].argmax())
                outputs.append(token)
                if token == tokenizer.eos_token_id:
                    break
                inputs = [token]
            expected.append(outputs)
            reasons.append('eos' if outputs[-1] == tokenizer.eos_token_id else 'length')
        loaded.synchronize()
    equal = all(run['outputs'] == expected and run['finish_reasons'] == reasons for run in runs)
    reclaimed = all(run['free_slots'] == capacity and run['remaining_linear_states'] == 0
                    and run['remaining_prefixes'] == 0 for run in runs)
    report = {'passed': equal and reclaimed, 'model': 'Qwen/Qwen3.5-2B', 'dtype': 'bfloat16',
              'device': str(loaded.device), 'reference_mode': 'recurrent-fp32',
              'reference_scope': 'HF recurrent equations with FP32 state; not native chunk-path compatibility',
              'max_new_tokens': maximum, 'cache_capacity': capacity, 'prompts': prompts,
              'prompt_ids': ids, 'runs': runs, 'reference_output_ids': expected,
              'finish_reasons': reasons,
              'texts': [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in expected]}
    import minideepinfer
    root = Path(minideepinfer.__file__).parent
    report['source_sha256'] = {'minideepinfer/' + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ['model_runner.py', 'models/qwen3_5.py', 'hybrid_cache.py',
                     'reference_qwen3_5.py', 'batch_engine.py']}
    report['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'finish_reasons': reasons,
                      'output_tokens': [len(x) for x in expected], 'texts': report['texts']}, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
