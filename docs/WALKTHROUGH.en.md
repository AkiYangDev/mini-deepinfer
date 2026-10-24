# Follow A, B and C through the pipeline

[Home](../README.en.md) · [简体中文](WALKTHROUGH.md) · [Your own text](PIPELINE_TEXT.md)

A has a long prompt, B generates just two tokens, and C arrives during execution. This example runs the project's Qwen3.5 forward pass through the existing `StoppingDecodePipeline` to show batching, waiting, and cache reuse.

## Run it

With compatible PyTorch installed, run from the repository directory. For a new environment, start with [CPU setup](START_HERE.en.md#tiny-cpu-model-no-checkpoint).

```bash
python -m minideepinfer.pipeline_demo
```

The model is tiny, randomly initialized, and runs on CPU without a checkpoint. Token IDs have no textual meaning. EOS is disabled so the example always generates 6, 2, and 3 tokens, respectively, and stops by length. To save observations:

```bash
python -m minideepinfer.pipeline_demo --output outputs/pipeline-demo.json
```

Each round has a budget of four input tokens. At most two requests can hold slots, and total capacity is 20. Admission reserves `prompt length + maximum output length - 1` slots for the whole request. The final output is returned without feeding it back into the model.

| Request | Prompt tokens | Output tokens | Slots | Joins the queue |
| --- | --- | --- | --- | --- |
| A | 9 | 6 | 14 | At the start |
| B | 2 | 2 | 3 | While collecting round 0 |
| C | 4 | 3 | 6 | While collecting round 1 |

## Why is CPU progress 0 while issued progress is 8?

The first snapshot reads:

```text
collect 0 | issued through 1
  A prefill 0..3: partial prompt; no sample
  arrived: B
  A: running; cached CPU/issued=0/8; slots=14
  B: waiting; cached CPU/issued=0/0; slots=0
  free slots: 6/20
```

Round 0's result is available, but the callback runs before the CPU commits it. Round 1 has already been submitted, so eight prompt tokens have been issued. CPU progress is still zero and becomes four after this callback returns.

Issued does not mean device-complete. CPU computation is synchronous here; the example demonstrates execution order, not NPU overlap or elapsed time.

A still has its ninth prompt token to process. Intermediate chunks update cache without producing an output, hence `partial prompt; no sample`. B arrives in this callback and cannot join round 1, which has already been issued.

## B generates while C waits

C arrives while collecting round 1. A and B hold 17 slots, leaving only three. C needs six, and both active-request places are occupied, so it waits without allocating slots or model state.

Round 2 handles A's last prompt token and B's two-token prompt together. Both rows can sample. Round 3 decodes one token for each request. The round 3 snapshot still shows B as running: its second output has not been committed yet. After the callback returns, that output is saved and B finishes.

Planning uses issued progress, while CPU result records advance in collection order. Treating these as one counter would risk repeating work or connecting a request to the wrong previous token.

## B released slots. Why are zero slots free?

The later snapshot includes:

```text
collect 4 | issued through 5
  A decode 10..10: sample available
  A: running; cached CPU/issued=10/12; slots=14
  B: finished; cached CPU/issued=3/3; slots=0
  C: running; cached CPU/issued=0/3; slots=6
  observed released: B
  reused B -> C: slots [14, 15, 16]
  free slots: 0/20
```

C took B's slots 14, 15, and 16, plus the three previously free slots. A now holds 14 and C holds six. Counting only free slots would miss the handoff.

`observed released` compares ownership between snapshots. `reused` checks each slot against its recorded prior owner. Neither is an exact device-event timestamp. The pipeline itself checks the request's last-use event before releasing it. This fixed-length example knows in advance when B has no more work to issue. With natural EOS, the CPU may learn about a stop after an extra round has been submitted; see [stopping and cleanup](STOPPING_PIPELINE.md).

C's prompt has four tokens, but A's decode takes one token from the round's budget first. C processes three prompt tokens, then the last token in a later round before starting decode.

## Check the ending

```text
drained | observed released: C
  A: length; 6 output tokens
  B: length; 2 output tokens
  C: length; 3 output tokens
whole-prompt reference outputs match: True
free slots: 20/20; clean: True
```

The program also runs each request through the original synchronous `Engine`, using the same read-only model with independent caches and whole prompts. It checks final token equality, length stops, and cleanup. It does not compare full logits; those checks remain in the [text verifier and preserved CPU/NPU results](PIPELINE_TEXT.md).

The [complete saved output](results/2026-10-24/pipeline-demo-cpu.txt) comes from an actual CPU run. Read [`pipeline_demo.py`](../minideepinfer/pipeline_demo.py): `observe()` only submits B/C and records state. Scheduling, sampling, stopping, and release stay in the original loop.

## Use real text on Ascend 910C

Prepare the tested environment and existing Qwen3.5-2B weights using [setup](START_HERE.en.md#real-qwen35-2b-on-ascend-910c). In Bash:

```bash
python -m minideepinfer.generate \
  --model weights/Qwen3.5-2B --device npu:0 --dtype float32 \
  --prompt "Explain prefill, decode, and why a request needs a KV cache." \
  --prompt "What is 1 plus 1? Answer in one short sentence." \
  --prompt "用两句话解释为什么不同请求的缓存需要隔离。" \
  --max-new-tokens 64 --cache-capacity 512 \
  --max-num-seqs 2 --max-batched-tokens 16 --chunked-prefill \
  --pipeline --stagger-prompts --trace-rounds
```

Real text uses a chat template, natural EOS, and the same maximum output length of 64 for all requests. Tokenization, stopping order, and reuse rounds depend on model output, so they will differ from the fixed nine-round example. This command uses the existing text frontend. The new walkthrough was run on CPU; preparing it did not reload the 2B checkpoint or measure performance.
