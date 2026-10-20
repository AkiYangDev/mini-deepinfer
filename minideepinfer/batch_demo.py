"""Dependency-free trace of changing batch membership; no real model."""

from .batch_engine import BatchEngine


class CountingBatchRunner:
    def run_batch(self, steps):
        print("batch: " + ", ".join(f"{s.request_id}/{s.phase}" for s in steps))
        for step in steps:
            print(f"  {step.request_id}: inputs={step.input_ids}, slots={step.slot_mapping}")
        return tuple(step.input_ids[-1] + 1 if step.is_last_prompt_chunk else None for step in steps)

    def release(self, request_id):
        pass


def main():
    print("Batch control-flow simulation: no weights or tensor KV.\n")
    engine = BatchEngine(CountingBatchRunner(), cache_capacity=10, max_num_seqs=2)
    a = engine.submit([10, 11], max_new_tokens=1)
    b = engine.submit([20], max_new_tokens=3)
    engine.step()
    c = engine.submit([30, 31], max_new_tokens=2)
    while engine.step():
        pass
    for request in (a, b, c):
        print(f"{request.request_id}: {request.output_ids} ({request.finish_reason})")
    print(f"Free slots: {engine.cache.free_count}/{engine.cache.capacity}")


if __name__ == "__main__":
    main()
