"""Dependency-free control-flow demo. No language model or tensor KV cache."""

from .engine import Engine, Step


class CountingRunner:
    """Predict last_input + 1; arithmetic makes the execution trace easy to read."""

    def run(self, step: Step) -> int:
        print(
            f"{step.request_id} {step.phase:7} input={step.input_ids} "
            f"positions={step.positions} write_slots={step.slot_mapping}"
        )
        return step.input_ids[-1] + 1


def main() -> None:
    print("Control-flow simulation: no model weights, NPU, or tensor KV.\n")
    engine = Engine(CountingRunner(), cache_capacity=8)
    first = engine.submit([10, 11], max_new_tokens=3)
    second = engine.submit([20], max_new_tokens=4, eos_token_id=22)
    for output in engine.run():
        print(
            f"  -> token={output.token_id}, cached_inputs={output.num_cached_tokens}, "
            f"stop={output.finish_reason or '-'}"
        )
    for request in (first, second):
        print(f"{request.request_id}: {request.output_ids} ({request.finish_reason})")
    print(f"Free slots: {engine.cache.free_count}/{engine.cache.capacity}")


if __name__ == "__main__":
    main()
