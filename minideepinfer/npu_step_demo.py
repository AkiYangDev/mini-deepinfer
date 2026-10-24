"""Tiny Ascend event experiment; no weights, model forward, or performance test."""

import argparse
import hashlib
import json
from pathlib import Path
import platform


def run(device):
    import inspect
    import torch
    import torch_npu
    from .kv_cache import SlotPool
    from .model_runner import select_device
    from .npu_events import PendingTokens, RecordedEvent
    from .step_state import DeferredRelease, batch_remap

    select_device(device)
    torch.set_num_threads(1)
    producer, readback, consumer = (torch.npu.Stream() for _ in range(3))
    def event():
        return RecordedEvent(torch.npu.Event())

    # Setup is complete before the experiment. Keep strong references across
    # streams; do not let the allocator reuse storage while a reader is active.
    scores = torch.tensor([[0., 9., 0.], [0., 8., 0.]], device=device)
    offsets = torch.tensor([10, 20], dtype=torch.int64, device=device)
    sampled = torch.empty(2, dtype=torch.int64, device=device)
    kv = torch.tensor([10, 11, 20, 21, -1, -1], device=device)
    snapshot = torch.empty_like(kv)
    next_inputs, next_samples = torch.empty_like(sampled), torch.empty_like(sampled)
    mapping = batch_remap(("A", "B"), ("B", "C"))
    old_rows = torch.tensor([max(row, 0) for row in mapping], device=device)
    new_rows = torch.tensor([row < 0 for row in mapping], device=device)
    fresh = torch.tensor([0, 30], device=device)
    c_values = torch.tensor([30, 31], device=device)
    host_tokens = torch.empty(2, dtype=torch.int64, pin_memory=True)
    host_next = torch.empty_like(host_tokens, pin_memory=True)
    torch.npu.synchronize(device)

    pool, released = SlotPool(6), []
    a_slots, b_slots = pool.reserve("A", 2), pool.reserve("B", 2)
    cleanup = DeferredRelease(pool, released.append)
    ready, last_use, sampled_ready = event(), event(), event()
    previous = PendingTokens(("A", "B"), host_tokens, ready, last_use,
                             (kv, scores, offsets, sampled, host_tokens))
    cleanup.retain(previous)
    report = dict(python=platform.python_version(), torch=torch.__version__,
                  torch_npu=torch_npu.__version__, device=device,
                  raw_unrecorded_query=torch.npu.Event().query(),
                  guarded_unrecorded_query=ready.query(), remap=list(mapping))
    try:
        try:
            previous.inputs_for(("B", "C"), {"C": 30})
        except RuntimeError:
            report['early_host_read_blocked'] = True
        else:
            report['early_host_read_blocked'] = False
        with torch.npu.stream(producer):
            sampled.copy_(scores.argmax(-1) + offsets)
            sampled_ready.record(producer)
        cleanup.retire("A")  # Cancellation; this does not abort device work.
        c_slots = pool.reserve("C", 2)
        sampled_ready.wait(readback)
        with torch.npu.stream(readback):
            host_tokens.copy_(sampled, non_blocking=True)
            ready.record(readback)  # Includes the D2H copy, not just argmax.
        report['copy_query_before_wait'] = ready.query()
        ready.synchronize()
        report['host_inputs'] = list(previous.inputs_for(("B", "C"), {"C": 30}))
        previous.close()
        # last_use is not recorded yet. Native Event.query() would report True.
        report['release_before_last_use_record'] = list(cleanup.collect())
        report['free_before_last_use_record'] = pool.free_count

        next_done = event()
        following = PendingTokens(("B", "C"), host_next, next_done, next_done,
                                  (kv, sampled, old_rows, new_rows, fresh, next_inputs,
                                   next_samples, snapshot, c_values, host_next))
        cleanup.retain(following)
        sampled_ready.wait(consumer)
        with torch.npu.stream(consumer):
            next_inputs.copy_(torch.where(new_rows, fresh, sampled.index_select(0, old_rows)))
            kv[4:6].copy_(c_values)
            snapshot.copy_(kv)  # This consumer still reads A's old slots.
            next_samples.copy_(next_inputs + 1)  # Integer stand-in for the next model step.
            host_next.copy_(next_samples, non_blocking=True)
            next_done.record(consumer)
            last_use.record(consumer)
        cleanup.retire("B")
        cleanup.retire("C")
        last_use.synchronize()
        report['release_after_last_use'] = list(cleanup.collect())
        report['d_slots'] = list(pool.reserve("D", 2))
        kv[:2].fill_(99)
        torch.npu.current_stream().synchronize()  # D's write is synchronous here.
        report['old_snapshot'] = snapshot.cpu().tolist()
        report['reused_storage'] = kv.cpu().tolist()
        report['next_tokens'] = list(following.inputs_for(("B", "C"), {}))
        following.close()
        cleanup.retire("D")
        cleanup.collect()
        report.update(a_slots=list(a_slots), b_slots=list(b_slots), c_slots=list(c_slots),
                      released=released, free_slots=pool.free_count,
                      host_buffers_pinned=host_tokens.is_pinned() and host_next.is_pinned(),
                      allocated_bytes=torch.npu.memory_allocated(),
                      reserved_bytes=torch.npu.memory_reserved())
        report['passed'] = all((report['early_host_read_blocked'],
            not report['guarded_unrecorded_query'], report['host_buffers_pinned'],
            report['host_inputs'] == [21, 30], report['next_tokens'] == [22, 31],
            report['release_before_last_use_record'] == [], pool.free_count == 6,
            report['free_before_last_use_record'] == 0,
            report['release_after_last_use'] == ['A'], report['d_slots'] == list(a_slots),
            report['old_snapshot'] == [10, 11, 20, 21, 30, 31],
            report['reused_storage'] == [99, 99, 20, 21, 30, 31],
            released == ['A', 'B', 'C', 'D']))
        root = Path(__file__).parent
        report['source_sha256'] = {f'minideepinfer/{name}': hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in ('npu_step_demo.py', 'npu_events.py', 'step_state.py', 'kv_cache.py')}
        report['installed_streams_sha256'] = hashlib.sha256(
            Path(inspect.getfile(torch_npu.npu.Event)).read_bytes()).hexdigest()
        return report
    finally:
        # Even on a launch/copy error, retain local tensors until device work
        # finishes. If this raises, propagate the error; do not pretend to free.
        torch.npu.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--output', type=Path, default=Path('outputs/npu-events.json'))
    args = parser.parse_args()
    if not args.device.startswith('npu:'):
        parser.error('choose an explicit NPU device, such as npu:0')
    report = run(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"事件未记录时：原生 query={report['raw_unrecorded_query']}，保护后={report['guarded_unrecorded_query']}")
    print(f"[B, C] 接到的输入：{report['host_inputs']}；下一轮结果：{report['next_tokens']}")
    print(f"末次使用事件记录前回收：{report['release_before_last_use_record']}；完成后：{report['release_after_last_use']}")
    print(f"D 复用槽位：{report['d_slots']}；旧快照：{report['old_snapshot'][:2]}；新值：{report['reused_storage'][:2]}")
    print(f"检查通过={report['passed']}；槽位回收={report['free_slots']}/6")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
