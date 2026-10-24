"""A deterministic CPU lesson. Events are advanced by hand; no model is run."""

from .kv_cache import SlotPool
from .step_state import DeferredRelease, StepState, batch_remap


class ManualEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def complete(self):
        self.done = True


def main():
    print("CPU 演示：用整数代替 token，手动推进完成事件，无需模型。\n")
    pool = SlotPool(6)
    a_slots, b_slots = pool.reserve("A", 2), pool.reserve("B", 2)
    released = []
    cleanup = DeferredRelease(pool, released.append)
    ready, last_use = ManualEvent(), ManualEvent()
    previous = StepState(("A", "B"), (11, 21), ready, last_use)
    cleanup.retain(previous)
    cleanup.retire("A")  # A was cancelled while its step was in flight.
    print("1. [A, B] 还没算完，先取消 A。")
    print(f"   本次回收={cleanup.collect()}，空闲槽位={pool.free_count}/6")

    c_slots = pool.reserve("C", 2)
    current = ("B", "C")
    print(f"2. 准备 [B, C]：行号映射={batch_remap(previous.request_ids, current)}")
    print(f"   A 占用={a_slots}，B 占用={b_slots}，C 占用={c_slots}")
    try:
        previous.inputs_for(current, {"C": 30})
    except RuntimeError:
        print("   提前读取被拒绝：采样结果还没准备好。")

    ready.complete()
    inputs = previous.inputs_for(current, {"C": 30})
    print(f"3. 采样结果已就绪：B 接上 {inputs[0]}，C 使用自己的新输入 {inputs[1]}。")
    print(f"   本次回收={cleanup.collect()}，空闲槽位={pool.free_count}/6")
    previous.close()
    print("4. CPU 已读完，但还有一次设备拷贝在用旧状态。")
    print(f"   本次回收={cleanup.collect()}，空闲槽位={pool.free_count}/6")

    # Keep B protected before releasing the preceding state.
    next_ready, next_done = ManualEvent(), ManualEvent()
    following = StepState(current, tuple(token + 1 for token in inputs), next_ready, next_done)
    cleanup.retain(following)
    last_use.complete()
    print("5. 设备拷贝也完成了，现在才能归还 A 的槽位。")
    print(f"   本次回收={cleanup.collect()}，空闲槽位={pool.free_count}/6")
    print(f"   新请求 D 复用了 A 的槽位：{pool.reserve('D', 2)}")
    cleanup.retire("D")  # D has no device work in this example.
    cleanup.retire("B")
    cleanup.retire("C")
    next_ready.complete()
    next_done.complete()
    print(f"6. [B, C] 已算完，CPU 还没读结果：本次回收={cleanup.collect()}")
    print(f"   读出下一轮 token：{following.inputs_for(current, {})}")
    following.close()
    cleanup.collect()
    print(f"   每个请求只释放一次：{released}；空闲槽位={pool.free_count}/6")


if __name__ == "__main__":
    main()
