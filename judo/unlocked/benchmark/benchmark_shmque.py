# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import multiprocessing as mp
from ctypes import c_bool
from time import perf_counter, sleep

from core.logging_util import setup_logger
from core.unlocked.ipc_utils import Event, SharedTimeStat
from core.unlocked.shmque import SIZEOF_SIZE_T, Frame, Memory, SharedMemoryQueue

setup_logger(log_level=logging.INFO)

n_trials = 1000
n_sub = 30
payload = b"\x00" * 3 * 1920 * 1080  # expected image payload size
payload = b"\x00" * 8 * 100  # expected state payload size


def benchmark_perf_counter() -> None:
    perf_time = SharedTimeStat("perf_time", lock=False)

    for _ in range(n_trials):  # publish more not to deprive views of data
        with perf_time:
            perf_counter()

    print(f"     Perf counter time: {perf_time}")


def benchmark_sleep() -> None:
    sleep_time = SharedTimeStat("sleep_time", lock=False)

    for _ in range(n_trials):  # publish more not to deprive views of data
        with sleep_time:
            sleep(1.0e-300)

    print(f"            Sleep time: {sleep_time}")


def benchmark_event() -> None:
    sync = Event("benchmarked")

    event_time = SharedTimeStat("event_time", lock=False)

    def push_fn() -> None:
        for _ in range(n_trials):  # publish more not to deprive views of data
            event_time.tic()
            sync.notify()

    push_process = mp.Process(target=push_fn)

    def view_fn() -> None:
        for _ in range(n_trials):
            with sync:
                event_time.toc()

    view_process = mp.Process(target=view_fn)
    view_process.start()
    push_process.start()
    push_process.join()
    view_process.join()

    print(f"            Event time: {event_time}")


def benchmark_memory_init() -> None:
    init_time = SharedTimeStat("init_time", lock=False)

    for i in range(n_trials):
        with init_time:
            Memory((i, payload))

    print(f"      Memory init time: {init_time}")


def benchmark_memory_encode_decode() -> None:
    memory: Memory = Memory((n_trials, payload))
    buffer = bytearray(len(memory))

    encode_time = SharedTimeStat("encode_time", lock=False)
    decode_time = SharedTimeStat("decode_time", lock=False)

    for _ in range(n_trials):
        with encode_time:
            memory.encode(buffer)
        with decode_time:
            value, _ = Memory.decode(buffer)
        assert value == n_trials

    print(f"    Memory encode time: {encode_time}")
    print(f"    Memory decode time: {decode_time}")


def benchmark_frame_init() -> None:
    size = 400_000_000  # expected memory rate for serializing large objects

    init_time = SharedTimeStat("init_time", lock=False)

    for _ in range(n_trials):
        with init_time:
            Frame(name="benchmark_frame_init", create=True, size=size)

    print(f"       Frame init time: {init_time}")


def benchmark_frame_push_memory() -> None:
    memory: Memory = Memory((n_trials, payload))

    frame: Frame = Frame(name="benchmark_frame_push_memory", create=True, size=(len(memory) + SIZEOF_SIZE_T) * n_trials)

    push_time = SharedTimeStat("push_time", lock=False)

    for i in range(n_trials):
        memory = Memory((i, payload))
        with push_time:
            frame.push_memory(memory)

    print(f"Frame push memory time: {push_time}")


def benchmark_frame_push() -> None:
    memory: Memory = Memory((n_trials, payload))

    frame: Frame = Frame(name="benchmark_frame_push", create=True, size=(len(memory) + SIZEOF_SIZE_T) * n_trials)

    push_time = SharedTimeStat("push_time", lock=False)

    for i in range(n_trials):
        with push_time:
            frame.push((i, payload))

    print(f"       Frame push time: {push_time}")


def benchmark_frame_get_item() -> None:
    memory: Memory = Memory((n_trials, payload))

    frame: Frame = Frame(name="benchmark_frame_get_item", create=True, size=(len(memory) + SIZEOF_SIZE_T) * n_trials)

    for i in range(n_trials):
        memory = Memory((i, payload))
        frame.push_memory(memory)

    get_time = SharedTimeStat("get_time", lock=False)

    success = mp.RawValue(c_bool, False)

    def get_process() -> None:
        frame_view: Frame = Frame(name="benchmark_frame_get_item", create=False)
        for i in range(n_trials):
            with get_time:
                value, _ = frame_view[i]
            if value != i:
                logging.error("Expected %d; instead, got %d", i, value)
                return
        success.value = True

    get_proc = mp.Process(target=get_process)
    get_proc.start()
    get_proc.join()

    assert success.value

    print(f"   Frame get item time: {get_time}")


def benchmark_queue_push() -> None:
    queue: SharedMemoryQueue = SharedMemoryQueue(name="benchmark_queue_push")
    queue.open()
    queue_push_time = SharedTimeStat("queue_push_time", lock=False)

    for i in range(n_trials):
        with queue_push_time:
            queue.push((i, payload))

    queue.close()

    print(f"       Queue push time: {queue_push_time}")


def benchmark_queue_view() -> None:
    queue: SharedMemoryQueue = SharedMemoryQueue(name="benchmark_queue_view")
    view = queue.make_view()

    queue.open()
    for i in range(n_trials):
        queue.push((i, payload))

    wait_time = SharedTimeStat("wait_time", lock=False)
    top_time = SharedTimeStat("top_time", lock=False)
    pop_time = SharedTimeStat("pop_time", lock=False)

    success = mp.RawValue(c_bool, False)

    def view_process() -> None:
        view.open()
        for i in range(n_trials):
            with wait_time:
                while not view.wait():
                    pass
            with top_time:
                value, _ = view.top()
            if value != i:
                logging.error("Expected %d; instead, got %d", i, value)
                view.close()
                return
            with pop_time:
                view.pop()
        view.close()
        success.value = True

    view_proc = mp.Process(target=view_process)
    view_proc.start()
    view_proc.join()

    assert success.value
    queue.close()

    print(f"  Queue view wait time: {wait_time}")
    print(f"   Queue view top time: {top_time}")
    print(f"   Queue view pop time: {pop_time}")


def benchmark_queue_transit() -> None:
    queue: SharedMemoryQueue = SharedMemoryQueue(name="benchmark_queue_transit")
    views = [queue.make_view() for _ in range(n_sub)]
    sync = mp.Barrier(n_sub + 1)

    transit_time = SharedTimeStat("transit_time", lock=False)
    view_wait_time = [SharedTimeStat(f"view_{sub_ind}_wait_time", lock=False) for sub_ind in range(n_sub)]

    def push_fn() -> None:
        queue.open()
        for i in range(n_trials):  # publish more not to deprive views of data
            # logging.info("Publisher iteration %d", i)
            queue.push((i, payload, transit_time.tic()))
        sync.wait()
        queue.close()

    push_process = mp.Process(target=push_fn, name="Push")

    def view_fn(sub_ind: int) -> None:
        view = views[sub_ind]
        view.open()
        for _ in range(n_trials):
            # logging.info("Subscriber %d iteration %d", sub_ind, i)
            with view_wait_time[sub_ind]:
                while not view.wait():
                    pass
            value, _, now = view.top()
            # logging.info("Subscriber %d value %d", sub_ind, value)
            view.pop()
            if sub_ind == 0:
                transit_time.tic(now)
                transit_time.toc()
        sync.wait()
        view.close()

    view_processes = [mp.Process(target=view_fn, args=(i,), name=f"View_{i}") for i in range(n_sub)]

    push_process.start()
    for view_proc in view_processes:
        view_proc.start()
    push_process.join()
    for view_proc in view_processes:
        view_proc.join()

    print(f"    Queue transit time: {transit_time}")
    # for sub_ind in range(n_sub):
    #     print( f"      View {sub_ind} wait time: {view_wait_time[sub_ind]}")


def main() -> None:
    benchmark_perf_counter()
    benchmark_sleep()
    benchmark_event()
    benchmark_memory_init()
    benchmark_memory_encode_decode()
    benchmark_frame_init()
    benchmark_frame_push_memory()
    benchmark_frame_push()
    benchmark_frame_get_item()
    benchmark_queue_push()
    benchmark_queue_view()
    benchmark_queue_transit()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
