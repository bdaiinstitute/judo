# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import multiprocessing as mp
import random
from ctypes import c_bool
from dataclasses import dataclass
from operator import eq as scalar_eq
from time import sleep
from typing import Any, Callable

import numpy as np
import pytest
from pytest import raises

from core.unlocked import SIZEOF_SIZE_T, Frame, Memory, SharedMemoryQueue


@dataclass
class MyPair:
    first: float
    second: float


random.seed(785672498529)


def int_gen(size: int) -> list:
    return [random.randint(0, 1_000_000_000) for _ in range(size)]


def pair_gen(size: int) -> list:
    return [MyPair(first=random.uniform(0, 1), second=random.uniform(0, 1)) for _ in range(size)]


def array_gen(size: int) -> list:
    return [np.random.randint(1_000_000_000, size=(10, 10)) for _ in range(size)]


type_gen_params = [(int, int_gen, scalar_eq), (MyPair, pair_gen, scalar_eq), (np.array, array_gen, np.array_equal)]


@pytest.mark.parametrize("T, gen, eq", type_gen_params)
def test_memory(T: type, gen: Callable, eq: Callable) -> None:
    t = gen(1)[0]
    t_mem: Memory = Memory(t)
    t_bytes = bytearray(len(t_mem))
    t_mem.encode(t_bytes)
    tt = Memory.decode(t_bytes)
    assert t is not tt
    assert eq(t, tt)


def test_shared_array() -> None:
    t = np.arange(10)
    t_mem: Memory = Memory(t)
    t_bytes = bytearray(len(t_mem))
    t_mem.encode(t_bytes)
    tt1 = Memory.decode(t_bytes)
    tt2 = Memory.decode(t_bytes)
    assert tt1 is not tt2
    assert np.array_equal(tt1, tt2)
    tt1[...] = 42
    assert np.array_equal(tt1, tt2)


@pytest.mark.parametrize("T, gen, eq", type_gen_params)
def test_frame(T: type, gen: Callable, eq: Callable) -> None:
    values = gen(5)
    frame: Frame = Frame(name="test_frame_shm", create=True, size=1024 * 10)
    frame_view: Frame = Frame(name="test_frame_shm", create=False)

    assert len(frame) == 0
    assert len(frame_view) == 0
    assert frame.size() >= 1024 * 10
    assert frame.free() == frame.size() - SIZEOF_SIZE_T
    assert frame.used() == SIZEOF_SIZE_T

    for val in values:
        frame.push(val)

    assert len(frame) == 5
    assert len(frame_view) == 5
    assert frame.free() + frame.used() == frame.size()
    assert frame.used() > SIZEOF_SIZE_T

    for index in range(5):
        assert eq(frame[index], values[index]), f"error at index {index}"
        assert eq(frame_view[index], values[index]), f"view error at index {index}"


@pytest.mark.parametrize("T, gen, eq", type_gen_params)
def test_frame_multiprocessing(T: type, gen: Callable, eq: Callable) -> None:
    def producer(name: str, values: list, cond: Any, events: list) -> None:
        fail = False
        frame: Frame = Frame(name=name, create=True, size=1024 * 10)
        try:
            for val in values:
                frame.push(val)
            assert len(frame) == 5
            with cond:
                cond.notify_all()
            while not all([event.is_set() for event in events]):
                sleep(random.uniform(0.05, 0.1))
        except Exception as e:
            print(e)
            fail = True
        if fail:
            exit(1)

    def consumer(name: str, values: list, cond: Any, ind: int, events: list) -> None:
        fail = False
        frame: Frame = Frame(name=name, create=False)
        try:
            with cond:
                cond.wait_for(lambda: len(frame) == 5)
            assert len(frame) == 5
            for index in range(5):
                assert eq(frame[index], values[index])
            events[ind].set()
            while not all([event.is_set() for event in events]):
                sleep(random.uniform(0.05, 0.1))
        except Exception as e:
            print(e)
            events[ind].set()
            fail = True
        if fail:
            exit(1)

    name = "test_frame_multiprocessing"
    values = gen(5)
    sub_num = 10
    manager = mp.Manager()
    cond = manager.Condition()
    events = [manager.Event() for _ in range(sub_num)]

    procs = [mp.Process(target=producer, args=(name, values, cond, events), daemon=True)] + [
        mp.Process(target=consumer, args=(name, values, cond, ind, events), daemon=True) for ind in range(sub_num)
    ]

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    assert all((proc.exitcode == 0 for proc in procs))


@pytest.mark.parametrize("T, gen, eq", type_gen_params)
def test_shmque(T: type, gen: Callable, eq: Callable) -> None:
    n = 10
    values = gen(n)
    queue: SharedMemoryQueue = SharedMemoryQueue(name="test_shmque", min_rate=1)
    view1 = queue.make_view()
    view2 = queue.make_view()
    queue.open()
    view1.open()
    view2.open()

    test_time = 0.1
    delay = 2 * test_time / n

    for val in values:
        sleep(random.uniform(0, delay))
        queue.push(val)

    sleep(0.01)
    assert len(queue) == n
    assert len(view2) == n
    assert len(view2) == n

    for index in range(n):
        random_index = random.randint(0, n - 1)
        if view1.begin <= random_index and random_index < view1.end:
            assert eq(view1[random_index], values[random_index])
        else:
            with raises(IndexError):
                view1[random_index]

        assert eq(view1[view1.begin], values[index])
        assert eq(view1.top(), values[index])
        view1.pop()

    assert len(view1) == 0
    assert len(view2) == n
    assert len(queue) == n

    view2.pop(n)

    assert len(view2) == 0

    # wait for flush
    sleep(0.01)
    assert len(queue) < n

    view1.close()
    view2.close()
    queue.close()


@pytest.mark.parametrize("T, gen, eq", type_gen_params)
def test_shmque_stress(T: type, gen: Callable, eq: Callable) -> None:

    random.seed(47299561901461)

    sub_num = 10
    act_num = 100
    test_time = 0.1
    delay = 2 * test_time / act_num

    queue: SharedMemoryQueue = SharedMemoryQueue(name="test_shmque_stress", min_rate=1)
    views = [queue.make_view() for _ in range(sub_num)]

    values = gen(act_num)

    success = mp.RawArray(c_bool, [False] * (sub_num + 1))

    def publisher() -> None:
        queue.open()
        for val in values:
            sleep(random.uniform(0, delay))
            queue.push(val)
        queue.close()
        success[0] = True

    def subscriber(ind: int) -> None:
        view = views[ind]
        view.open()
        for val in values:
            while not view.wait():
                pass
            assert eq(view.top(), val)
            with raises(IndexError):
                view[view.begin - 1]
            with raises(IndexError):
                view[view.end]
            assert view.begin < view.end
            index = random.randint(view.begin, view.end - 1)
            assert eq(view[index], values[index])
            view.pop()
        view.close()
        success[ind + 1] = True

    procs = [mp.Process(target=publisher)] + [
        mp.Process(target=subscriber, args=(sub_ind,)) for sub_ind in range(sub_num)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert all(success)
