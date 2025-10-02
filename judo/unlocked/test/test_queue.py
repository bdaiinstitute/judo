# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import random
from dataclasses import dataclass
from enum import Enum, auto
from threading import Thread
from time import sleep

import pytest
from pytest import raises

from core.unlocked import Queue, QueueView


@dataclass
class MyInt:
    value: int


type_params = [int, MyInt]


@pytest.mark.parametrize("T", type_params)
def test_queue(T: type) -> None:
    q: Queue = Queue()
    q.push(T(8))
    q.push(T(7))
    assert q[0] == T(8)
    assert q[1] == T(7)
    assert q.top() == T(8)
    assert q.pop() == T(8)
    assert q[1] == T(7)
    with raises(IndexError):
        q[0]
    assert q.pop() == T(7)
    with raises(IndexError):
        q[0]
    with raises(IndexError):
        q[1]
    with raises(IndexError):
        q.pop()
    q.push(T(3))
    with raises(IndexError):
        q[0]
    with raises(IndexError):
        q[1]
    assert q[2] == T(3)


@pytest.mark.parametrize("T", type_params)
def test_queue_view(T: type) -> None:
    q: Queue = Queue()
    qw = q.make_view()

    q.push(T(8))
    q.push(T(7))
    with raises(IndexError):
        qw.last()
    with raises(IndexError):
        qw.top()
    with raises(IndexError):
        qw.pop()
    qw.flush()
    q.flush()
    assert len(q) == 2
    assert qw.top() == T(8)
    assert qw.last() == T(7)
    assert qw.pop() == T(8)
    assert qw.top() == T(7)
    assert qw.last() == T(7)
    q.push(T(3))
    assert len(q) == 3
    assert qw.top() == T(7)
    assert qw.last() == T(7)
    q.flush()
    assert len(q) == 2
    qw.flush(flush_queue=True)
    assert len(q) == 1
    assert qw.pop() == T(3)
    q.flush()
    assert len(q) == 0


@pytest.mark.parametrize("T", type_params)
def test_queue_stress(T: type) -> None:
    q: Queue = Queue()

    random.seed(83285020238474093)

    class Act(Enum):
        Push = auto()
        Pop = auto()
        Item = auto()

    test_num = 100
    act_num = 1000

    for _ in range(test_num):
        r_seq = [random.randint(0, 1_000_000_000) for _ in range(act_num)]
        push_pos = 0
        pop_pos = 0
        q_len = 0
        q.clear()
        for act in random.choices(list(Act.__members__.values()), k=act_num):
            match act:
                case Act.Push:
                    q.push(T(r_seq[push_pos]))
                    push_pos += 1
                    q_len += 1
                case Act.Pop:
                    if q_len > 0:
                        assert q.pop() == T(r_seq[pop_pos])
                        pop_pos += 1
                        q_len -= 1
                    else:
                        with raises(IndexError):
                            q.pop()
                case Act.Item:
                    i = random.randint(0, act_num)
                    if q.begin <= i and i < q.end:
                        assert q[i] == T(r_seq[i])
                    else:
                        with raises(IndexError):
                            q[i]

        assert len(q) == q_len


@pytest.mark.parametrize("T", type_params)
def test_queue_threaded_stress(T: type) -> None:
    q: Queue = Queue()
    sub_num = 10
    views = [q.make_view() for _ in range(sub_num)]

    random.seed(83285020238474093)

    test_num = 10
    act_num = 100
    test_time = 0.5
    delay = 2 * test_time / test_num / act_num

    publisher_success = False
    subscriber_success = [False] * sub_num

    for _ in range(test_num):
        r_seq = [random.randint(0, 1_000_000_000) for _ in range(act_num)]

        publisher_success = False
        for i in range(sub_num):
            subscriber_success[i] = False

        def publisher(r_seq: list) -> None:
            nonlocal publisher_success
            for r in r_seq:
                q.push(T(r))
                sleep(random.uniform(0, delay))
            publisher_success = True

        def subscriber(r_seq: list, sub_ind: int, qw: QueueView) -> None:
            nonlocal subscriber_success
            for r in r_seq:
                if len(qw) == 0:
                    while not qw.ready:
                        # if we use the following then the queue view is always of length 1
                        # qw.wait(random.uniform(0, delay))
                        # for stress testing we will let it accumulate a bit
                        sleep(random.uniform(0, delay))
                    qw.flush()
                assert qw.top() == T(r)
                with raises(IndexError):
                    qw[qw.begin - 1]
                with raises(IndexError):
                    qw[qw.end]
                assert qw.begin < qw.end
                index = random.randint(qw.begin, qw.end - 1)
                assert qw[index] == T(r_seq[index])
                for el, ex in zip(qw, (r_seq[i] for i in range(qw.begin, qw.end)), strict=True):
                    assert el == T(ex)
                assert qw.pop() == T(r)
            subscriber_success[sub_ind] = True

        threads = [Thread(target=publisher, args=(r_seq,))]
        for sub_ind in range(sub_num):
            threads.append(Thread(target=subscriber, args=(r_seq, sub_ind, views[sub_ind])))

        q.clear()

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert publisher_success and all(subscriber_success)
