# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import pytest

from core.unlocked import Const as const


def test_simple_const() -> None:
    @dataclass
    class Simple:
        value: int

    const_simple = const(Simple(value=5))

    with pytest.raises(RuntimeError):
        const_simple.value = -5
    assert const_simple.value == 5

    simple = Simple(value=6)
    simple.value = 6
    assert simple.value == 6


def test_composed_const() -> None:
    @dataclass
    class Inner:
        value: int

    @dataclass
    class Outer:
        first: Inner
        second: Inner

    const_composed = const(Outer(first=Inner(value=4), second=Inner(value=2)))

    assert const_composed.first.value == 4
    with pytest.raises(RuntimeError):
        const_composed.second.value = -2
    assert const_composed.second.value == 2

    composed = Outer(first=Inner(value=6), second=Inner(value=6))

    assert composed.first.value == 6
    composed.second.value = -6
    assert composed.second.value == -6


def test_mutable_const() -> None:
    class Mutable:
        def __init__(self, value: Any):
            self.value = value

        def add_one(self) -> None:
            self.value += 1

    const_mutable = const(Mutable(value=3))

    with pytest.raises(RuntimeError):
        const_mutable.add_one()
    assert const_mutable.value == 3

    mutable = Mutable(value=7)

    mutable.add_one()
    assert mutable.value == 8
