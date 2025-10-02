# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from pytest import raises

from core.unlocked import Node


def test_annotation() -> None:
    def no_return() -> None:
        pass

    def return_int() -> int:
        return 0

    class My:
        """Empty test class."""

    def return_My_str() -> tuple[My, str]:
        return My(), "test"

    stage = Node("test_None", no_return, frequency=1).output_stage
    assert len(stage) == 0
    with raises(IndexError):
        stage[0]

    stage = Node("test_int", return_int, frequency=1).output_stage
    assert len(stage) == 1
    with raises(IndexError):
        stage[1]

    stage = Node("test_tuple", return_My_str, frequency=1).output_stage
    assert len(stage) == 2
    with raises(IndexError):
        stage[2]
