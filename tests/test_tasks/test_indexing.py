# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo import TEST_PATH
from judo.tasks.base import Task, TaskConfig


@dataclass
class TestConfig(TaskConfig):
    """Test config."""


class TestTask(Task[TestConfig]):
    """Test task."""

    def __init__(self, model_path: str = f"{TEST_PATH}/test_tasks/xml/test.xml") -> None:
        """Initalizes a test task."""
        super().__init__(model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: TestConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Blank reward function for testing."""
        return np.zeros(states.shape[0])


def test_joint_position_indexing() -> None:
    """Test to see if joint position index getter is working."""
    test_task = TestTask()
    assert test_task.get_joint_position_start_index("body_x") == 0
    assert test_task.get_joint_position_start_index("body_y") == 1
    assert test_task.get_joint_position_start_index("body_z") == 2


def test_joint_velocity_indexing() -> None:
    """Test to see if joint velocity index getter is working."""
    test_task = TestTask()
    assert test_task.get_joint_velocity_start_index("body_x") == 3
    assert test_task.get_joint_velocity_start_index("body_y") == 4
    assert test_task.get_joint_velocity_start_index("body_z") == 5


def test_sensor_indexing() -> None:
    """Test to see if sensor index getter is working."""
    test_task = TestTask()
    assert test_task.get_sensor_start_index("body_pos") == 0
