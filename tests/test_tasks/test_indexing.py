# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from judo.tasks.base import Task, TaskConfig


@dataclass
class TestConfig(TaskConfig):
    """Test config."""


class TestTask(Task[TestConfig]):
    """Test task."""

    config_t: type[TestConfig] = TestConfig

    def __init__(self, model_path: str) -> None:
        """Initalizes a test task."""
        super().__init__(model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Blank reward function for testing."""
        return np.zeros(states.shape[0])


def test_joint_position_indexing(task_text_xml_path: Path) -> None:
    """Test to see if joint position index getter is working."""
    test_task = TestTask(model_path=str(task_text_xml_path))
    assert test_task.get_joint_position_start_index("body_x") == 0
    assert test_task.get_joint_position_start_index("body_y") == 1
    assert test_task.get_joint_position_start_index("body_z") == 2


def test_joint_velocity_indexing(task_text_xml_path: Path) -> None:
    """Test to see if joint velocity index getter is working."""
    test_task = TestTask(model_path=str(task_text_xml_path))
    assert test_task.get_joint_velocity_start_index("body_x") == 3
    assert test_task.get_joint_velocity_start_index("body_y") == 4
    assert test_task.get_joint_velocity_start_index("body_z") == 5


def test_sensor_indexing(task_text_xml_path: Path) -> None:
    """Test to see if sensor index getter is working."""
    test_task = TestTask(model_path=str(task_text_xml_path))
    assert test_task.get_sensor_start_index("body_pos") == 0
