# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from judo.app.utils import register_tasks_from_cfg
from judo.tasks import get_registered_tasks
from judo.tasks.base import Task


class Simulation(ABC):
    """Base class for a simulation object.

    Middleware nodes should instantiate this class and implement methods to send, process, and receive data.
    """

    def __init__(
        self,
        init_task: str = "cylinder_push",
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the simulation."""
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)

        self.paused = False
        self.set_task(init_task)

    def set_task(self, task_name: str) -> None:
        """Initialize task from task name."""
        task_entry = get_registered_tasks().get(task_name)
        if task_entry is None:
            raise ValueError(f"Task {task_name} not found in task registry")

        task_cls, _ = task_entry
        self.task: Task = task_cls()
        self.task.reset()

    @abstractmethod
    def step(self, command: np.ndarray) -> None:
        """Step the simulation forward by one timestep.

        Args:
            command: Control command for this timestep.
        """

    def pause(self) -> None:
        """Toggle paused state."""
        self.paused = not self.paused

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Timestep the simulation expects to run at."""
