# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC, abstractmethod
from typing import Callable

from omegaconf import DictConfig

from judo.app.utils import register_tasks_from_cfg
from judo.tasks import get_registered_tasks
from judo.tasks.base import Task


class Simulation(ABC):
    """Base class for a simulation object.

    This class contains the data required to run a simulation. This includes configurations, a control spline, and task
    information. It can be inherited from to implement specific simulation backends.

    Middleware nodes should instantiate this class and implement methods to send, process, and receive data.
    """

    def __init__(
        self,
        init_task: str = "cylinder_push",
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the simulation node."""
        # handling custom task registration
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)

        self.control: Callable | None = None
        self.paused = False
        self.set_task(init_task)

    def set_task(self, task_name: str) -> None:
        """Helper to initialize task from task name."""
        task_entry = get_registered_tasks().get(task_name)
        if task_entry is None:
            raise ValueError(f"Init task {task_name} not found in task registry")

        task_cls, _ = task_entry
        self.task: Task = task_cls()
        self.task.reset()

    @abstractmethod
    def step(self) -> None:
        """Step the simulation forward by one timestep."""

    def pause(self) -> None:
        """Event handler for processing pause status updates."""
        self.paused = not self.paused

    def update_control(self, control_spline: Callable) -> None:
        """Event handler for processing controls received from controller node."""
        self.control = control_spline

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Timestep the simulation expects to run at."""
