# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time

import numpy as np

from judo.app.structs import SplineData
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.controller import Controller, ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks


class ControllerData:
    """Base class that encompasses the minimal amount of data for a controller node to access.

    This class is a small container which includes the data required for a node to run. This include configurations,
    a controller, task information, and an optimizer. Middleware nodes should inherit from this class and implement
    methods to send, process, and receive data.
    """

    def __init__(
        self,
        init_task: str,
        init_optimizer: str,
        task_registration_cfg: DictConfig | None = None,
        optimizer_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initializes basic attributes of a controller data container."""
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)
        if optimizer_registration_cfg is not None:
            register_optimizers_from_cfg(optimizer_registration_cfg)
        self.paused = False
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()
        self.setup(init_task, init_optimizer)
        self.last_plan_time = 0.0

    def setup(self, task_name: str, optimizer_name: str) -> None:
        """Sets up the controller data container with the given task and optimizer."""
        task_entry = self.available_tasks.get(task_name)
        optimizer_entry = self.available_optimizers.get(optimizer_name)

        assert task_entry is not None, f"Task {task_name} not found in registered tasks!"
        assert optimizer_entry is not None, f"Optimizer {optimizer_name} not found in registered optimizers!"

        # Set up the task
        self.task_cls, self.task_config_cls = task_entry
        self.task = self.task_cls()
        self.task_config = self.task_config_cls()

        # Set up the optimizer
        self.optimizer_cls, self.optimizer_config_cls = optimizer_entry
        self.optimizer_config = self.optimizer_config_cls()
        self.optimizer = self.optimizer_cls(self.optimizer_config, self.task.nu)

        # Set up the controller
        self.controller_config_cls = ControllerConfig
        self.controller_config = self.controller_config_cls()
        self.controller_config.set_override(task_name)
        self.controller = Controller(
            self.controller_config,
            self.task,
            self.task_config,
            self.optimizer,
            self.optimizer_config,
        )

        self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.curr_time = self.task.data.time

    def step(self) -> None:
        """Updates the controls state internally."""
        if self.states.shape != (self.controller.model.nq + self.controller.model.nv,):
            return
        elif self.paused:
            return

        start = time.perf_counter()
        self.controller.update_action(self.states, self.curr_time)
        end = time.perf_counter()
        self.last_plan_time = end - start

    @property
    def spline_data(self) -> np.ndarray:
        """Returns the spline data for the current state."""
        return SplineData(self.controller.times, self.controller.nominal_knots)
