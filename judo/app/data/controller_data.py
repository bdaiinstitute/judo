# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
# TODO(@bhung): We need to figure out how to properly test this.

import time
from typing import Any

import numpy as np
from omegaconf import DictConfig

from judo.app.structs import MujocoState, SplineData
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.controller import Controller, ControllerConfig
from judo.optimizers import Optimizer, OptimizerConfig, get_registered_optimizers
from judo.tasks import Task, TaskConfig, get_registered_tasks


class ControllerData:
    """Base class that encompasses the minimal amount of data for a controller node to access.

    This class is a small container which includes the data required for a node to run. This include configurations,
    a controller, task information, and an optimizer. Middleware nodes should keep data within this class and
    implement methods to send, process, and receive data from here.
    """

    def __init__(
        self,
        init_task: str,
        init_optimizer: str,
        task_registration_cfg: DictConfig | None = None,
        optimizer_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initializes basic attributes of a controller data container."""
        self.task_name = init_task
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)
        if optimizer_registration_cfg is not None:
            register_optimizers_from_cfg(optimizer_registration_cfg)
        self.last_plan_time = 0.0
        self.paused = False
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()
        self._setup(init_task, init_optimizer)

    def _setup(self, task_name: str, optimizer_name: str) -> None:
        """Set up the task and optimizer for the controller."""
        task_entry = self.available_tasks.get(task_name)
        optimizer_entry = self.available_optimizers.get(optimizer_name)

        assert task_entry is not None, f"Task {task_name} not found in task registry."
        assert optimizer_entry is not None, f"Optimizer {optimizer_name} not found in optimizer registry."

        # instantiate the task/optimizer/controller
        task_cls, task_config_cls = task_entry
        optimizer_cls, optimizer_config_cls = optimizer_entry

        self.task = task_cls()
        optimizer_config = optimizer_config_cls()
        optimizer = optimizer_cls(optimizer_config, self.task.nu)

        self.controller_config = ControllerConfig()
        self.controller_config.set_override(task_name)
        self.controller = Controller(
            self.controller_config,
            self.task,
            optimizer,
        )

        # Initialize the task data.
        self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.curr_time = self.task.data.time

    @property
    def task_config(self) -> TaskConfig:
        """Returns the task config, which is uniquely defined by the task."""
        return self.task.config

    @property
    def optimizer(self) -> Optimizer:
        """Returns the optimizer, which is uniquely defined by the controller."""
        return self.controller.optimizer

    @property
    def optimizer_config(self) -> OptimizerConfig:
        """Returns the optimizer config, which is uniquely defined by the controller."""
        return self.controller.optimizer_cfg

    @property
    def optimizer_cls(self) -> type:
        """Returns the optimizer class."""
        return self.controller.optimizer.__class__

    @property
    def optimizer_config_cls(self) -> type:
        """Returns the optimizer config class."""
        return self.controller.optimizer_cfg.__class__

    def update_task(
        self,
        task: Task,
        optimizer: Optimizer,
    ) -> None:
        """Updates the task, task config, and optimizer.

        Args:
            task_cls: The class of the task.
            task_config_cls: The class of the task config.
            task: The task instance.
            task_config: The task config instance.
            optimizer: The optimizer instance.
        """
        self.task = task
        self.controller = Controller(
            self.controller_config,
            self.task,
            optimizer,
        )
        self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.curr_time = self.task.data.time

    def reset_task(self) -> None:
        """Resets the task and controller, setting the states to the default values."""
        self.task.reset()
        self.controller.reset()
        self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.curr_time = self.task.data.time

    def update_optimizer_config(self, optimizer_cfg: Any) -> None:
        """Updates the optimizer config."""
        self.controller.optimizer.config = optimizer_cfg

    def update_states(self, state_msg: MujocoState) -> None:
        """Updates the states."""
        self.states = np.concatenate([state_msg.qpos, state_msg.qvel])
        self.curr_time = state_msg.time
        self.controller.system_metadata = state_msg.sim_metadata
        self.controller.task.time = state_msg.time

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

    def pause(self) -> None:
        """Pauses or starts the controller."""
        self.paused = not self.paused

    def update_optimizer(
        self,
        optimizer: Optimizer,
    ) -> None:
        """Updates the optimizer based on a given name."""
        self.controller.optimizer = optimizer

    @property
    def spline_data(self) -> SplineData:
        """Returns the spline data for the current state."""
        return SplineData(self.controller.times, self.controller.nominal_knots)
