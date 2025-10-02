# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
# TODO(@bhung): We need to figure out how to properly test this.

import time
from typing import Any
import threading
import logging
from copy import copy

import numpy as np
from omegaconf import DictConfig

from judo.app.structs import MujocoState, SplineData
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.controller import Controller, ControllerConfig
from judo.optimizers import Optimizer, OptimizerConfig, OptimizerConfigType, OptimizerType, get_registered_optimizers
from judo.tasks import Task, TaskConfig, get_registered_tasks
from judo.tasks.base import TaskConfig
from judo.optimizers.base import OptimizerConfig
from judo.unlocked.node import _ADDED_SLEEP_DURATION


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
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)
        if optimizer_registration_cfg is not None:
            register_optimizers_from_cfg(optimizer_registration_cfg)
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()

    def update_task(
        self,
        task_name: str,
    ) -> None:
        """Updates the task, task config, and optimizer."""
        logging.info("Updating task to %s", task_name)
        task_entry = self.available_tasks.get(task_name)
        if task_entry is None:
            raise ValueError(f"Unknown task {task_name}")
        self.task_cls, self.task_config_cls = task_entry

        self.task = self.task_cls()
        self.task_config = self.task_config_cls()
        self._wait_for("optimizer_config")
        self.optimizer_config.set_override(task_name)
        self._wait_for("optimizer_cls")
        self.optimizer = self.optimizer_cls(self.optimizer_config, self.task.nu)

        self.controller = Controller(
            self.controller_config,
            self.task,
            self.task_config,
            self.optimizer,
            self.optimizer_config,
        )
        self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.curr_time = self.task.data.time
        logging.info("Task has been updated to %s", task_name)

    def update_optimizer(
        self,
        optimizer_name: str,
    ) -> None:
        """Updates the optimizer based on a given name."""
        logging.info("Updating optimizer to %s", optimizer_name)
        optimizer_entry = self.available_optimizers.get(optimizer_name)
        if optimizer_entry is None:
            raise ValueError(f"Unknown optimizer {optimizer_name}.")
        self.optimizer_cls, self.optimizer_config_cls = optimizer_entry
        self.optimizer_config = self.optimizer_config_cls()
        self._wait_for("task")
        self.optimizer = self.optimizer_cls(self.optimizer_config, self.task.nu)
        self._wait_for("controller")
        self.controller.optimizer = self.optimizer
        logging.info("Optimizer has been updated to %s", optimizer_name)


    def reset_task(self, _: int) -> None:
        """Resets the task and controller, setting the states to the default values."""
        logging.info("Resetting task")
        self._wait_for("task")
        self.task.reset()
        self._wait_for("controller")
        self.controller.reset()
        self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.curr_time = self.task.data.time

    def update_optimizer_config(self, optimizer_config: OptimizerConfig) -> None:
        """Updates the optimizer config."""
        logging.info("Updating optimizer config")
        self.optimizer_config = copy(optimizer_config)
        self._wait_for("controller")
        self.controller.optimizer.config = self.optimizer_config
        self.controller.optimizer_cfg = self.optimizer_config

    def update_task_config(self, task_config: TaskConfig) -> None:
        """Callback to update optimizer task config on receiving a new config message."""
        logging.info("Updating task config")
        self.task_config = copy(task_config)

    def update_controller_config(self, controller_config: ControllerConfig) -> None:
        """Callback to update controller config on receiving a new config message."""
        logging.info("Updating controller config")
        self.controller_config = copy(controller_config)
        self.delay_dt = 1. / self.controller_config.control_freq
        self._wait_for("controller")
        self.controller.controller_cfg = controller_config

    def step(self, state_msg: MujocoState) -> tuple[SplineData, tuple[np.ndarray, int], float]:
        """Updates the controls state internally."""
        self.states = np.concatenate([state_msg.qpos, state_msg.qvel])
        self.curr_time = state_msg.time
        self._wait_for("controller")
        self.controller.system_metadata = state_msg.sim_metadata
        self.controller.task.time = state_msg.time

        if self.states.shape != (self.controller.model.nq + self.controller.model.nv,):
            raise ValueError("State and model dimension mismatch")

        start = time.perf_counter()
        self.controller.update_action(self.states, self.curr_time)
        end = time.perf_counter()
        plan_time = end - start

        if plan_time + _ADDED_SLEEP_DURATION < self.delay_dt:
            time.sleep(self.delay_dt - plan_time - _ADDED_SLEEP_DURATION)

        control = SplineData(self.controller.times, self.controller.nominal_knots)

        return control, (self.controller.traces, self.controller.all_traces_rollout_size), plan_time

    def _wait_for(self, attr_name: str) -> None:
        while not hasattr(self, attr_name):
            time.sleep(0.1)

