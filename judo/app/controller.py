# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from threading import Lock

import numpy as np
import pyarrow as pa
from dora_utils.dataclasses import from_event, to_arrow
from dora_utils.node import DoraNode, on_event
from omegaconf import DictConfig

from judo.app.structs import MujocoState, SplineData
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.controller import Controller, ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks


class ControllerNode(DoraNode):
    """Controller node."""

    def __init__(
        self,
        init_task: str = "cylinder_push",
        init_optimizer: str = "cem",
        node_id: str = "controller",
        max_workers: int | None = None,
        task_registration_cfg: DictConfig | None = None,
        optimizer_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the controller node."""
        super().__init__(node_id=node_id, max_workers=max_workers)

        # handling custom task and optimizer registration
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)
        if optimizer_registration_cfg is not None:
            register_optimizers_from_cfg(optimizer_registration_cfg)

        self.paused = False
        self.lock = Lock()
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()
        self.setup(init_task, init_optimizer)

    def setup(self, task_name: str, optimizer_name: str) -> None:
        """Set up the task and optimizer for the controller."""
        task_entry = self.available_tasks.get(task_name)
        optimizer_entry = self.available_optimizers.get(optimizer_name)

        assert task_entry is not None, f"Task {task_name} not found in task registry."
        assert optimizer_entry is not None, f"Optimizer {optimizer_name} not found in optimizer registry."

        # instantiate the task/optimizer/controller
        self.task_cls, self.task_config_cls = task_entry
        self.optimizer_cls, self.optimizer_config_cls = optimizer_entry

        self.task = self.task_cls()
        self.task_config = self.task_config_cls()
        self.optimizer_config = self.optimizer_config_cls()
        self.optimizer = self.optimizer_cls(self.optimizer_config, self.task.nu)

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

        # Initialize the task data.
        self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
        self.curr_time = self.task.data.time

        # Write (default) controls onto the controls topic.
        self.write_controls()

    @on_event("INPUT", "task")
    def update_task(self, event: dict) -> None:
        """Updates the task type."""
        new_task = event["value"].to_numpy(zero_copy_only=False)[0]
        task_entry = self.available_tasks.get(new_task)
        if task_entry is not None:
            task_cls, task_config_cls = task_entry
            task = task_cls()
            task_config = task_config_cls()
            self.optimizer_config.set_override(new_task)
            optimizer = self.optimizer_cls(self.optimizer_config, task.nu)
            with self.lock:
                # update the task and optimizer
                self.task_cls = task_cls
                self.task_config_cls = task_config_cls
                self.task = task
                self.task_config = task_config
                self.optimizer = optimizer
                self.controller = Controller(
                    self.controller_config,
                    self.task,
                    self.task_config,
                    self.optimizer,
                    self.optimizer_config,
                )

                # because the task updated, we need to reinitialize the optimizer
                self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
                self.curr_time = self.task.data.time
                self.write_controls()
        else:
            raise ValueError(f"Task {new_task} not found in task registry.")

    @on_event("INPUT", "task_reset")
    def reset_task(self, event: dict) -> None:
        """Resets the task."""
        with self.lock:
            self.task.reset()
            self.controller.reset()
            self.states = np.concatenate([self.task.data.qpos, self.task.data.qvel])
            self.curr_time = self.task.data.time
            self.write_controls()

    @on_event("INPUT", "sim_pause")
    def set_paused_status(self, event: dict) -> None:
        """Event handler for processing pause status updates."""
        self.paused = not self.paused

    @on_event("INPUT", "optimizer")
    def update_optimizer(self, event: dict) -> None:
        """Updates the optimizer type."""
        new_optimizer = event["value"].to_numpy(zero_copy_only=False)[0]
        optimizer_entry = self.available_optimizers.get(new_optimizer)
        if optimizer_entry is not None:
            optimizer_cls, optimizer_config_cls = optimizer_entry
            optimizer_config = optimizer_config_cls()
            optimizer = optimizer_cls(optimizer_config, self.task.nu)
            with self.lock:
                self.optimizer = optimizer
                self.controller.optimizer = optimizer
                self.optimizer_config_cls = optimizer_config_cls
                self.optimizer_config = optimizer_config
                self.optimizer_cls = optimizer_cls
        else:
            raise ValueError(f"Optimizer {new_optimizer} not found in optimizer registry.")

    @on_event("INPUT", "controller_config")
    def update_controller_config(self, event: dict) -> None:
        """Callback to update controller config on receiving a new config message."""
        self.controller_config = from_event(event, self.controller_config_cls)
        self.controller.controller_cfg = self.controller_config

    @on_event("INPUT", "optimizer_config")
    def update_optimizer_config(self, event: dict) -> None:
        """Callback to update optimizer config on receiving a new config message."""
        self.optimizer_config = from_event(event, self.optimizer_config_cls)
        self.controller.optimizer_cfg = self.optimizer_config
        self.controller.optimizer.config = self.optimizer_config

    @on_event("INPUT", "task_config")
    def update_task_config(self, event: dict) -> None:
        """Callback to update optimizer task config on receiving a new config message."""
        self.controller.task_cfg = from_event(event, self.task_config_cls)
        self.task_config = self.controller.task_cfg

    def write_controls(self) -> None:
        """Util that publishes the current controller spline."""
        # send control action
        spline_data = SplineData(self.controller.times, self.controller.nominal_knots)
        arr, metadata = to_arrow(spline_data)
        self.node.send_output("controls", arr, metadata)

        # send traces
        if self.controller.traces is not None and len(self.controller.traces) > 0:
            metadata = {
                "all_traces_rollout_size": str(self.controller.all_traces_rollout_size),
                "shape": self.controller.traces.shape,
            }
            self.node.send_output("traces", pa.array(self.controller.traces.flatten()), metadata=metadata)

    @on_event("INPUT", "states")
    def update_states(self, event: dict) -> None:
        """Callback to update states on receiving a new state measurement."""
        state_msg = from_event(event, MujocoState)
        self.states = np.concatenate([state_msg.qpos, state_msg.qvel])
        self.curr_time = state_msg.time
        self.controller.system_metadata = state_msg.sim_metadata
        self.controller.task.time = state_msg.time  # sets the time in the task.data obj

    def step(self) -> None:
        """Updates controls using current state info, and writes to /controls."""
        if self.states.shape != (self.controller.model.nq + self.controller.model.nv,):
            # the task has changed and the new task has a different number of states
            return
        elif self.paused:
            return

        # always profile the controller update action
        start = time.perf_counter()
        self.controller.update_action(self.states, self.curr_time)
        end = time.perf_counter()
        self.node.send_output("plan_time", pa.array([end - start]))
        self.write_controls()

    def spin(self) -> None:
        """Spin logic for the controller node."""
        while True:
            start_time = time.time()
            self.parse_messages()
            self.step()

            # Force controller to run at fixed rate specified by control_freq.
            sleep_dt = 1 / self.controller_config.control_freq - (time.time() - start_time)
            time.sleep(max(0, sleep_dt))
