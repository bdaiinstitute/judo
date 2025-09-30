# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from threading import Lock

import pyarrow as pa
from dora_utils.dataclasses import from_event, to_arrow
from dora_utils.node import DoraNode, on_event
from omegaconf import DictConfig

from judo.app.data.controller_data import ControllerData
from judo.app.structs import MujocoState
from judo.controller import ControllerConfig


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
        self._data = ControllerData(
            init_task=init_task,
            init_optimizer=init_optimizer,
            task_registration_cfg=task_registration_cfg,
            optimizer_registration_cfg=optimizer_registration_cfg,
        )
        self.write_controls()
        self.lock = Lock()

    @on_event("INPUT", "task")
    def update_task(self, event: dict) -> None:
        """Updates the task type."""
        new_task = event["value"].to_numpy(zero_copy_only=False)[0]
        task_entry = self._data.available_tasks.get(new_task)
        if task_entry is not None:
            task_cls, task_config_cls = task_entry
            task = task_cls()
            task_config = task_config_cls()
            self._data.optimizer_config.set_override(new_task)
            optimizer = self._data.optimizer_cls(self._data.optimizer_config, task.nu)
            with self.lock:
                self._data.update_task(task, task_config, optimizer)
                self.write_controls()
        else:
            raise ValueError(f"Task {new_task} not found in task registry.")

    @on_event("INPUT", "task_reset")
    def reset_task(self, event: dict) -> None:
        """Resets the task."""
        with self.lock:
            self._data.reset_task()
            self.write_controls()

    @on_event("INPUT", "sim_pause")
    def set_paused_status(self, event: dict) -> None:
        """Event handler for processing pause status updates."""
        self._data.pause()

    @on_event("INPUT", "optimizer")
    def update_optimizer(self, event: dict) -> None:
        """Updates the optimizer type."""
        new_optimizer = event["value"].to_numpy(zero_copy_only=False)[0]
        optimizer_entry = self._data.available_optimizers.get(new_optimizer)
        if optimizer_entry is not None:
            optimizer_cls, optimizer_config_cls = optimizer_entry
            optimizer_config = optimizer_config_cls()
            optimizer = optimizer_cls(optimizer_config, self._data.task.nu)
            with self.lock:
                self._data.update_optimizer(optimizer, optimizer_config_cls, optimizer_config, optimizer_cls)
        else:
            raise ValueError(f"Optimizer {new_optimizer} not found in optimizer registry.")

    @on_event("INPUT", "controller_config")
    def update_controller_config(self, event: dict) -> None:
        """Callback to update controller config on receiving a new config message."""
        self._data.controller_config = from_event(event, ControllerConfig)
        self._data.controller.controller_cfg = self._data.controller_config

    @on_event("INPUT", "optimizer_config")
    def update_optimizer_config(self, event: dict) -> None:
        """Callback to update optimizer config on receiving a new config message."""
        self._data.update_optimizer_config(from_event(event, self._data.optimizer_config_cls))

    @on_event("INPUT", "task_config")
    def update_task_config(self, event: dict) -> None:
        """Callback to update optimizer task config on receiving a new config message."""
        self._data.task_config = from_event(event, type(self._data.task_config))
        self._data.task_config = self._data.task_config

    def write_controls(self) -> None:
        """Util that publishes the current controller spline."""
        # send control action
        arr, metadata = to_arrow(self._data.spline_data)
        self.node.send_output("controls", arr, metadata)

        # send traces
        if self._data.controller.traces is not None and len(self._data.controller.traces) > 0:
            metadata = {
                "all_traces_rollout_size": str(self._data.controller.all_traces_rollout_size),
                "shape": self._data.controller.traces.shape,
            }
            self.node.send_output("traces", pa.array(self._data.controller.traces.flatten()), metadata=metadata)

    @on_event("INPUT", "states")
    def update_states(self, event: dict) -> None:
        """Callback to update states on receiving a new state measurement."""
        state_msg = from_event(event, MujocoState)
        self._data.update_states(state_msg)

    def step(self) -> None:
        """Updates controls using current state info, and writes to /controls."""
        self._data.step()
        self.node.send_output("plan_time", pa.array([self._data.last_plan_time]))
        self.write_controls()

    def spin(self) -> None:
        """Spin logic for the controller node."""
        while True:
            start_time = time.time()
            self.parse_messages()
            self.step()

            # Force controller to run at fixed rate specified by control_freq.
            sleep_dt = 1 / self._data.controller_config.control_freq - (time.time() - start_time)
            time.sleep(max(0, sleep_dt))
