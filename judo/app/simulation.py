# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import threading
import time
import warnings

from dora_utils.dataclasses import from_arrow, to_arrow
from dora_utils.node import DoraNode, on_event
from mujoco import mj_step
from omegaconf import DictConfig

from judo.app.structs import MujocoState, SplineData
from judo.app.utils import register_tasks_from_cfg
from judo.tasks import get_registered_tasks
from judo.tasks.base import Task


class SimulationNode(DoraNode):
    """The simulation node."""

    def __init__(
        self,
        node_id: str = "simulation",
        init_task: str = "cylinder_push",
        max_workers: int | None = None,
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the simulation node."""
        super().__init__(node_id=node_id, max_workers=max_workers)

        # handling custom task registration
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)

        self.task_reset_lock = threading.Lock()
        self.config_lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.control = None
        self.paused = False
        self.set_task(init_task)
        self.write_states()

    def set_task(self, task_name: str) -> None:
        """Helper to initialize task from task name."""
        task_entry = get_registered_tasks().get(task_name)
        if task_entry is None:
            raise ValueError(f"Init task {task_name} not found in task registry")

        task_cls, task_config_cls = task_entry

        self.task: Task = task_cls()
        self.task_config = task_config_cls()
        self.task.reset()

    @on_event("INPUT", "task")
    def update_task(self, event: dict) -> None:
        """Event handler for processing task updates."""
        new_task = event["value"].to_numpy(zero_copy_only=False)[0]
        self.set_task(new_task)

    def step(self) -> None:
        """Step the simulation forward by one timestep."""
        if self.control is not None and not self.paused:
            try:
                self.task.data.ctrl[:] = self.control(self.task.data.time)
                self.task.pre_sim_step()
                mj_step(self.task.sim_model, self.task.data)
                self.task.post_sim_step()
            except ValueError:
                # we're switching tasks and the new task has a different number of actuators
                return

    def spin(self) -> None:
        """Spin logic for the simulation node."""
        while True:
            start_time = time.time()
            self.parse_messages()
            self.step()
            self.write_states()

            # Force controller to run at fixed rate specified by model dt.
            dt_des = self.task.sim_model.opt.timestep
            dt_elapsed = time.time() - start_time
            if dt_elapsed < dt_des:
                time.sleep(dt_des - dt_elapsed)
            else:
                warnings.warn(
                    f"Sim step {dt_elapsed:.3f} longer than desired step {dt_des:.3f}!",
                    stacklevel=2,
                )

    def write_states(self) -> None:
        """Reads data from simulation and writes to output topic."""
        sim_state = MujocoState(
            time=self.task.data.time,
            qpos=self.task.data.qpos,  # type: ignore
            qvel=self.task.data.qvel,  # type: ignore
            xpos=self.task.data.xpos,  # type: ignore
            xquat=self.task.data.xquat,  # type: ignore
            mocap_pos=self.task.data.mocap_pos,  # type: ignore
            mocap_quat=self.task.data.mocap_quat,  # type: ignore
            sim_metadata=self.task.get_sim_metadata(),
        )
        arr, metadata = to_arrow(sim_state)
        self.node.send_output("states", arr, metadata)

    @on_event("INPUT", "sim_pause")
    def set_paused_status(self, event: dict) -> None:
        """Event handler for processing pause status updates."""
        self.paused = not self.paused

    @on_event("INPUT", "task_reset")
    def reset_task(self, event: dict) -> None:
        """Resets the task."""
        with self.task_reset_lock:
            self.task.reset()

    @on_event("INPUT", "controls")
    def update_control(self, event: dict) -> None:
        """Event handler for processing controls received from controller node."""
        spline_data = from_arrow(event["value"], event["metadata"], SplineData)
        control = spline_data.spline()
        with self.control_lock:
            self.control = control
