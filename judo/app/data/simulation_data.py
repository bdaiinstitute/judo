# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Callable
import logging
import time

from mujoco import mj_step
from omegaconf import DictConfig

from judo.app.structs import MujocoState, SplineData
from judo.app.utils import register_tasks_from_cfg
from judo.tasks import get_registered_tasks
from judo.tasks.base import Task, TaskConfig


class SimulationData:
    """Data container for the simulation node.

    This class is a small container which includes the data required for a simulation  node to run. This include
    configurations, a control spline, and task information. Middleware nodes should keep data within this class and
    implement methods to send, process, and receive data from here.
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

        self.control = None
        self.paused = False

    def update_task(self, task_name: str) -> None:
        """Helper to initialize task from task name."""
        logging.info("Updating task to %s", task_name)
        task_entry = get_registered_tasks().get(task_name)
        if task_entry is None:
            raise ValueError(f"Unknown task {task_name}")

        task_cls, task_config_cls = task_entry

        self.task: Task = task_cls()
        self.task_config: TaskConfig = task_config_cls()
        self.task.reset()
        logging.info("Task has been updated to %s", task_name)

    def reset_task(self, _: int) -> None:
        """Resets the task."""
        logging.info("Resetting task")
        self.task.reset()

    def pause(self, _: int) -> None:
        """Event handler for processing pause status updates."""
        self.paused = not self.paused

    def step(self, control: SplineData | None) -> MujocoState:
        """Step the simulation forward by one timestep."""
        if self.paused:
            raise ValueError("Simulation is paused.")
        if control is not None:
            self.control = control

        if self.control is not None:
            self._wait_for("task")
            try:
                self.task.data.ctrl[:] = self.control.spline()(self.task.data.time)
            except ValueError:
                pass
        self._wait_for("task")
        self.task.pre_sim_step()
        mj_step(self.task.sim_model, self.task.data)
        self.task.post_sim_step()

        return MujocoState(
            time=self.task.data.time,
            qpos=self.task.data.qpos,  # type: ignore
            qvel=self.task.data.qvel,  # type: ignore
            xpos=self.task.data.xpos,  # type: ignore
            xquat=self.task.data.xquat,  # type: ignore
            mocap_pos=self.task.data.mocap_pos,  # type: ignore
            mocap_quat=self.task.data.mocap_quat,  # type: ignore
            sim_metadata=self.task.get_sim_metadata(),
        )

    def _wait_for(self, attr_name: str) -> None:
        while not hasattr(self, attr_name):
            time.sleep(0.1)
