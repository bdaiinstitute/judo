# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Callable

from mujoco import mj_step
from omegaconf import DictConfig

from judo.app.structs import MujocoState
from judo.app.utils import register_tasks_from_cfg
from judo.tasks import get_registered_tasks
from judo.tasks.base import Task


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
        self.set_task(init_task)

    def set_task(self, task_name: str) -> None:
        """Helper to initialize task from task name."""
        task_entry = get_registered_tasks().get(task_name)
        if task_entry is None:
            raise ValueError(f"Init task {task_name} not found in task registry")

        task_cls, _ = task_entry
        self.task: Task = task_cls()
        self.reset_task()
        self._set_state()

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
                pass

        # Sets the internal state message based on the control and simuilation output
        self._set_state()

    def _set_state(self) -> None:
        """Set the state of the simulation."""
        self.sim_state = MujocoState(
            time=self.task.data.time,
            qpos=self.task.data.qpos,  # type: ignore
            qvel=self.task.data.qvel,  # type: ignore
            xpos=self.task.data.xpos,  # type: ignore
            xquat=self.task.data.xquat,  # type: ignore
            mocap_pos=self.task.data.mocap_pos,  # type: ignore
            mocap_quat=self.task.data.mocap_quat,  # type: ignore
            sim_metadata=self.task.get_sim_metadata(),
        )

    def pause(self) -> None:
        """Event handler for processing pause status updates."""
        self.paused = not self.paused

    def reset_task(self) -> None:
        """Resets the task."""
        self.task.reset()

    def update_control(self, control_spline: Callable) -> None:
        """Event handler for processing controls received from controller node."""
        self.control = control_spline
