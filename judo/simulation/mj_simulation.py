# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import cast

import numpy as np
from omegaconf import DictConfig

from judo.app.structs import MujocoState
from judo.simulation.base import Simulation


class MJSimulation(Simulation):
    """Mujoco simulation object.

    This class contains the data required to run a Mujoco simulation. This includes configurations, a control spline,
    and task information.

    Middleware nodes should instantiate this class and implement methods to send, process, and receive data.
    """

    def __init__(
        self,
        init_task: str = "cylinder_push",
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the simulation node."""
        super().__init__(init_task=init_task, task_registration_cfg=task_registration_cfg)
        # Initialize the simulation backend for the task
        self.sim_backend = self.task.SimBackend(self.task.task_to_sim_ctrl)

    def set_task(self, task_name: str) -> None:
        """Override to reinitialize sim backend when task changes."""
        super().set_task(task_name)
        self.sim_backend = self.task.SimBackend(self.task.task_to_sim_ctrl)

    def step(self) -> None:
        """Step the simulation forward by one timestep."""
        if self.control is not None and not self.paused:
            try:
                control_value = self.control(self.task.data.time)
                self.task.pre_sim_step()
                # Use task-specific simulation backend (handles control mapping internally)
                self.sim_backend.sim(self.task.sim_model, self.task.data, control_value)
                self.task.post_sim_step()
            except ValueError:
                # we're switching tasks and the new task has a different number of actuators
                pass

    @property
    def sim_state(self) -> MujocoState:
        """Returns the current simulation state."""
        return MujocoState(
            time=self.task.data.time,
            qpos=cast(np.ndarray, self.task.data.qpos),
            qvel=cast(np.ndarray, self.task.data.qvel),
            xpos=cast(np.ndarray, self.task.data.xpos),
            xquat=cast(np.ndarray, self.task.data.xquat),
            mocap_pos=cast(np.ndarray, self.task.data.mocap_pos),
            mocap_quat=cast(np.ndarray, self.task.data.mocap_quat),
            sim_metadata=self.task.get_sim_metadata(),
        )

    @property
    def timestep(self) -> float:
        """Returns the simulation timestep."""
        return self.task.sim_model.opt.timestep
