# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo Simulation with direct actuator control."""

import numpy as np
from mujoco import mj_step
from omegaconf import DictConfig

from judo.app.structs import MujocoState
from judo.simulation.base import Simulation


class MJSimulation(Simulation):
    """MuJoCo simulation with direct actuator control.

    Applies controls directly to MuJoCo actuators and steps
    the physics simulation forward.
    """

    def __init__(
        self,
        init_task: str = "spot_base",
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the MuJoCo simulation.

        Args:
            init_task: Name of the task to initialize.
            task_registration_cfg: Optional task registration configuration.
        """
        super().__init__(init_task=init_task, task_registration_cfg=task_registration_cfg)

    def step(self, command: np.ndarray) -> None:
        """Step the simulation forward.

        Args:
            command: Control array in task format (task.nu dimensions).
        """
        if self.paused:
            return

        command = self.task.task_to_sim_ctrl(command)
        self.task.data.ctrl[:] = command[: self.task.model.nu]
        self.task.pre_sim_step()
        mj_step(self.task.sim_model, self.task.data)
        self.task.post_sim_step()

    def set_task(self, task_name: str) -> None:
        """Set the current task.

        Args:
            task_name: Name of the task to set.
        """
        super().set_task(task_name)

    @property
    def sim_state(self) -> MujocoState:
        """Returns the current simulation state."""
        return MujocoState(
            time=self.task.data.time,
            qpos=self.task.data.qpos,
            qvel=self.task.data.qvel,
            xpos=self.task.data.xpos,
            xquat=self.task.data.xquat,
            mocap_pos=self.task.data.mocap_pos,
            mocap_quat=self.task.data.mocap_quat,
            sim_metadata=self.task.get_sim_metadata(),
        )

    @property
    def timestep(self) -> float:
        """Returns the effective simulation timestep (accounting for substeps)."""
        return self.task.dt
