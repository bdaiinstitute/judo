# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from mujoco import mj_step
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

    def step(self) -> None:
        """Step the simulation forward by one timestep."""
        if self.control is not None and not self.paused:
            try:
                control_input = self.control(self.task.data.time)
                processed_control = self.task.task_to_sim_ctrl(control_input)
                self.task.data.ctrl[:] = processed_control
                self.task.pre_sim_step()
                mj_step(self.task.sim_model, self.task.data)
                self.task.post_sim_step()
            except ValueError:
                # we're switching tasks and the new task has a different number of actuators
                pass

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
        """Returns the simulation timestep."""
        return self.task.sim_model.opt.timestep
