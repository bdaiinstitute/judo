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
                self.task.data.ctrl[:] = self.control(self.task.data.time)
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
            qpos=self.task.data.qpos,  # type: ignore
            qvel=self.task.data.qvel,  # type: ignore
            xpos=self.task.data.xpos,  # type: ignore
            xquat=self.task.data.xquat,  # type: ignore
            mocap_pos=self.task.data.mocap_pos,  # type: ignore
            mocap_quat=self.task.data.mocap_quat,  # type: ignore
            sim_metadata=self.task.get_sim_metadata(),
        )

    def _set_state(self) -> None:
        """Set the state of the simulation.

        Ignored in mujoco because all the data just comes from the task object and is never internally used.
        """
        pass

    @property
    def timestep(self) -> float:
        """Returns the simulation timestep."""
        return self.task.sim_model.opt.timestep
