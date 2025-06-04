# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.math_utils import quat_diff, quat_diff_so3

XML_PATH = str(MODEL_PATH / "xml/leap_cube.xml")
SIM_XML_PATH = str(MODEL_PATH / "xml/leap_cube_sim.xml")
QPOS_HOME = np.array(
    [
        0.0, 0.03, 0.1, 1.0, 0.0, 0.0, 0.0,  # cube
        0.5, -0.75, 0.75, 0.25,  # index
        0.5, 0.0, 0.75, 0.25,  # middle
        0.5, 0.75, 0.75, 0.25,  # ring
        0.65, 0.9, 0.75, 0.6,  # thumb
    ]
)  # fmt: skip


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class LeapCubeConfig(TaskConfig):
    """Reward configuration LEAP cube rotation task."""

    w_pos: float = 100.0
    w_rot: float = 0.1


class LeapCube(Task[LeapCubeConfig]):
    """Defines the LEAP cube rotation task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = SIM_XML_PATH) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.goal_pos = np.array([0.0, 0.03, 0.1])
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.qpos_home = QPOS_HOME
        self.reset_command = np.array(
            [
                0.5, -0.75, 0.75, 0.25,  # index
                0.5, 0.0, 0.75, 0.25,  # middle
                0.5, 0.75, 0.75, 0.25,  # ring
                0.65, 0.9, 0.75, 0.6,  # thumb
            ]
        )  # fmt: skip
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: LeapCubeConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the LEAP cube rotation tracking task reward."""
        if system_metadata is None:
            system_metadata = {}
        goal_quat = system_metadata.get("goal_quat", np.array([1.0, 0.0, 0.0, 0.0]))

        # weights
        w_pos = config.w_pos
        w_rot = config.w_rot

        # "standard" tracking task
        qo_pos_traj = states[..., :3]
        qo_quat_traj = states[..., 3:7]
        qo_pos_diff = qo_pos_traj - self.goal_pos
        qo_quat_diff = quat_diff_so3(qo_quat_traj, goal_quat)

        pos_cost = w_pos * 0.5 * np.square(qo_pos_diff).sum(-1).mean(-1)
        rot_cost = w_rot * 0.5 * np.square(qo_quat_diff).sum(-1).mean(-1)
        rewards = -(pos_cost + rot_cost)
        return rewards

    def post_sim_step(self) -> None:
        """Checks if the cube has dropped and resets if so."""
        has_dropped = self.data.qpos[2] < -0.3

        # we reset here if the cube has dropped
        if has_dropped:
            self.reset()

        # check whether goal quat needs to be updated
        goal_quat = self.goal_quat
        q_diff = quat_diff(self.data.qpos[3:7], goal_quat)
        sin_a_2 = np.linalg.norm(q_diff[1:])
        angle = 2 * np.arctan2(sin_a_2, q_diff[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        at_goal = np.abs(angle) < 0.4
        if at_goal:
            self._update_goal_quat()

    def _update_goal_quat(self) -> None:
        """Updates the goal quaternion."""
        # generate uniformly random quaternion
        # https://stackoverflow.com/a/44031492
        uvw = np.random.rand(3)
        goal_quat = np.array(
            [
                np.sqrt(1 - uvw[0]) * np.sin(2 * np.pi * uvw[1]),
                np.sqrt(1 - uvw[0]) * np.cos(2 * np.pi * uvw[1]),
                np.sqrt(uvw[0]) * np.sin(2 * np.pi * uvw[2]),
                np.sqrt(uvw[0]) * np.cos(2 * np.pi * uvw[2]),
            ]
        )
        self.data.mocap_quat[0] = goal_quat
        self.goal_quat = goal_quat

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        self.data.qpos[:] = self.qpos_home
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.reset_command
        self._update_goal_quat()
        mujoco.mj_forward(self.model, self.data)

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns the simulation's goal quat."""
        return {"goal_quat": self.goal_quat}
