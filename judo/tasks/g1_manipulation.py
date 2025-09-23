# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.math_utils import quat_diff, quat_diff_so3

XML_PATH = str(MODEL_PATH / "xml/g1_manipulation.xml")
QPOS_HOME = np.array(
    [
      0, 0, 0,
      0.2, 0.2, 0, 0, 0, 0, 0,
      0, 1.05, 0, 0, 0, 0, 0,
      0.2, -0.2, 0, 0, 0, 0, 0,
      0, -1.05, 0, 0, 0, 0, 0,
      0.4, 0, 1.0,
      1, 0, 0, 0
    ]
)  # fmt: skip


@slider("w_object_pos", 0.0, 10.0, 1.00)
@slider("w_object_rot", 0.0, 10.0, 0.0)
@slider("w_hand_proximity", 0.0, 10.0, 0.1)
@dataclass
class G1ManipulationConfig(TaskConfig):
    """Reward configuration G1 task."""

    cutoff_time: float = 0.25
    w_object_pos: float = 0.05
    w_object_rot: float = 0.2
    w_hand_proximity: float = 0.1

    dropped_threshold: float = 0.7
    too_far_threshold: float = 0.6
    angle_threshold: float = 0.2  # radians


class G1Manipulation(Task[G1ManipulationConfig]):
    """Defines the G1 manipulation task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the G1 manipulation task."""
        super().__init__(model_path, sim_model_path=sim_model_path)

        # object indices
        self.object_pose_adr = self.get_joint_position_start_index("box_joint")
        self.object_pos_slice = slice(self.object_pose_adr + 0, self.object_pose_adr + 3)
        self.object_quat_slice = slice(self.object_pose_adr + 3, self.object_pose_adr + 7)

        # robot indices
        self.joint_pos_idx = np.arange(31)
        self.finger_names = [
            "left_hand_thumb",
            "left_hand_index",
            "left_hand_middle",
            "right_hand_thumb",
            "right_hand_index",
            "right_hand_middle",
        ]
        self.num_fingers = len(self.finger_names)

        # sensors
        self.finger_sensor_indices = [self.get_sensor_start_index(finger_name) for finger_name in self.finger_names]

        # dynamic goal pose
        self.system_metadata: dict[str, Any] = {
            "goal_pos": np.array([0.5, 0.0, 0.85]),
            "goal_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }

        self.reset()

    def pre_rollout(self, curr_state: np.ndarray, config: G1ManipulationConfig) -> None:
        """Check for dropped objects or goal reached."""
        has_dropped = self.data.qpos[self.object_pose_adr + 2] < config.dropped_threshold
        is_too_far = (
            np.linalg.norm(self.data.qpos[self.object_pos_slice][:2] - self.system_metadata["goal_pos"][:2])
            > config.too_far_threshold
        )

        # we reset here if the cube has dropped
        if has_dropped or is_too_far:
            self.reset()

        # check whether goal quat needs to be updated
        goal_quat = self.system_metadata["goal_quat"]
        q_diff = quat_diff(self.data.qpos[self.object_quat_slice], goal_quat)
        sin_a_2 = np.linalg.norm(q_diff[1:])
        angle = 2 * np.arctan2(sin_a_2, q_diff[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        at_goal = np.abs(angle) < config.angle_threshold
        if at_goal:
            self._update_goal_quat()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: G1ManipulationConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the G1 manipulation task reward."""
        if system_metadata is None:
            system_metadata = {}
        goal_pos = system_metadata.get("goal_pos", np.array([0.5, 0.0, 0.85]))
        goal_quat = system_metadata.get("goal_quat", np.array([1.0, 0.0, 0.0, 0.0]))

        # weights
        w_object_pos = config.w_object_pos
        w_object_rot = config.w_object_rot
        w_hand_proximity = config.w_hand_proximity
        # "standard" tracking task
        object_pos_traj = states[..., self.object_pos_slice]
        object_quat_traj = states[..., self.object_quat_slice]

        object_pos_diff = object_pos_traj[..., 0:2] - goal_pos[0:2]  # do not penalize z-axis
        object_quat_diff = quat_diff_so3(object_quat_traj, goal_quat)

        object_pos_cost = w_object_pos * 0.5 * np.square(object_pos_diff).sum(-1).mean(-1)
        object_quat_cost = w_object_rot * 0.5 * np.square(object_quat_diff).sum(-1).mean(-1)
        rewards = -(object_pos_cost + object_quat_cost)

        hand_proximity_cost = np.zeros((states.shape[0],))
        for finger_idx in self.finger_sensor_indices:
            hand_proximity_diff = object_pos_traj - sensors[..., finger_idx : finger_idx + 3]
            hand_proximity_cost += (
                w_hand_proximity * 0.5 * np.square(hand_proximity_diff).sum(-1).mean(-1) / self.num_fingers
            )

        rewards = -hand_proximity_cost - object_pos_cost - object_quat_cost
        return rewards

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        self.data.qpos[:] = QPOS_HOME
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.reset_command
        self._update_goal_quat()
        mujoco.mj_forward(self.model, self.data)

    @property
    def reset_command(self) -> np.ndarray:
        """Returns the reset command for the robot."""
        return self.data.qpos[self.joint_pos_idx]

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
        self.system_metadata["goal_quat"] = goal_quat

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Mujoco actuator limits for this task."""
        limits = self.model.actuator_ctrlrange.copy()
        limits[0:3, 0] = -0.001
        limits[0:3, 1] = +0.001
        return limits  # type: ignore
