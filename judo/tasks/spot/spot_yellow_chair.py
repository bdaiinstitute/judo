# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import GOAL_POSITIONS, SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.utils.indexing import get_pos_indices, get_sensor_indices, get_vel_indices

XML_PATH = str(MODEL_PATH / "xml/spot_components/spot_yellow_chair.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
RESET_OBJECT_POSE = np.array([3, 0, 0.275, 1, 0, 0, 0])
# annulus object position sampling
RADIUS_MIN = 1.0
RADIUS_MAX = 1.5
USE_LEGS = False

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
DEFAULT_OBJECT_POS = np.array([0.0, 0.0])


@dataclass
class SpotYellowChairConfig(SpotBaseConfig):
    """Config for the spot box manipulation task."""

    goal_position: np.ndarray = field(default_factory=lambda: GOAL_POSITIONS().origin)
    w_fence: float = 1000.0
    w_orientation: float = 50.0
    orientation_sparsity: float = 5.0
    w_torso_proximity: float = 50.0
    torso_proximity_threshold: float = 1.0
    w_gripper_proximity: float = 8.0
    orientation_threshold: float = 0.7
    w_controls: float = 2.0
    w_object_velocity: float = 64.0
    position_tolerance: float = 0.2
    orientation_tolerance: float = 0.1


class SpotYellowChair(SpotBase[SpotYellowChairConfig]):
    """Task getting Spot to move a box to a desired goal location."""

    name: str = "spot_yellow_chair"
    config_t: type[SpotYellowChairConfig] = SpotYellowChairConfig

    def __init__(self, model_path: str = XML_PATH) -> None:
        """Initialize Spot yellow chair task.

        Args:
            model_path: Path to the XML model file
        """
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["yellow_chair_joint"])
        self.object_vel_idx = get_vel_indices(self.model, ["yellow_chair_joint"])
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")
        self.object_z_axis_idx = get_sensor_indices(self.model, "object_z_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot box moving task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]
        object_linear_velocity = qvel[..., self.object_vel_idx[0:3]]

        object_z_axis = sensors[..., self.object_z_axis_idx]
        gripper_to_object = sensors[..., self.end_effector_to_object_idx]

        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Compute l2 distance from object pos. to goal.
        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        orientation_alignement = np.minimum(
            np.dot(object_z_axis, Z_AXIS) - 1, self.config.orientation_threshold
        )  # ranging from -2 to 0
        object_orientation_reward = +self.config.w_orientation * np.exp(
            self.config.orientation_sparsity * orientation_alignement
        ).sum(axis=-1)  # ranging from 0 to w_orientation

        # Compute l2 distance from torso pos. to object pos.
        torso_proximity_reward = self.config.w_torso_proximity * np.minimum(
            self.config.torso_proximity_threshold, np.linalg.norm(body_pos - object_pos, axis=-1)
        ).mean(-1)

        # Compute l2 distance from torso pos. to object pos.
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            gripper_to_object,
            axis=-1,
        ).mean(-1)

        # Compute squared l2 norm of the object velocity.
        object_linear_velocity_reward = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert object_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            spot_fence_reward
            + spot_fallen_reward
            + goal_reward
            + object_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + object_linear_velocity_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        # radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        # theta = 2 * np.pi * np.random.rand()
        # object_pos = np.array([radius * np.cos(theta), radius * np.cos(theta)]) + np.random.randn(2)
        object_pos = DEFAULT_OBJECT_POS + np.random.randn(2) * 0.001
        # reset_object_pose = np.array([*object_pos, 0.254, 1, 0, 0, 0])
        # random_angle = 2 * np.pi * np.random.rand()
        reset_object_pose = np.array([*object_pos, 0.375, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, 0])
        spot_pos = DEFAULT_SPOT_POS + np.random.randn(2) * 0.001
        return np.array(
            [
                *spot_pos,
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *reset_object_pose,
            ]
        )

    def success(
        self, model: MjModel, data: MjData, config: SpotYellowChairConfig, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Check if the yellow chair is upright, regardless of position."""
        # Get object z-axis sensor data for orientation check
        object_z_axis = data.sensordata[self.object_z_axis_idx]

        # Check orientation tolerance (object should be upright, z-axis aligned with world z-axis)
        orientation_alignment = np.dot(object_z_axis, Z_AXIS)
        orientation_success = orientation_alignment >= (1.0 - config.orientation_tolerance)

        return bool(orientation_success)
