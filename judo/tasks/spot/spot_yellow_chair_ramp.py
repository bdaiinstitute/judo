# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from judo import MODEL_PATH
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.utils.indexing import get_pos_indices, get_sensor_indices, get_vel_indices

XML_PATH = str(MODEL_PATH / "xml/spot_components/spot_yellow_chair_ramp.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
RESET_OBJECT_POSE = np.array([3, 0, 0.275, 1, 0, 0, 0])
# annulus object position sampling
RADIUS_MIN = 0.1
RADIUS_MAX = 0.5
USE_LEGS = False
DEFAULT_GOAL = np.array([2.0, 4.5, 0.256])

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
DEFAULT_OBJECT_POS = np.array([0.0, 0.0])


@dataclass
class SpotYellowChairRampConfig(SpotBaseConfig):
    """Config for the spot box manipulation task."""

    goal_position: np.ndarray = field(default_factory=lambda: DEFAULT_GOAL)
    w_orientation: float = 100.0
    w_torso_proximity: float = 0.25
    w_gripper_proximity: float = 4.0
    orientation_threshold: float = 0.4
    w_controls: float = 2.0
    w_goal: float = 100.0
    w_object_velocity: float = 128.0
    fall_penalty: float = 10000.0
    w_object_off_ramp: float = 1000.0
    w_object_centered: float = 15.0
    w_spot_off_ramp: float = 1000.0
    position_tolerance: float = 0.005


class SpotYellowChairRamp(SpotBase):
    """Task getting Spot to move a box to a desired goal location."""

    name: str = "spot_yellow_chair_ramp"
    config_t: type[SpotYellowChairRampConfig] = SpotYellowChairRampConfig

    def __init__(self, model_path: str = XML_PATH) -> None:
        """Initialize Spot yellow chair ramp task.

        Args:
            model_path: Path to the XML model file
        """
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["yellow_chair_joint"])
        self.object_vel_idx = get_vel_indices(self.model, ["yellow_chair_joint"])
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")
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

        object_y_axis = sensors[..., self.object_y_axis_idx]
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Compute l2 distance from object pos. to goal.
        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        object_orientation_reward = -self.config.w_orientation * (
            np.abs(np.dot(object_y_axis, Z_AXIS)) > self.config.orientation_threshold
        ).sum(axis=-1)

        # Compute l2 distance from torso pos. to object pos.
        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(
            -1
        )

        # Compute l2 distance from torso pos. to object pos.
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object,
            axis=-1,
        ).mean(-1)

        # Compute squared l2 norm of the object velocity.
        object_linear_velocity_reward = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        object_off_ramp_x = np.logical_or(object_pos[..., 0] <= 1.0, object_pos[..., 0] >= 3.0)
        object_off_ramp_y = object_pos[..., 1] >= 5.25
        object_off_ramp = np.logical_or(object_off_ramp_x, object_off_ramp_y)

        object_off_ramp_penalty = -self.config.w_object_off_ramp * object_off_ramp.mean(-1)

        object_centered = -self.config.w_object_centered * np.abs(object_pos[..., 0] - 2.0).mean(-1)

        spot_off_ramp = body_pos[..., 1] >= 4.5
        spot_off_ramp_penalty = -self.config.w_spot_off_ramp * spot_off_ramp.mean(-1)

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert object_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)
        assert object_off_ramp_penalty.shape == (batch_size,)
        assert spot_off_ramp_penalty.shape == (batch_size,)

        return (
            spot_fallen_reward
            + goal_reward
            + object_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + object_linear_velocity_reward
            + controls_reward
            + object_off_ramp_penalty
            + object_centered
            + spot_off_ramp_penalty
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        # object_pos = np.array([radius * np.cos(theta), radius * np.cos(theta)]) + np.random.randn(2)
        object_pos = DEFAULT_OBJECT_POS + np.random.randn(2) * 0.001
        random_angle = 2 * np.pi * np.random.rand()
        reset_object_pose = np.array([*object_pos, 0.254, np.cos(random_angle / 2), 0, 0, np.sin(random_angle / 2)])

        return np.array(
            [
                *DEFAULT_SPOT_POS + np.random.randn(2) * 0.001,
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
        self, model: MjModel, data: MjData, config: SpotYellowChairRampConfig, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Check if the yellow chair has reached the top platform area of the ramp."""
        # Get object position
        object_pos = data.qpos[self.object_pose_idx[0:3]]

        # Top platform boundaries (calculated from ramp definition in ramp.xml)
        # Ramp body at (3.090, 5.412, 0) + platform at relative (1.215, 0.61, 0.125) with size (1.215, 0.61, 0.125)
        platform_x_min, platform_x_max = 0.66, 3.09  # account for ramp being turned 180 degrees
        platform_y_min, platform_y_max = 4.192, 5.412
        platform_z_min = 0.20  # Slightly below platform surface for tolerance

        # Check if chair is within the top platform area
        x_in_bounds = platform_x_min <= object_pos[0] <= platform_x_max
        y_in_bounds = platform_y_min <= object_pos[1] <= platform_y_max
        z_in_bounds = object_pos[2] >= platform_z_min

        return bool(x_in_bounds and y_in_bounds and z_in_bounds)
