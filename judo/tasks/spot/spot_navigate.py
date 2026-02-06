# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""SpotNavigate task - navigate Spot to a goal location.

Adapted from starfish/dexterity/tasks/spot_navigate.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT
from judo.utils.fields import np_1d_field


@dataclass
class SpotNavigateConfig(SpotBaseConfig):
    """Configuration for the SpotNavigate task."""

    w_goal: float = 60.0
    fall_penalty: float = 2500.0
    w_controls: float = 0.0
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, STANDING_HEIGHT]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotNavigate(SpotBase[SpotNavigateConfig]):
    """Task getting Spot to navigate to a desired goal location."""

    name: str = "spot_navigate"
    config_t: type[SpotNavigateConfig] = SpotNavigateConfig  # type: ignore[assignment]
    config: SpotNavigateConfig

    def __init__(
        self,
        config: SpotNavigateConfig | None = None,
    ) -> None:
        """Initialize the SpotNavigate task."""
        super().__init__(use_arm=False, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the navigate task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(
            body_pos - self.config.goal_position[None, None], axis=-1
        ).mean(-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return spot_fallen_reward + goal_reward + controls_reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the navigate task."""
        return np.array([0, 0, STANDING_HEIGHT, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos])
