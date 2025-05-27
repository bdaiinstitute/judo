# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Optional

import mujoco
import numpy as np

from judo.tasks.cost_functions import quadratic_norm
from judo.tasks.mujoco_task import MujocoTask
from judo.tasks.task import TaskConfig
from judo.viser_app.path_utils import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/cylinder_push.xml")


@dataclass
class CylinderPushConfig(TaskConfig):
    """Reward configuration for the cylinder push task."""

    default_command: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )
    w_pusher_proximity: float = 0.5
    w_pusher_velocity: float = 0.0
    w_cart_position: float = 0.1
    pusher_goal_offset: float = 0.25
    # We make the position 3 dimensional so that it triggers goal visualization in Viser.
    cart_goal_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0, 0])
    )
    cutoff_time: float = 0.15


class CylinderPush(MujocoTask[CylinderPushConfig]):
    """Defines the cylinder push balancing task."""

    def __init__(self) -> None:
        super().__init__(XML_PATH)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: CylinderPushConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Implements the cylinder push reward from MJPC.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The cylinder push reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and the cart.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance from the cart to the goal.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]

        pusher_pos = states[..., 0:2]
        cart_pos = states[..., 2:4]
        pusher_vel = states[..., 4:6]
        cart_goal = config.cart_goal_position[0:2]

        cart_to_goal = cart_goal - cart_pos
        cart_to_goal_norm = np.linalg.norm(cart_to_goal, axis=-1, keepdims=-1)
        cart_to_goal_direction = cart_to_goal / cart_to_goal_norm

        pusher_goal = cart_pos - config.pusher_goal_offset * cart_to_goal_direction

        pusher_proximity = quadratic_norm(pusher_pos - pusher_goal)
        pusher_reward = -config.w_pusher_proximity * pusher_proximity.sum(-1)

        velocity_reward = -config.w_pusher_velocity * quadratic_norm(pusher_vel).sum(-1)

        goal_proximity = quadratic_norm(cart_pos - cart_goal)
        goal_reward = -config.w_cart_position * goal_proximity.sum(-1)

        assert pusher_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)

        return pusher_reward + velocity_reward + goal_reward

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        theta = 2 * np.pi * np.random.rand(2)
        self.data.qpos = np.array(
            [
                np.cos(theta[0]),
                np.sin(theta[0]),
                2 * np.cos(theta[1]),
                2 * np.sin(theta[1]),
            ]
        )
        self.data.qvel = np.zeros(4)
        mujoco.mj_forward(self.model, self.data)
