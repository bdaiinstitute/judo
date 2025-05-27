# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Optional

import mujoco
import numpy as np

from judo.tasks.cost_functions import (
    quadratic_norm,
    smooth_l1_norm,
)
from judo.tasks.mujoco_task import MujocoTask
from judo.tasks.task import TaskConfig
from judo.viser_app.path_utils import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/cartpole.xml")


@dataclass
class CartpoleConfig(TaskConfig):
    """Reward configuration for the cartpole task."""

    default_command: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([0.0])
    )
    w_vertical: float = 10.0
    w_centered: float = 10.0
    w_velocity: float = 0.1
    w_control: float = 0.1
    p_vertical: float = 0.01
    p_centered: float = 0.1
    cutoff_time: float = 0.15


class Cartpole(MujocoTask[CartpoleConfig]):
    """Defines the cartpole balancing task."""

    def __init__(self) -> None:
        super().__init__(XML_PATH)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: CartpoleConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Implements the cartpole reward from MJPC.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The cartpole reward has four terms:
            * `vertical_rew`, penalizing the distance between the pole angle and vertical.
            * `centered_rew`, penalizing the distance from the cart to the origin.
            * `velocity_rew` penalizing squared linear and angular velocity.
            * `control_rew` penalizing any actuation.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.

        Returns:
            A list of rewards shaped (batch_size,) where reward at index i represents the reward for that batched traj
        """
        batch_size = states.shape[0]

        vertical_rew = -config.w_vertical * smooth_l1_norm(
            np.cos(states[..., 1]) - 1, config.p_vertical
        ).sum(-1)
        centered_rew = -config.w_centered * smooth_l1_norm(
            states[..., 0], config.p_centered
        ).sum(-1)
        velocity_rew = -config.w_velocity * quadratic_norm(states[..., 2:]).sum(-1)
        control_rew = -config.w_control * quadratic_norm(controls).sum(-1)

        assert vertical_rew.shape == (batch_size,)
        assert centered_rew.shape == (batch_size,)
        assert velocity_rew.shape == (batch_size,)
        assert control_rew.shape == (batch_size,)

        return vertical_rew + centered_rew + velocity_rew + control_rew

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        self.data.qpos = np.array([1.0, np.pi]) + np.random.randn(2)
        self.data.qvel = 1e-1 * np.random.randn(2)
        mujoco.mj_forward(self.model, self.data)

    def is_terminated(self, config: CartpoleConfig) -> bool:
        """Termination condition for cartpole. End if position / velocity are small enough."""
        return np.logical_and(
            np.linalg.norm(self.data.qpos) <= 1e-2,
            np.linalg.norm(self.data.qvel) <= 1e-2,
        ).astype(bool)
