# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import (
    quadratic_norm,
    smooth_l1_norm,
)

XML_PATH = str(MODEL_PATH / "xml/cartpole.xml")


@dataclass
class CartpoleConfig(TaskConfig):
    """Reward configuration for the cartpole task."""

    w_vertical: float = 10.0
    w_centered: float = 10.0
    w_velocity: float = 0.1
    w_control: float = 0.1
    p_vertical: float = 0.01
    p_centered: float = 0.1


class Cartpole(Task[CartpoleConfig]):
    """Defines the cartpole balancing task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the cartpole task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: CartpoleConfig,
        system_metadata: dict[str, Any] | None = None,
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

        vertical_rew = -config.w_vertical * smooth_l1_norm(np.cos(states[..., 1]) - 1, config.p_vertical).sum(-1)
        centered_rew = -config.w_centered * smooth_l1_norm(states[..., 0], config.p_centered).sum(-1)
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
