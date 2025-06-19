# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np

from judo.optimizers.base import Optimizer, OptimizerConfig


@dataclass
class PredictiveSamplingConfig(OptimizerConfig):
    """Configuration for predictive sampling."""

    sigma: float = 0.05


class PredictiveSampling(Optimizer[PredictiveSamplingConfig]):
    """Predictive sampling planner."""

    def __init__(self, config: PredictiveSamplingConfig, nu: int) -> None:
        """Initialize predictive sampling optimizer."""
        super().__init__(config, nu)

    @property
    def sigma(self) -> float:
        """Get the sigma value."""
        return self.config.sigma

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots given a nominal control input.

        Predictive sampling adds fixed Gaussian noise to the nominal control input.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """
        num_nodes = self.num_nodes
        num_rollouts = self.num_rollouts
        _sigma = self.sigma

        if self.use_noise_ramp:
            ramp = self.noise_ramp * np.linspace(1 / num_nodes, 1, num_nodes)[:, None]
            sigma = ramp * _sigma
        else:
            sigma = _sigma
        noised_knots = nominal_knots + sigma * np.random.randn(num_rollouts - 1, num_nodes, self.nu)
        return np.concatenate([nominal_knots[None], noised_knots])

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update the nominal control knots based on the sampled controls and rewards.

        Predictive sampling takes the best sampled control input.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
        i_best = rewards.argmax()
        return sampled_knots[i_best]
