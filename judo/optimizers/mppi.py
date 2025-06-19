# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np

from judo.gui import slider
from judo.optimizers.base import Optimizer, OptimizerConfig


@slider("sigma", 0.001, 1.0, 0.01)
@slider("temperature", 0.001, 2.0, 0.05)
@dataclass
class MPPIConfig(OptimizerConfig):
    """Configuration for predictive sampling."""

    sigma: float = 0.1
    temperature: float = 0.05


class MPPI(Optimizer[MPPIConfig]):
    """The MPPI optimizer."""

    def __init__(self, config: MPPIConfig, nu: int) -> None:
        """Initialize the MPPI optimizer."""
        super().__init__(config, nu)

    @property
    def sigma(self) -> float:
        """Get the sigma value."""
        return self.config.sigma

    @property
    def temperature(self) -> float:
        """Get the temperature value."""
        return self.config.temperature

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots given a nominal control input.

        MPPI adds fixed Gaussian noise to the nominal control input.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """
        num_nodes = self.num_nodes
        num_rollouts = self.num_rollouts
        _sigma = self.sigma

        if self.use_noise_ramp:
            ramp = self.noise_ramp * np.linspace(1 / num_nodes, 1, num_nodes, endpoint=True)[:, None]
            sigma = ramp * _sigma
        else:
            sigma = _sigma
        noised_knots = nominal_knots + sigma * np.random.randn(num_rollouts - 1, num_nodes, self.nu)
        return np.concatenate([nominal_knots[None], noised_knots])

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update the nominal control knots based on the sampled controls and rewards.

        MPPI uses a weighted average of the sampled controls based on the rewards.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
        # See algorithm 2 for the abridged details.
        # We can imagine sigma = 0, phi(x_t) = 0 for our MPPI implementation
        # https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf
        costs = -rewards
        beta = np.min(costs)

        _weights = np.exp(-(costs - beta) / self.temperature)
        weights = _weights / np.sum(_weights)
        nominal_knots = np.sum(weights[:, None, None] * sampled_knots, axis=0)
        return nominal_knots
