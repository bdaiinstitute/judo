# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from judo.optimizers.base import Optimizer, OptimizerConfig


@dataclass
class CrossEntropyMethodConfig(OptimizerConfig):
    """Configuration for cross-entropy method."""

    sigma_min: float = 0.1
    sigma_max: float = 1.0
    num_elites: int = 2


class CrossEntropyMethod(Optimizer[CrossEntropyMethodConfig]):
    """The cross-entropy method."""

    def __init__(self, config: CrossEntropyMethodConfig, nu: int) -> None:
        """Initialize cross-entropy method optimizer."""
        super().__init__(config, nu)
        num_nodes = config.num_nodes
        self.sigma = ((self.sigma_min + self.sigma_max) / 2) * np.ones((num_nodes, nu))

    @property
    def sigma_min(self) -> float:
        """Get the minimum sigma value."""
        return self.config.sigma_min

    @property
    def sigma_max(self) -> float:
        """Get the maximum sigma value."""
        return self.config.sigma_max

    @property
    def num_elites(self) -> int:
        """Get the number of elites."""
        return self.config.num_elites

    def pre_optimization(self, old_times: np.ndarray, new_times: np.ndarray) -> None:
        """Update sigma if the number of nodes has changed."""
        if len(self.sigma) != self.num_nodes:
            self.sigma = interp1d(
                old_times,
                self.sigma,
                axis=0,
                fill_value="extrapolate",  # interp1d has wrong typing # type: ignore
                kind="linear",
            )(new_times)

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots given a nominal control input.

        CEM adds fitted Gaussian noise to the nominal control input.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """
        num_nodes = self.num_nodes
        num_rollouts = self.num_rollouts
        noise_ramp = self.noise_ramp

        if self.use_noise_ramp:
            ramp = np.linspace(noise_ramp / num_nodes, noise_ramp, num_nodes, endpoint=True)[:, None]
            self.sigma = np.clip(self.sigma * ramp, self.sigma_min, self.sigma_max)
        noised_knots = nominal_knots + self.sigma[None] * np.random.randn(num_rollouts - 1, num_nodes, self.nu)
        return np.concatenate([nominal_knots[None], noised_knots])

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update the nominal control knots based on the sampled controls and rewards.

        CEM takes the top k sampled control inputs and fits a Gaussian to them.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
        elite_inds = np.flip(np.argsort(rewards))[: self.num_elites]
        elite_knots = sampled_knots[elite_inds]
        nominal_knots = elite_knots.mean(0)
        self.sigma = np.clip(np.sqrt(elite_knots.var(0)), self.sigma_min, self.sigma_max)
        return nominal_knots
