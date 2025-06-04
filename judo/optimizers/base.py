# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from judo.config import OverridableConfig
from judo.gui import slider


@slider("num_nodes", 3, 12, 1)
@dataclass
class OptimizerConfig(OverridableConfig):
    """Base class for all optimizer configurations."""

    num_rollouts: int = 16
    num_nodes: int = 4
    use_noise_ramp: bool = False
    noise_ramp: float = 2.5


OptimizerConfigT = TypeVar("OptimizerConfigT", bound=OptimizerConfig)


class Optimizer(ABC, Generic[OptimizerConfigT]):
    """Base class for all optimizers."""

    def __init__(self, config: OptimizerConfigT, nu: int) -> None:
        """Initialize the optimizer."""
        self.config = config
        self.nu = nu

    @property
    def num_rollouts(self) -> int:
        """Get the number of rollouts."""
        return self.config.num_rollouts

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes."""
        return self.config.num_nodes

    @property
    def use_noise_ramp(self) -> bool:
        """Get the use noise ramp flag."""
        return self.config.use_noise_ramp

    @property
    def noise_ramp(self) -> float:
        """Get the noise ramp value."""
        return self.config.noise_ramp

    def pre_optimization(self, old_times: np.ndarray, new_times: np.ndarray) -> None:
        """An entrypoint to the optimizer before optimization.

        This is used to update optimizer parameters with new information.

        Args:
            old_times: The old times for spline interpolation right before sampling. Shape=(num_nodes,).
            new_times: The new times for spline interpolation right before sampling. Shape=(num_nodes,).
        """

    def stop_cond(self) -> bool:
        """Check if the optimization should stop aside from reaching max iters (by default, never).

        Returns:
            True if the optimization should stop, False otherwise.
        """
        return False

    @abstractmethod
    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots given a nominal control input.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """

    @abstractmethod
    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update the nominal control knots based on the sampled controls and rewards.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
