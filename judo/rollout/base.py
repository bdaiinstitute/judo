# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC, abstractmethod

import numpy as np


class AbstractRolloutBackend(ABC):
    """Abstract base class for conducting multithreaded rollouts."""

    def __init__(self, num_threads: int) -> None:
        """Initialize the backend with a number of threads."""
        self.num_threads = num_threads

    @abstractmethod
    def rollout(self, x0: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout depending on the backend.

        Args:
            x0: Initial state for the rollout of shape (num_rollouts, nq + nv).
            controls: Control inputs for the rollout of shape (num_rollouts, num_steps, nu).
        """

    @abstractmethod
    def update(self, num_threads: int) -> None:
        """Update the backend with a new number of threads."""
        pass
