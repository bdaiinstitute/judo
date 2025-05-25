# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class TaskConfig:
    """Base task configuration dataclass."""


ConfigT = TypeVar("ConfigT", bound=TaskConfig)
ModelT = TypeVar("ModelT")


#### This base class should only contain abstract methods ###
#### Specific implementations should be made in a child-class ###
class Task(Generic[ConfigT, ModelT]):
    """Base container for sampling-based MPC tasks. Implements reward, simulation step, and resetting behavior."""

    @abstractmethod
    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: ConfigT,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Abstract reward function for task."""

    @abstractmethod
    def sim_step(self, controls: Optional[interp1d]) -> None:
        """Generic simulation step. Reads controls and updates self.data."""

    @abstractmethod
    def rollout(
        self,
        models: list[ModelT],
        states: np.ndarray,
        controls: np.ndarray,
        additional_info: dict[str, Any],
        output_states: np.ndarray,
        output_sensors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generic threaded rollout. Performs rollouts from a set of states using a set of controls."""

    @abstractmethod
    def make_models(self, num_models: int) -> list[ModelT]:
        """Method that creates a vector of model/system objects that can be used for rollouts."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Control dimension for this task."""

    @property
    @abstractmethod
    def actuator_ctrlrange(self) -> np.ndarray:
        """Actuator limits for this task.

        Returns:
            A numpy array, shape (self.nu, 2), with self.actuator_ctrlrange[i] = (lower_lim, upper_lim) for actuator i.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset behavior for task. Sets config + velocities to zeros."""

    @property
    @abstractmethod
    def additional_task_info(self) -> dict[str, Any]:
        """Get additional state information that might be needed by the controller.

        Use this method to provide task-specific information.
        """

    @property
    @abstractmethod
    def dt(self) -> float:
        """Timestep for this task."""

    @abstractmethod
    def is_terminated(self, config: TaskConfig) -> bool:
        """Defines termination conditions for task."""
