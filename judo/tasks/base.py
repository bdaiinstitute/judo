# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec

from judo.utils.mujoco import RolloutBackend, SimBackend


@dataclass
class TaskConfig:
    """Base task configuration dataclass."""


ConfigT = TypeVar("ConfigT", bound=TaskConfig)


class Task(ABC, Generic[ConfigT]):
    """Task definition."""

    default_backend = "mujoco"  # Default backend for all tasks
    config_t: type[ConfigT]

    def __init__(self, model_path: Path | str = "", sim_model_path: Path | str | None = None) -> None:
        """Initialize the Mujoco task."""
        if not model_path:
            raise ValueError("Model path must be provided.")
        self.config = self.config_t()
        self.spec = MjSpec.from_file(str(model_path))
        self.model = self.spec.compile()
        self.data = MjData(self.model)
        self.model_path = model_path
        self.sim_model = self.model if sim_model_path is None else MjModel.from_xml_path(str(sim_model_path))

        self.RolloutBackend = RolloutBackend
        self.SimBackend = SimBackend

    @property
    def time(self) -> float:
        """Returns the current simulation time."""
        return self.data.time

    @time.setter
    def time(self, value: float) -> None:
        """Sets the current simulation time."""
        self.data.time = value

    @abstractmethod
    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Abstract reward function for task.

        Args:
            states: The rolled out states (after the initial condition). Shape=(num_rollouts, T, nq + nv).
            sensors: The rolled out sensors readings. Shape=(num_rollouts, T, total_num_sensor_dims).
            controls: The rolled out controls. Shape=(num_rollouts, T, nu).
            config: The current task config (passed in from the top-level controller).
            system_metadata: Any additional metadata from the system that is useful for computing the reward. For
                example, in the cube rotation task, the system could pass in new goal cube orientations to the
                controller here.

        Returns:
            rewards: The reward for each rollout. Shape=(num_rollouts,).
        """

    @property
    def nu(self) -> int:
        """Number of control inputs. The same as the MjModel for this task."""
        return self.model.nu

    @property
    def ctrlrange(self) -> np.ndarray:
        """Mujoco actuator limits for this task. Same as actuator limits for this task."""
        limits = self.model.actuator_ctrlrange
        limited: np.ndarray = self.model.actuator_ctrllimited.astype(bool)  # type: ignore
        limits[~limited] = np.array([-np.inf, np.inf], dtype=limits.dtype)  # if not limited, set to inf
        return limits  # type: ignore

    def reset(self) -> None:
        """Reset behavior for task. Sets config + velocities to zeros."""
        self.data.qpos = np.zeros_like(self.data.qpos)
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        """Returns Mujoco physics timestep for default physics task."""
        return self.model.opt.timestep

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Maps the controls from the optimizer to the controls used in the simulation.

        This can be overridden by tasks that have different control mappings. By default, it is the identity
        function.

        Args:
            controls: The controls from the optimizer. Shape=(num_rollouts, T, nu).

        Returns:
            mapped_controls: The controls to be used in the simulation. Shape=(num_rollouts, T, nu).
        """
        return controls

    def pre_rollout(self, curr_state: np.ndarray) -> None:
        """Pre-rollout behavior for task (does nothing by default).

        Args:
            curr_state: Current state of the task. Shape=(nq + nv,).
        """

    def post_rollout(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post-rollout behavior for task (does nothing by default).

        Same inputs as in reward function.
        """

    def pre_sim_step(self) -> None:
        """Pre-simulation step behavior for task (does nothing by default)."""

    def post_sim_step(self) -> None:
        """Post-simulation step behavior for task (does nothing by default)."""

    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns metadata from the simulation.

        We need this function because the simulation thread runs separately from the controller thread, but there are
        task objects in both. This function is used to pass information about the simulation's version of the task to
        the controller's version of the task.

        For example, the LeapCube task has a goal quaternion that is updated in the simulation thread based on whether
        the goal was reached (which the controller thread doesn't know about). When a new goal is set, it must be passed
        to the controller thread via this function.
        """
        return {}

    def optimizer_warm_start(self) -> np.ndarray:
        """Returns a warm start for the optimizer.

        This is used to provide an initial guess for the optimizer when optimizing the task before any iterations.
        """
        return np.zeros(self.nu)

    def get_sensor_start_index(self, sensor_name: str) -> int:
        """Returns the starting index of a sensor in the 'sensors' array given the sensor's name.

        Args:
            sensor_name: The name of the sensor to get the index of.
        """
        return self.model.sensor(sensor_name).adr[0]

    def get_joint_position_start_index(self, joint_name: str) -> int:
        """Returns the starting index of a joint's position in the 'states' array given the joint's name.

        Args:
            joint_name: The name of the joint to get the starting index in the position of the state array.
        """
        return self.model.jnt_qposadr[self.model.joint(joint_name).id]

    def get_joint_velocity_start_index(self, joint_name: str) -> int:
        """Returns the starting index of a joint's velocity in the 'states' array given the joint's name.

        NOTE: This is the index of the joint's velocity in the state array, which is after the position indices!

        Args:
            joint_name: The name of the joint to get the starting index in the state array of.
        """
        return self.model.nq + self.model.jnt_dofadr[self.model.joint(joint_name).id]

    def success(self, model: MjModel, data: MjData, config: ConfigT, metadata: dict[str, Any] | None = None) -> bool:
        """If applicable, returns whether the task was successful at some given state (always False by default).

        This is used by the performance benchmarker to determine whether to terminate an episode. Note that this should
        be run on the simulated model and data, which should run in the highest fidelity, not the controller's copy.

        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            config: The current task config (passed in from the top-level controller).
            metadata: Any additional information that might be helpful for determining success.

        Returns:
            success: Whether the task was successful.
        """
        raise NotImplementedError("The success criteria needs to be implemented by the child task.")

    def failure(self, model: MjModel, data: MjData, config: ConfigT, metadata: dict[str, Any] | None = None) -> bool:
        """If applicable, returns whether the task has failed at some given state (always False by default).

        This is used by the performance benchmarker to determine whether to terminate an episode. Note that this should
        be run on the simulated model and data, which should run in the highest fidelity, not the controller's copy.

        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            config: The current task config (passed in from the top-level controller).
            metadata: Any additional information that might be helpful for determining failure.

        Returns:
            failure: Whether the task has failed.
        """
        raise NotImplementedError("The episode failure criteria needs to be implemented by the child task.")

    def compute_metrics(
        self,
        model: MjModel,
        data: MjData,
        config: ConfigT,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """If applicable, computes any non-reward-based metrics for the task (returns empty dict by default).

        This is used by the performance benchmarker to report additional metrics at the end of an episode. This function
        assumes that any metric values are computed per-step. Note that this should be run on the simulated model and
        data, which should run in the highest fidelity, not the controller's copy.

        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            config: The current task config (passed in from the top-level controller).
            metadata: Any additional information that might be helpful for computing metrics.

        Returns:
            metrics: A dictionary that can hold anything, but should be a single element so it's stackable in a list
                easily (e.g., a float, string, etc.)
        """
        return {}

    def reduce_metrics(self, metric_dict: dict[str, list[Any]]) -> dict[str, Any]:
        """If applicable, reduces a list of metrics to a single metric value (e.g., by averaging).

        This is used by the performance benchmarker to report a single metric value at the end of an episode. This
        function assumes that any metric values are computed per-step. By default, we take the mean of each metric if
        possible, otherwise we take the last value.

        Args:
            metric_dict: A dictionary of lists of metrics, where each list contains the metric values for each step.

        Returns:
            reduced_metrics: A dictionary of reduced metrics.
        """
        reduced_metrics: dict[str, Any] = {}
        for key, values in metric_dict.items():
            try:
                reduced_metrics[key] = float(np.mean(values))
            except (TypeError, ValueError):
                reduced_metrics[key] = values[-1]
        return reduced_metrics
