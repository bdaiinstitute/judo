# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.interpolate import interp1d

from judo.controllers.controller import (
    Controller,
    ControllerConfig,
)
from judo.mujoco_helpers.utils import get_trace_sensors
from judo.tasks.task import Task, TaskConfig
from judo.viser_app.gui import slider

MAX_NUM_TRACES = 5


@slider("num_rollouts", 1, 128, 1)
@slider("horizon", 0.1, 10.0)
@slider("control_freq", 0.25, 50.0)
@slider("num_nodes", 1, 20)
@dataclass
class SamplingBaseConfig(ControllerConfig):
    """Base controller config with spline parameters."""

    horizon: float = 1.0
    num_nodes: int = 3
    num_rollouts: int = 32
    spline_order: Literal["zero", "slinear", "cubic"] = "slinear"
    control_freq: float = 20.0
    use_noise_ramp: bool = False


class SamplingBase(Controller):
    """Base class for all sampling controller implementations."""

    def __init__(
        self,
        task: Task,
        config: SamplingBaseConfig,
        reward_config: TaskConfig,
    ):
        self.task = task
        self.config = config
        self.model = task.model
        self.reward_config = reward_config
        self.reward_function = task.reward

        try:
            self.num_physics_substeps = task.physics_substeps
        except AttributeError:
            self.num_physics_substeps = 1

        self.states = np.zeros(
            (
                self.config.num_rollouts,
                self.num_timesteps * self.num_physics_substeps,
                self.model.nq + self.model.nv,
            )
        )
        self.sensors = np.zeros(
            (
                self.config.num_rollouts,
                self.num_timesteps * self.num_physics_substeps,
                self.model.nsensordata,
            )
        )
        self.rollout_controls = np.zeros(
            (
                self.config.num_rollouts,
                self.num_timesteps * self.num_physics_substeps,
                self.model.nu,
            )
        )
        self.rewards = np.zeros((self.config.num_rollouts,))
        self.reset()

        self.models = self.task.make_models(self.config.num_rollouts)
        self.trace_sensors = get_trace_sensors(self.model)
        self.num_elite = min(MAX_NUM_TRACES, len(self.rewards))
        self.num_trace_sensors = len(self.trace_sensors)
        self.sensor_rollout_size = self.num_timesteps * self.num_physics_substeps - 1
        self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors

    def resize_data(self) -> None:
        """
        Resize states, sensors, and models to (config.num_rollouts, num_timesteps, ...).
        """
        R = self.config.num_rollouts
        T = self.num_timesteps * self.num_physics_substeps

        # remember current rollout count
        old_R = self.states.shape[0]

        # helper to resize a 3+D array on axes 0 and 1
        def _resize(arr: np.ndarray) -> np.ndarray:
            # new shape: (R, T, *rest)
            new_shape = (R, T) + arr.shape[2:]
            new_arr = np.zeros(new_shape, dtype=arr.dtype)

            # how much data to keep along each axis
            keep_R = min(arr.shape[0], R)
            keep_T = min(arr.shape[1], T)

            # copy the overlapping block
            new_arr[:keep_R, :keep_T, ...] = arr[:keep_R, :keep_T, ...]
            return new_arr

        # resize both arrays
        self.states = _resize(self.states)
        self.sensors = _resize(self.sensors)

        # rebuild models only if rollout count changed
        if old_R != R:
            self.models = self.task.make_models(R)

    @property
    def num_timesteps(self) -> int:
        """Helper function to recalculate the number of timesteps for simulation"""
        return np.ceil(self.config.horizon / self.task.dt).astype(int)

    @property
    def rollout_times(self) -> np.ndarray:
        """Helper function to calculate the rollout times based on the horizon length"""
        return self.task.dt * np.arange(self.num_timesteps)

    @property
    def spline_timesteps(self) -> np.ndarray:
        """Helper function to create new timesteps for spline queries."""
        return np.linspace(0, self.config.horizon, self.config.num_nodes, endpoint=True)

    def update_action(
        self, curr_state: np.ndarray, curr_time: float, additional_info: dict[str, Any]
    ) -> None:
        """Abstract method for updating controller actions from current state/time."""
        raise NotImplementedError("Must be implemented in a subclass.")

    def action(self, time: float) -> np.ndarray:
        """Abstract method for querying current action from controller."""
        raise NotImplementedError("Need to implement in subclass.")

    @property
    def spline(self) -> interp1d:
        """Spline defining the current control signal to be applied."""
        # Implemented as a property to help with mypy type checking.
        return self._spline

    @spline.setter
    def spline(self, value: interp1d) -> None:
        self._spline = value

    def update_spline(self, times: np.ndarray, controls: np.ndarray) -> None:
        """Update the spline with new timesteps / controls."""
        self.spline = make_spline(times, controls, self.config.spline_order)

    @property
    def controls(self) -> np.ndarray:
        """Contains the control signals applied in the current rollout."""
        # Implemented as a property to help with mypy type checking.
        return self._controls

    @controls.setter
    def controls(self, value: np.ndarray) -> None:
        self._controls = value

    def set_default_controls(self) -> None:
        """Set default value for the Controller.controls. if there is no default value set to zero."""
        reward_config = self.reward_config

        if not hasattr(reward_config, "default_command"):
            self.controls = np.zeros((self.config.num_nodes, self.task.nu))
        else:
            assert (
                len(reward_config.default_command) == self.task.nu
            ), f"Default command must be {self.task.nu}"
            self.controls = np.tile(
                reward_config.default_command, (self.config.num_nodes, 1)
            )

    def reset(self) -> None:
        """Reset the controls, candidate controls and the spline to their default values."""
        self.set_default_controls()
        self.candidate_controls = np.tile(
            self.controls, (self.config.num_rollouts, 1, 1)
        )
        self.update_spline(self.task.data.time + self.spline_timesteps, self.controls)

    def update_traces(self) -> None:
        """Update traces by extracting data from sensors readings.

        We need to have num_spline_points - 1 line segments. Sensors will initially be of shape
        (num_rollout x num_timesteps * num_physics_substeps x nsensordata) and needs to end up being in shape
        (num_elite * num_trace_sensors * size of a single rollout x 2 (first and last point of spline) x 3 (3d pos))
        """
        # Resize traces if forced by config change.
        self.sensor_rollout_size = self.num_timesteps * self.num_physics_substeps - 1
        self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors
        if self.num_elite != min(MAX_NUM_TRACES, self.config.num_rollouts):
            self.num_elite = min(MAX_NUM_TRACES, self.config.num_rollouts)
        sensors = np.repeat(self.sensors, 2, axis=1)

        # Order the actions from best to worst so that the first `num_trace_sensors` x `num_nodes` traces
        # correspond to the best rollout and are using a special colors
        elite_actions = np.argsort(self.rewards)[-self.num_elite :][::-1]

        total_traces_rollouts = int(
            self.num_elite * self.num_trace_sensors * self.sensor_rollout_size
        )
        # Calculates list of the elite indicies
        trace_inds = [
            self.model.sensor_adr[id] + pos
            for id in self.trace_sensors
            for pos in range(3)
        ]

        # Filter out the non-elite indices we don't care about
        sensors = sensors[elite_actions, :, :]
        # Remove everything but the trace sensors we care about, leaving htis column as size num_trace_sensors * 3
        sensors = sensors[:, :, trace_inds]
        # Remove the first and last part the trajectory to form line segments properly
        # Array will be doubled and look something like: [(0, 0), (1, 1), (4, 4)]
        # We want it to look like: [(0, 1), (1, 4)]
        sensors = sensors[:, 1:-1, :]

        # We doubled it so the number of entries is going to be the size of the rollout * 2
        separated_sensors_size = (self.num_elite, self.sensor_rollout_size, 2, 3)

        # Each block of (i, self.sensor_rollout_size) needs to be interleaved together into a stack of
        # [block(i, ), block (i + 1, ), ..., block(i + n)]
        elites = np.zeros(
            (self.num_trace_sensors * self.num_elite, self.sensor_rollout_size, 2, 3)
        )
        for sensor in range(self.num_trace_sensors):
            s1 = np.reshape(
                sensors[:, :, sensor * 3 : (sensor + 1) * 3], separated_sensors_size
            )
            elites[sensor :: self.num_trace_sensors] = s1
        self.traces = np.reshape(elites, (total_traces_rollouts, 2, 3))


def make_spline(times: np.ndarray, controls: np.ndarray, spline_order: str) -> interp1d:
    """Helper function for creating spline objects.

    Args:
        times: array of times for knot points, shape (T,).
        controls: (possibly batched) array of controls to interpolate, shape (..., T, m).
        spline_order: Order to use for interpolation. Same as parameter for scipy.interpolate.interp1d.
        extrapolate: Whether to allow extrapolation queries. Default true (for re-initialization).
    """
    # fill values for "before" and "after" spline extrapolation.
    fill_value = (controls[..., 0, :], controls[..., -1, :])
    return interp1d(
        times,
        controls,
        kind=spline_order,
        axis=-2,
        copy=False,
        fill_value=fill_value,
        bounds_error=False,
    )
