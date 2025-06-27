# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

from judo.config import OverridableConfig
from judo.gui import slider
from judo.optimizers import Optimizer, OptimizerConfig
from judo.tasks.base import Task, TaskConfig
from judo.utils.mujoco import RolloutBackend, make_model_data_pairs
from judo.utils.normalization import (
    IdentityNormalizer,
    Normalizer,
    NormalizerType,
    make_normalizer,
    normalizer_registry,
)
from judo.visualizers.utils import get_trace_sensors


@slider("horizon", 0.1, 10.0)
@slider("control_freq", 0.25, 50.0)
@dataclass
class ControllerConfig(OverridableConfig):
    """Base controller config."""

    horizon: float = 1.0
    spline_order: Literal["zero", "linear", "cubic"] = "linear"
    control_freq: float = 20.0
    max_opt_iters: int = 1
    max_num_traces: int = 5
    action_normalizer: Literal["none", "min_max", "running"] = "none"


class Controller:
    """The controller object."""

    def __init__(
        self,
        controller_config: ControllerConfig,
        task: Task,
        task_config: TaskConfig,
        optimizer: Optimizer,
        optimizer_config: OptimizerConfig,
        rollout_backend: Literal["mujoco"] = "mujoco",
    ) -> None:
        """Initialize the controller.

        Args:
            controller_config: The configuration for the controller.
            task: The Task object that specifies the environment.
            task_config: The configuration for the task.
            optimizer: The optimizer object that will be used for optimization.
            optimizer_config: The configuration for the optimizer.
            rollout_backend: The backend to use for rollouts. Currently only "mujoco" is supported.
        """
        self.controller_cfg = controller_config

        self.task = task
        self.task_cfg = task_config

        self.optimizer = optimizer
        self.optimizer_cfg = optimizer_config

        self.model = task.model
        self.model_data_pairs = make_model_data_pairs(self.model, self.optimizer_cfg.num_rollouts)

        self.rollout_backend = RolloutBackend(num_threads=self.optimizer_cfg.num_rollouts, backend=rollout_backend)

        self.action_normalizer = self._init_action_normalizer()

        # a container for any metadata from the system that we want to pass to the task
        self.system_metadata = {}

        self.states = np.zeros((self.optimizer_cfg.num_rollouts, self.num_timesteps, self.model.nq + self.model.nv))
        self.sensors = np.zeros((self.optimizer_cfg.num_rollouts, self.num_timesteps, self.model.nsensordata))
        self.rollout_controls = np.zeros((self.optimizer_cfg.num_rollouts, self.num_timesteps, self.model.nu))
        self.rewards = np.zeros((self.optimizer_cfg.num_rollouts,))
        self.reset()

        self.traces = None
        self.trace_sensors = get_trace_sensors(self.model)
        self.num_trace_elites = min(self.max_num_traces, len(self.rewards))
        self.num_trace_sensors = len(self.trace_sensors)
        self.sensor_rollout_size = self.num_timesteps - 1
        self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors

    @property
    def horizon(self) -> float:
        """Helper function to recalculate the horizon for simulation."""
        return self.controller_cfg.horizon

    @property
    def max_num_traces(self) -> int:
        """Helper function to recalculate the max number of traces for simulation."""
        return self.controller_cfg.max_num_traces

    @property
    def max_opt_iters(self) -> int:
        """Helper function to recalculate the max number of optimization iterations for simulation."""
        return self.controller_cfg.max_opt_iters

    @property
    def spline_order(self) -> str:
        """Helper function to recalculate the spline order for simulation."""
        return self.controller_cfg.spline_order

    @property
    def action_normalizer_type(self) -> NormalizerType:
        """Helper function to get the type of action normalizer."""
        return self.controller_cfg.action_normalizer

    @property
    def num_timesteps(self) -> int:
        """Helper function to recalculate the number of timesteps for simulation."""
        return np.ceil(self.horizon / self.task.dt).astype(int)

    @property
    def rollout_times(self) -> np.ndarray:
        """Helper function to calculate the rollout times based on the horizon length."""
        return self.task.dt * np.arange(self.num_timesteps)

    @property
    def spline_timesteps(self) -> np.ndarray:
        """Helper function to create new timesteps for spline queries."""
        return np.linspace(0, self.horizon, self.optimizer_cfg.num_nodes, endpoint=True)

    def update_action(self, curr_state: np.ndarray, curr_time: float) -> None:
        """Abstract method for updating controller actions from current state/time."""
        assert curr_state.shape == (self.model.nq + self.model.nv,)
        assert self.optimizer_cfg.num_rollouts > 0, "Need at least one rollout!"

        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == "cubic":
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4

        # Adjust time + move policy forward.
        new_times = curr_time + self.spline_timesteps
        nominal_knots = self.spline(new_times)
        nominal_knots_normalized = self.action_normalizer.normalize(nominal_knots)

        # resizing any variables due to changes in the GUI
        if len(self.model_data_pairs) != self.optimizer_cfg.num_rollouts:
            self.model_data_pairs = make_model_data_pairs(self.model, self.optimizer_cfg.num_rollouts)
            self.rollout_backend.update(self.optimizer_cfg.num_rollouts)

        normalizer_cls = normalizer_registry.get(self.action_normalizer_type)
        if normalizer_cls is None:
            warnings.warn(
                f"Invalid action normalizer type '{self.action_normalizer_type}'. "
                f"Available types: {list(normalizer_registry.keys())}. "
                "Falling back to 'none' normalizer.",
                stacklevel=2,
            )
            normalizer_cls = IdentityNormalizer

        # force the normalizer to be re-initialized when the type changes in GUI
        # TODO(yunhai): check for changes in the normalizer config and update when appropriate
        if not isinstance(self.action_normalizer, normalizer_cls):
            self.action_normalizer = self._init_action_normalizer()

        # call entrypoint prior to optimization
        self.optimizer.pre_optimization(self.times, new_times)

        # run optimization loop
        i = 0
        while i < self.max_opt_iters and not self.optimizer.stop_cond():
            # sample controls and clamp to action bounds
            candidate_knots_normalized = self.optimizer.sample_control_knots(nominal_knots_normalized)
            candidate_knots_normalized = np.clip(
                candidate_knots_normalized,
                self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 0]),
                self.action_normalizer.normalize(self.task.actuator_ctrlrange[:, 1]),
            )
            self.candidate_knots = self.action_normalizer.denormalize(candidate_knots_normalized)

            # Evaluate rollout controls at sim timesteps.
            candidate_splines = make_spline(new_times, self.candidate_knots, self.spline_order)
            self.rollout_controls = candidate_splines(curr_time + self.rollout_times)

            # Roll out dynamics with action sequences.
            self.task.pre_rollout(curr_state, self.task_cfg)
            self.states, self.sensors = self.rollout_backend.rollout(
                self.model_data_pairs,
                curr_state,
                self.rollout_controls,
            )
            self.task.post_rollout(
                self.states,
                self.sensors,
                self.rollout_controls,
                self.task_cfg,
                self.system_metadata,
            )
            self.rewards = self.task.reward(
                self.states,
                self.sensors,
                self.rollout_controls,
                self.task_cfg,
                self.system_metadata,
            )

            # Update nominal knots for next optimization iteration
            nominal_knots_normalized = self.optimizer.update_nominal_knots(candidate_knots_normalized, self.rewards)

            # Update action normalizer
            self.action_normalizer.update(self.candidate_knots)

            i += 1

        # Update nominal controls and spline.
        self.nominal_knots = self.action_normalizer.denormalize(nominal_knots_normalized)
        self.times = new_times
        self.update_spline(self.times, self.nominal_knots)
        self.update_traces()

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.spline(time)

    def update_spline(self, times: np.ndarray, controls: np.ndarray) -> None:
        """Update the spline with new timesteps / controls."""
        self.spline = make_spline(times, controls, self.spline_order)

    def reset(self) -> None:
        """Reset the controls, candidate controls and the spline to their default values."""
        self.task.reset()
        if self.optimizer_cfg.num_nodes < 4 and self.spline_order == "cubic":
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            self.optimizer_cfg.num_nodes = 4
        self.nominal_knots = np.tile(self.task.optimizer_warm_start(), (self.optimizer_cfg.num_nodes, 1))
        self.candidate_knots = np.tile(self.nominal_knots, (self.optimizer_cfg.num_rollouts, 1, 1))
        self.times = self.task.data.time + self.spline_timesteps
        self.update_spline(self.times, self.nominal_knots)

    def update_traces(self) -> None:
        """Update traces by extracting data from sensors readings.

        We need to have num_spline_points - 1 line segments. Sensors will initially be of shape
        (num_rollout x num_timesteps x nsensordata) and needs to end up being in shape
        (num_elite * num_trace_sensors * size of a single rollout x 2 (first and last point of spline) x 3 (3d pos))
        """
        # Resize traces if forced by config change.
        self.sensor_rollout_size = self.num_timesteps - 1
        self.all_traces_rollout_size = self.sensor_rollout_size * self.num_trace_sensors
        if self.num_trace_elites != min(self.max_num_traces, self.optimizer_cfg.num_rollouts):
            self.num_trace_elites = min(self.max_num_traces, self.optimizer_cfg.num_rollouts)
        sensors = np.repeat(self.sensors, 2, axis=1)

        # Order the actions from best to worst so that the first `num_trace_sensors` x `num_nodes` traces
        # correspond to the best rollout and are using a special colors
        elite_actions = np.argsort(self.rewards)[-self.num_trace_elites :][::-1]

        total_traces_rollouts = int(self.num_trace_elites * self.num_trace_sensors * self.sensor_rollout_size)
        # Calculates list of the elite indicies
        trace_inds = [self.model.sensor_adr[id] + pos for id in self.trace_sensors for pos in range(3)]

        # Filter out the non-elite indices we don't care about
        sensors = sensors[elite_actions, :, :]
        # Remove everything but the trace sensors we care about, leaving htis column as size num_trace_sensors * 3
        sensors = sensors[:, :, trace_inds]
        # Remove the first and last part the trajectory to form line segments properly
        # Array will be doubled and look something like: [(0, 0), (1, 1), (4, 4)]
        # We want it to look like: [(0, 1), (1, 4)]
        sensors = sensors[:, 1:-1, :]

        # We doubled it so the number of entries is going to be the size of the rollout * 2
        separated_sensors_size = (self.num_trace_elites, self.sensor_rollout_size, 2, 3)

        # Each block of (i, self.sensor_rollout_size) needs to be interleaved together into a stack of
        # [block(i, ), block (i + 1, ), ..., block(i + n)]
        elites = np.zeros((self.num_trace_sensors * self.num_trace_elites, self.sensor_rollout_size, 2, 3))
        for sensor in range(self.num_trace_sensors):
            s1 = np.reshape(sensors[:, :, sensor * 3 : (sensor + 1) * 3], separated_sensors_size)
            elites[sensor :: self.num_trace_sensors] = s1
        self.traces = np.reshape(elites, (total_traces_rollouts, 2, 3))

    def _init_action_normalizer(self) -> Normalizer:
        """Initialize the action normalizer."""
        action_normalizer_kwargs = {}
        if self.action_normalizer_type == "min_max":
            action_normalizer_kwargs["min"] = self.task.actuator_ctrlrange[:, 0]
            action_normalizer_kwargs["max"] = self.task.actuator_ctrlrange[:, 1]
        elif self.action_normalizer_type == "running":
            action_normalizer_kwargs["init_std"] = 1.0  # TODO(yunhai): make this configurable
        return make_normalizer(self.action_normalizer_type, self.model.nu, **action_normalizer_kwargs)


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
        fill_value=fill_value,  # interp1d is incorrectly typed # type: ignore
        bounds_error=False,
    )
