# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.


from dataclasses import dataclass
from typing import Any

import numpy as np

from judo.controllers.sampling_base import (
    SamplingBase,
    SamplingBaseConfig,
    make_spline,
)
from judo.tasks.task import Task, TaskConfig
from judo.viser_app.gui import slider


@slider("temperature", 0.1, 2.0, 0.05)
@dataclass
class MPPIConfig(SamplingBaseConfig):
    """Configuration for predictive sampling."""

    # Parameter controlling the noise level in the action sampling space
    sigma: float = 0.05
    # Create a variable for the temperature hyperparameter. Increasing it biases to averaging the solutions
    # Note: this is a known bad quantity because I just chose a lambda that would give you enough fidelity to change it
    # to a low or a high number. Typically lower values are required to bias more complex systems. Too low of a value
    # can cause instability
    temperature: float = 1.0


class MPPI(SamplingBase):
    """Predictive sampling planner.

    Args:
        config: configuration object with hyperparameters for planner.
        model: mujoco model of system being controlled.
        data: current configuration data for mujoco model.
        reward_func: function mapping batches of states/controls to batches of rewards.
    """

    def __init__(
        self,
        task: Task,
        config: MPPIConfig,
        reward_config: TaskConfig,
    ):
        super().__init__(task, config, reward_config)

    def update_action(
        self, curr_state: np.ndarray, curr_time: float, additional_info: dict[str, Any]
    ) -> None:
        """Performs rollouts + reward computation from current state."""
        assert curr_state.shape == (self.model.nq + self.model.nv,)
        assert self.config.num_rollouts > 0, "Need at least one rollout!"

        # Check if num_rollouts has changed and resize arrays accordingly.
        if self.states.shape[:2] != (self.config.num_rollouts, self.num_timesteps):
            self.resize_data()

        # Adjust time + move policy forward.
        new_times = curr_time + self.spline_timesteps
        base_controls = self.spline(new_times)[None]

        # Sample action noise (leaving one sequence noiseless).
        noised_controls = base_controls + self.config.sigma * np.random.randn(
            self.config.num_rollouts - 1, self.config.num_nodes, self.task.nu
        )
        candidate_controls = np.concatenate([base_controls, noised_controls])

        # Clamp controls to action bounds.
        candidate_controls = np.clip(
            candidate_controls,
            self.task.actuator_ctrlrange[:, 0],
            self.task.actuator_ctrlrange[:, 1],
        )

        # Evaluate rollout controls at sim timesteps.
        candidate_splines = make_spline(
            new_times, candidate_controls, self.config.spline_order
        )
        self.rollout_controls = candidate_splines(curr_time + self.rollout_times)

        # Create lists of states / controls for rollout.
        curr_state_batch = np.tile(curr_state, (self.config.num_rollouts, 1))

        # Roll out dynamics with action sequences and set the cutoff time for each controller here
        self.task.cutoff_time = self.reward_config.cutoff_time

        # Roll out dynamics with action sequences.
        self.task.rollout(
            self.models,
            curr_state_batch,
            self.rollout_controls,
            additional_info,
            self.states,
            self.sensors,
        )

        # Evalate rewards. We have the negative of the version in the code because our rewards are negative
        self.rewards = self.reward_function(
            self.states,
            self.sensors,
            self.rollout_controls,
            self.reward_config,
            additional_info,
        )
        costs = -self.rewards

        # See algorithm 2 for the abridged details.
        # We can imagine sigma = 0, phi(x_t) = 0 for our MPPI implementation
        # https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf
        beta = np.min(costs)

        weighted_rewards = np.exp(-(costs - beta) / self.config.temperature)
        # Basically softmax
        weights = weighted_rewards / np.sum(weighted_rewards)
        self.controls = np.sum(
            weights[:, np.newaxis, np.newaxis] * candidate_controls, axis=0
        )

        self.update_spline(new_times, self.controls)

        # Update traces.
        self.update_traces()

    def action(self, time: float) -> np.ndarray:
        """Current best action of policy."""
        return self.spline(time)
