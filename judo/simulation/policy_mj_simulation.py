# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo Simulation with locomotion policy support."""

from pathlib import Path

import numpy as np
from mujoco import mj_forward
from omegaconf import DictConfig

from judo.simulation.mj_simulation import MJSimulation
from judo.tasks.spot.spot_constants import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME, POLICY_OUTPUT_DIM
from mujoco_extensions.policy_rollout import create_systems_vector, threaded_rollout  # type: ignore


class PolicyMJSimulation(MJSimulation):
    """MuJoCo simulation with locomotion policy support.

    For tasks with locomotion_policy_path set, uses C++ mujoco_extensions
    threaded_rollout to run the neural network policy at 50Hz.

    The simulation maintains internal state for the locomotion policy
    (last_policy_output) to ensure smooth transitions between timesteps.
    """

    def __init__(
        self,
        init_task: str = "spot_base",
        task_registration_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the policy simulation.

        Args:
            init_task: Name of the task to initialize.
            task_registration_cfg: Optional task registration configuration.
        """
        super().__init__(init_task=init_task, task_registration_cfg=task_registration_cfg)

        self._systems = None
        self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)

        # Initialize C++ systems if task uses locomotion policy
        if self.task.locomotion_policy_path is not None:
            self._init_cpp_systems(self.task.locomotion_policy_path)

    def _init_cpp_systems(self, policy_path: str | Path) -> None:
        """Initialize the C++ systems vector for threaded rollout.

        Args:
            policy_path: Path to the ONNX locomotion policy file.
        """
        self._systems = create_systems_vector(
            self.task.model,  # Pass the MjModel directly
            str(policy_path),
            1,  # Single system for simulation
        )

    def step(self, command: np.ndarray) -> None:
        """Step the simulation forward.

        Routes to the C++ policy rollout if systems are initialized,
        otherwise falls back to direct actuator control.

        Args:
            command: Control array in task format (task.nu dimensions).
                For locomotion tasks, will be converted to policy command internally.
        """
        if self._systems is not None:
            if self.paused:
                return
            command = self.task.task_to_sim_ctrl(command)
            self._step_with_locomotion_policy(command)
        else:
            super().step(command)

    def _step_with_locomotion_policy(self, command: np.ndarray) -> None:
        """Execute a single step using the C++ rollout backend.

        Args:
            command: Command array for the locomotion policy.
        """
        # Get current state
        state = np.concatenate([self.task.data.qpos, self.task.data.qvel])

        # Ensure command is 1D
        command = np.asarray(command, dtype=np.float64).flatten()

        # Reshape for threaded rollout:
        # states: (num_threads, nq+nv)
        # commands: (num_threads, num_timesteps, cmd_dim)
        # last_outputs: (num_threads, POLICY_OUTPUT_DIM)
        states = np.array([state], dtype=np.float64)
        commands = np.array([[command]], dtype=np.float64)
        last_outputs = np.array([self._last_policy_output], dtype=np.float64)

        # Run rollout
        self.task.pre_sim_step()
        out_states, out_sensors, policy_outputs = threaded_rollout(
            self._systems,
            states,
            commands,
            last_outputs,
            1,  # num_threads
            self.task.physics_substeps,
            DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
        )
        self.task.post_sim_step()

        # Update simulation state from rollout result
        final_state = np.array(out_states[0][-1])
        nq = self.task.model.nq
        self.task.data.qpos[:] = final_state[:nq]
        self.task.data.qvel[:] = final_state[nq:]
        self.task.data.time += self.task.dt

        # Compute derived quantities (xpos, xquat, etc.) for visualization
        mj_forward(self.task.model, self.task.data)

        # Update last policy output for continuity
        self._last_policy_output = np.array(policy_outputs[0])

    def reset_policy_state(self) -> None:
        """Reset the internal policy state to zeros."""
        self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)

    def set_task(self, task_name: str) -> None:
        """Set the current task and reinitialize C++ systems if needed.

        Args:
            task_name: Name of the task to set.
        """
        super().set_task(task_name)

        # Reinitialize systems based on new task's policy
        if self.task.locomotion_policy_path is not None:
            self._init_cpp_systems(self.task.locomotion_policy_path)
            self._last_policy_output = np.zeros(POLICY_OUTPUT_DIM)
        else:
            self._systems = None

    @property
    def last_policy_output(self) -> np.ndarray:
        """Returns the last policy output (12-dim leg actions)."""
        return self._last_policy_output.copy()
