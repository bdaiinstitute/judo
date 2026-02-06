# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""MuJoCo rollout backends for parallel trajectory simulation."""

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from mujoco import MjData, MjModel
from mujoco.rollout import Rollout

from judo.tasks.spot.spot_constants import DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME


class RolloutBackend:
    """Unified backend for conducting multithreaded rollouts.

    Supports two backends:
    - "mujoco": Uses mujoco.rollout for direct physics simulation with controls
    - "mujoco_extensions": Uses C++ mujoco_extensions with ONNX locomotion policy inference

    For Spot tasks, the command format is a 25-dim vector:
    [base_vel(3), arm(7), legs(12), torso(3)]
    """

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        policy_path: str | Path | None = None,
        physics_substeps: int = 2,
    ) -> None:
        """Initialize the rollout backend.

        Args:
            model: MuJoCo model for the scene.
            num_threads: Number of parallel rollout threads.
            policy_path: Path to ONNX locomotion policy. If provided, uses the
                C++ backend with policy inference. Otherwise uses standard MuJoCo.
            physics_substeps: Physics steps per control step (policy backend only).
        """
        self.num_threads = num_threads
        self.model = model
        self.physics_substeps = physics_substeps

        if policy_path is not None:
            self.backend = "mujoco_extensions"
            self._setup_mujoco_extensions(model, policy_path, num_threads)
        else:
            self.backend = "mujoco"
            self._setup_mujoco(model, num_threads)

    def _setup_mujoco(self, model: MjModel, num_threads: int) -> None:
        """Setup the standard MuJoCo rollout backend."""
        self._model_data_pairs = self._make_model_data_pairs(model, num_threads)
        self._rollout_obj = Rollout(nthread=num_threads)

    def _setup_mujoco_extensions(self, model: MjModel, policy_path: str | Path, num_threads: int) -> None:
        """Setup the mujoco_extensions C++ rollout backend with ONNX policy."""
        try:
            from mujoco_extensions.policy_rollout import (  # type: ignore[attr-defined]  # noqa: PLC0415
                create_systems_vector,
                threaded_rollout,
            )
        except ImportError as e:
            raise ImportError(
                "mujoco_extensions is required. Build with: pixi run -e dev cmake --build mujoco_extensions/build"
            ) from e

        self._systems = create_systems_vector(
            model,
            str(policy_path),
            num_threads,
        )
        self._threaded_rollout = threaded_rollout
        self._policy_path = policy_path

    @staticmethod
    def _make_model_data_pairs(model: MjModel, num_pairs: int) -> list[tuple[MjModel, MjData]]:
        """Create model/data pairs for mujoco threaded rollout."""
        models = [deepcopy(model) for _ in range(num_pairs)]
        datas = [MjData(m) for m in models]
        return list(zip(models, datas, strict=True))

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Conduct parallel rollouts.

        Args:
            x0: Initial state, shape (nq+nv,). Will be tiled to num_threads internally.
            controls: Control inputs, shape (num_threads, num_timesteps, nu).
            last_policy_output: Previous policy outputs for mujoco_extensions backend,
                shape (num_threads, 12). Required for mujoco_extensions, ignored for mujoco.

        Returns:
            Tuple of:
                - states: Rolled out states, shape (num_threads, num_timesteps, nq+nv)
                - sensors: Sensor readings, shape (num_threads, num_timesteps, nsensor)
                - policy_outputs: Final policy outputs (mujoco_extensions only), shape (num_threads, 12).
                    None for mujoco backend.
        """
        # Tile x0 if it's a single state
        if x0.ndim == 1:
            x0 = np.tile(x0, (self.num_threads, 1))

        if self.backend == "mujoco":
            states, sensors = self._rollout_mujoco(x0, controls)
            return states, sensors, None
        else:
            if last_policy_output is None:
                raise ValueError("last_policy_output is required for 'mujoco_extensions' backend")
            return self._rollout_mujoco_extensions(x0, controls, last_policy_output)

    def _rollout_mujoco(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rollout using standard MuJoCo backend."""
        ms, ds = zip(*self._model_data_pairs, strict=True)
        ms = list(ms)
        ds = list(ds)

        nq = ms[0].nq
        nv = ms[0].nv
        nu = ms[0].nu

        # Prepend time to batched x0
        full_states = np.concatenate([time.time() * np.ones((len(ms), 1)), x0], axis=-1)

        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[-1] == nu
        assert controls.shape[0] == full_states.shape[0]

        _states, _sensors = self._rollout_obj.rollout(ms, ds, full_states, controls)

        out_states = np.array(_states)[..., 1:]  # Remove time from state
        out_sensors = np.array(_sensors)
        return out_states, out_sensors

    def _rollout_mujoco_extensions(
        self,
        x0: np.ndarray,
        command: np.ndarray,
        last_policy_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rollout using mujoco_extensions C++ backend with policy inference."""
        x0 = np.asarray(x0, dtype=np.float64)
        command = np.asarray(command, dtype=np.float64)
        last_policy_output = np.asarray(last_policy_output, dtype=np.float64)

        states, sensors, policy_outputs = self._threaded_rollout(
            self._systems,
            x0,
            command,
            last_policy_output,
            self.num_threads,
            self.physics_substeps,
            DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME,
        )

        return np.array(states), np.array(sensors), np.array(policy_outputs)

    def update(self, num_threads: int) -> None:
        """Update the number of threads.

        Recreates internal state (model/data pairs or C++ systems) for new thread count.

        Args:
            num_threads: New number of parallel threads.
        """
        self.num_threads = num_threads

        if self.backend == "mujoco":
            self._rollout_obj.close()
            self._setup_mujoco(self.model, num_threads)
        elif self.backend == "mujoco_extensions":
            self._setup_mujoco_extensions(self.model, self._policy_path, num_threads)
