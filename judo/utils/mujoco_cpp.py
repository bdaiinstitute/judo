# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from mujoco import MjData, MjModel

if TYPE_CHECKING:
    # Type stubs for type checking - these don't actually import the module
    def rollout(model: Any, data: Any, x0: Any, controls: Any) -> Any: ...  # noqa: D103

    def sim(model: MjModel, data: MjData, x0: Any, controls: Any) -> None: ...  # noqa: D103
else:
    try:
        from judo_cpp import rollout, sim
    except ImportError:
        # Create dummy functions for when C++ extensions aren't available
        def rollout(*args, **kwargs):  # type: ignore[no-untyped-def, misc]  # noqa: ANN002, ANN003, ANN201, D103
            raise ImportError("judo_cpp module is not available. Please build the C++ extensions.")

        def sim(*args, **kwargs):  # type: ignore[no-untyped-def, misc]  # noqa: ANN002, ANN003, ANN201, D103
            raise ImportError("judo_cpp module is not available. Please build the C++ extensions.")


class RolloutBackend:
    """The backend for conducting multithreaded rollouts."""

    def __init__(
        self, num_threads: int, backend: Literal["mujoco", "mujoco_spot", "mujoco_cpp"], task_to_sim_ctrl: Callable
    ) -> None:
        """Initialize the backend with a number of threads."""
        self.backend = backend
        if self.backend == "mujoco":
            self.setup_mujoco_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        self.task_to_sim_ctrl = task_to_sim_ctrl

    def setup_mujoco_backend(self) -> None:
        """Setup the mujoco backend."""
        if self.backend == "mujoco":
            self.rollout_func = rollout
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def rollout(
        self,
        model_data_pairs: list[tuple[MjModel, MjData]],
        x0: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout depending on the backend."""
        # unpack models into a list of models and data
        ms, ds = zip(*model_data_pairs, strict=True)
        ms = list(ms)
        ds = list(ds)

        # getting shapes
        nq = ms[0].nq
        nv = ms[0].nv
        nu = ms[0].nu

        # the state passed into mujoco's rollout function includes the time
        # shape = (num_rollouts, num_states + 1)
        x0_batched = np.tile(x0, (len(ms), 1))
        processed_controls = self.task_to_sim_ctrl(controls)
        assert x0_batched.shape[-1] == nq + nv
        assert x0_batched.ndim == 2
        assert processed_controls.ndim == 3
        assert processed_controls.shape[-1] == nu
        assert processed_controls.shape[0] == x0_batched.shape[0]

        # rollout
        if self.backend == "mujoco":
            _states, _out_sensors = self.rollout_func(ms, ds, x0_batched, processed_controls)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        out_states = np.array(_states)
        out_sensors = np.array(_out_sensors)
        return out_states, out_sensors

    def update(self, num_threads: int) -> None:
        """Update the backend with a new number of threads."""
        if self.backend == "mujoco":
            self.setup_mujoco_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


class SimBackend:
    """The backend for conducting simulation."""

    def __init__(self, task_to_sim_ctrl: Callable) -> None:
        """Initialize the backend."""
        self.task_to_sim_ctrl = task_to_sim_ctrl

    def sim(self, sim_model: MjModel, sim_data: MjData, sim_controls: np.ndarray) -> None:
        """Conduct a simulation step using cpp sim."""
        processed_ctrl = self.task_to_sim_ctrl(sim_controls)
        x0 = np.concatenate([sim_data.qpos, sim_data.qvel])
        sim(sim_model, sim_data, x0, processed_ctrl)
