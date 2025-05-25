# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
from copy import deepcopy
from typing import Any, Optional, Tuple

import mujoco
import numpy as np
from mujoco import MjData, MjModel
from scipy.interpolate import interp1d

from judo.tasks.task import ConfigT, Task, TaskConfig
from mujoco_extensions.policy_rollout import threaded_physics_rollout_in_place


class MujocoTask(Task[ConfigT, Tuple[MjModel, MjData]]):
    """Container for task based on Mujoco"""

    def __init__(self, model_path: str = ""):
        if not model_path:
            raise NotImplementedError("Model path must be specified in child class.")
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.cutoff_time = None  # Placeholder
        self._additional_info: dict[str, Any] = {}

    @property
    def additional_task_info(self) -> dict[str, Any]:
        """Get additional state information that might be needed by the controller.

        Override this method in child classes to provide task-specific information.
        """
        return self._additional_info

    def sim_step(self, controls: Optional[interp1d]) -> None:
        """Generic mujoco simulation step."""
        # Read current action from spline.

        if controls is None:
            current_action = self.default_idle_command
        else:
            current_action = controls(self.data.time)
        assert (
            current_action.shape == (self.model.nu,)
        ), f"For default sim step, control shape (got {current_action.shape}) must == model.nu (got {self.model.nu})"

        # Write current action into MjData.
        self.data.ctrl[:] = current_action

        # Step mujoco physics.
        mujoco.mj_step(self.model, self.data)

    def rollout(
        self,
        models: list[Tuple[MjModel, MjData]],
        states: np.ndarray,
        controls: np.ndarray,
        additional_info: dict[str, Any],
        output_states: np.ndarray,
        output_sensors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generic mujoco threaded rollout."""
        assert (
            len(models) == states.shape[0] == controls.shape[0]
        ), "Models, state_batch, and control_batch must be same size in batch dim."

        model_batch, data_batch = list(
            zip(*models, strict=True)
        )  # Unzip (model, data) tuples into batches of models/data.

        threaded_physics_rollout_in_place(
            list(model_batch),
            list(data_batch),
            states,
            controls,
            output_states,
            output_sensors,
        )

    def make_models(self, num_models: int) -> list[Tuple[MjModel, MjData]]:
        """Makes a list of (MjModel, MjData) tuples for use with rollouts."""
        models = [deepcopy(self.model) for _ in range(num_models)]
        data = [MjData(m) for m in models]

        return list(zip(models, data, strict=True))

    @property
    def nu(self) -> int:
        """Number of control inputs. The same as the MjModel for this task."""
        return self.model.nu

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Mujoco actuator limits for this task."""
        return self.model.actuator_ctrlrange

    def reset(self) -> None:
        """Reset behavior for task. Sets config + velocities to zeros."""
        self.data.qpos = np.zeros_like(self.data.qpos)
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        """Returns Mujoco physics timestep for default physics task."""
        return self.model.opt.timestep

    @property
    def default_idle_command(self) -> np.ndarray:
        """Default idling command for the task."""
        return np.zeros((self.nu,))

    def is_terminated(self, config: TaskConfig) -> bool:
        """Defines if the current state is terminal."""
        return False
