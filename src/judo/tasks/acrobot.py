# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Optional

import mujoco
import numpy as np

from judo.tasks.cost_functions import quadratic_norm
from judo.tasks.mujoco_task import MujocoTask
from judo.tasks.task import TaskConfig
from judo.viser_app.path_utils import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/acrobot.xml")


@dataclass
class AcrobotConfig(TaskConfig):
    """Reward configuration for the acrobot task."""

    default_command: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([0.0])
    )
    w_vertical: float = 10.0
    w_velocity: float = 0.1
    w_control: float = 0.1
    p_vertical: float = 0.01
    cutoff_time: float = 0.15


class Acrobot(MujocoTask[AcrobotConfig]):
    """Defines the acrobot balancing task."""

    def __init__(self) -> None:
        super().__init__(XML_PATH)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: AcrobotConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Implements the acrobot reward from MJPC.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The acrobot reward has four terms:
            * `vertical_rew`, penalizing the distance between the pole angle and vertical.
            * `velocity_rew` penalizing squared linear and angular velocity.
            * `control_rew` penalizing any actuation.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]

        vertical_rew = -config.w_vertical * quadratic_norm(
            np.cos(states[:, :, 0]) + np.cos(states[:, :, 0] + states[:, :, 1]) - 2
        )
        # print("vertical rewards", np.around(vertical_rew, decimals=2))
        velocity_rew = -config.w_velocity * quadratic_norm(states[..., 2:]).sum(-1)
        control_rew = -config.w_control * quadratic_norm(controls).sum(-1)

        assert vertical_rew.shape == (batch_size,)
        assert velocity_rew.shape == (batch_size,)
        assert control_rew.shape == (batch_size,)

        return vertical_rew + velocity_rew + control_rew

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        self.data.qpos = np.array([np.pi, 0.5])
        self.data.qvel = 1e-111 * np.random.randn(2)
        mujoco.mj_forward(self.model, self.data)
