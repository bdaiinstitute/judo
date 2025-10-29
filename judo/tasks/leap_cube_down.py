# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np

from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.utils.assets import retrieve_model_from_remote

QPOS_HOME = np.array(
    [
        -0.04, -0.035, -0.065, 1.0, 0.0, 0.0, 0.0,  # cube
        1.0, 0.0, 0.8, 0.8,  # index
        1.0, 0.0, 0.8, 0.8,  # middle
        1.0, 0.0, 0.8, 0.8,  # ring
        1.0, 1.0, 0.4, 0.9,  # thumb
    ]
)  # fmt: skip


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class LeapCubeDownConfig(LeapCubeConfig):
    """Reward configuration LEAP cube rotation task."""

    w_rot: float = 0.05


class LeapCubeDown(LeapCube):
    """Defines the LEAP cube with palm down rotation task."""

    name: str = "leap_cube_down"
    config_t: type[LeapCubeDownConfig] = LeapCubeDownConfig

    def __init__(self, model_path: str | None = None, sim_model_path: str | None = None) -> None:
        """Initializes the LEAP cube rotation task."""
        default_model_path, default_sim_model_path = retrieve_model_from_remote("leap_cube_palm_down", force=False)
        if model_path is None:
            model_path = default_model_path
        if sim_model_path is None:
            sim_model_path = default_sim_model_path
        super(LeapCube, self).__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.goal_pos = np.array([-0.04, -0.035, -0.065])
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.qpos_home = QPOS_HOME
        self.reset_command = np.array(
            [
                1.0, 0.0, 0.8, 0.8,  # index
                1.0, 0.0, 0.8, 0.8,  # middle
                1.0, 0.0, 0.8, 0.8,  # ring
                1.0, 1.0, 0.4, 0.9,  # thumb
            ]
        )  # fmt: skip
        self.reset()
