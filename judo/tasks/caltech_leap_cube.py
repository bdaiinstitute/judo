# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig

XML_PATH = str(MODEL_PATH / "xml/caltech_leap_cube.xml")
SIM_XML_PATH = str(MODEL_PATH / "xml/caltech_leap_cube_sim.xml")
QPOS_HOME = np.array(
    [
        0.11, 0.005, 0.04, 1.0, 0.0, 0.0, 0.0,  # cube
        0.5, -0.75, 0.75, 0.25,  # index
        0.5, 0.0, 0.75, 0.25,  # middle
        0.5, 0.75, 0.75, 0.25,  # ring
        0.65, 0.9, 0.75, 0.6,  # thumb
    ]
)  # fmt: skip


@slider("w_pos", 0.0, 200.0)
@slider("w_rot", 0.0, 1.0)
@dataclass
class CaltechLeapCubeConfig(LeapCubeConfig):
    """Reward configuration LEAP cube rotation task."""


class CaltechLeapCube(LeapCube):
    """Defines the LEAP cube rotation task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str = SIM_XML_PATH) -> None:
        """Initializes the LEAP cube rotation task."""
        super(LeapCube, self).__init__(model_path, sim_model_path=sim_model_path)
        self.goal_pos = np.array([0.11, 0.005, 0.03])
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.qpos_home = QPOS_HOME
        self.reset_command = np.array(
            [
                0.5, -0.75, 0.75, 0.25,  # index
                0.5, 0.0, 0.75, 0.25,  # middle
                0.5, 0.75, 0.75, 0.25,  # ring
                0.65, 0.9, 0.75, 0.6,  # thumb
            ]
        )  # fmt: skip
        self.reset()
