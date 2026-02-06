# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import unittest

import dexterity.mujoco_extensions.policy_rollout as mpr
import numpy as np
from dexterity.spot_interface.spot.constants import LOCOMOTION_POLICY_PATH, STANDING_HEIGHT

# %%
COMMAND_SIZE = 25
MODEL_PATH = "dexterity/models/xml/scenes/test_scenes/spot_wheel_rim.xml"

START_POS = np.array(
    [0, 0, STANDING_HEIGHT] + [1, 0, 0, 0] + [0.12, 0.72, -1.45] * 4 + [0, -3, 3, 0, 0, 0, 0] + [10, 10, 10, 1, 0, 0, 0]
)
START_VEL = np.arange(6 + 19 + 6)
START_STATE = np.concatenate((START_POS, START_VEL))


class TestPolicyRollout(unittest.TestCase):
    """Testing Policy Rollout"""

    def test_import(self) -> None:
        self.assertIn("System", mpr.__dir__())

    def test_system_creation(self) -> None:
        system = mpr.System(MODEL_PATH, LOCOMOTION_POLICY_PATH)  # type: ignore

        mpr.set_state(system, START_STATE)  # type: ignore
        state_arr = system.get_state()
        self.assertTrue(np.array_equal(state_arr, START_STATE))

        commands = np.zeros((10, COMMAND_SIZE))
        physics_substeps = 2

        system.rollout(START_STATE, commands, physics_substeps)
        for _ in range(10):
            system.policy_inference()

        self.assertIsNotNone(system.get_control())
