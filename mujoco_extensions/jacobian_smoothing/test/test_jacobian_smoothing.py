# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import unittest

import dexterity.mujoco_extensions.jacobian_smoothing as mjs
import mujoco
import numpy as np


class TestJacobianSmoothing(unittest.TestCase):
    """Testing Jacobian related functions"""

    def test_import(self) -> None:
        self.assertIn("get_contact_jacobian", mjs.__dir__())

    def test_contact_jacobian(self) -> None:
        model_filename = "dexterity/models/xml/scenes/legacy/planar_hand.xml"
        model = mujoco.MjModel.from_xml_path(model_filename)
        data = mujoco.MjData(model)

        # TODO (slecleach) make one simulation step for a better test.
        jacobian = mjs.get_contact_jacobian(model, data, mjs.ContactLocation.RELATIVE)  # type: ignore
        self.assertIs(type(jacobian), np.ndarray)
        self.assertEqual(jacobian.shape, (0, 3, 7))
