# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Tests for all Spot tasks - verifies instantiation, dimensions, reward, and reset."""

from typing import Any

import numpy as np
import pytest

from judo.tasks.spot import (
    SpotBase,
    SpotBoxPush,
    SpotNavigate,
    SpotTireRoll,
    SpotTireUpright,
)

# (TaskClass, expected_nu)
# nu depends on use_arm, use_gripper, use_legs, use_torso:
#   base only: 3
#   base + arm: 10 (3 base + 7 arm)
#   base + arm + gripper: 11 (3 base + 7 arm + 1 gripper_selection)
#   base + arm + legs: 17 (3 base + 7 arm + 6 legs + 1 leg_selection)
#   base + arm + torso: 13 (3 base + 7 arm + 3 torso)
#   base + arm + gripper + torso: 14 (3 base + 7 arm + 1 gripper_selection + 3 torso)
SPOT_TASKS = [
    (SpotBase, 10),  # use_arm=True
    (SpotNavigate, 3),  # use_arm=False, base only
    (SpotTireUpright, 17),  # use_arm=True, use_legs=True: 3+7+6+1
    (SpotTireRoll, 11),  # use_arm=True, use_gripper=True
    (SpotBoxPush, 10),  # use_arm=True
]


@pytest.fixture(params=SPOT_TASKS, ids=lambda x: x[0].__name__)
def spot_task_and_nu(request: pytest.FixtureRequest) -> Any:
    """Fixture that instantiates each Spot task and provides expected nu."""
    task_cls, expected_nu = request.param
    task = task_cls()
    return task, expected_nu


class TestSpotTaskInstantiation:
    """Test that all Spot tasks can be instantiated (XML loads, sensors resolve)."""

    def test_instantiation(self, spot_task_and_nu: Any) -> None:
        """Verify model and data are initialized."""
        task, _ = spot_task_and_nu
        assert task.model is not None
        assert task.data is not None

    def test_nu(self, spot_task_and_nu: Any) -> None:
        """Verify action dimension matches expected value."""
        task, expected_nu = spot_task_and_nu
        assert task.nu == expected_nu, f"{task.__class__.__name__}: expected nu={expected_nu}, got {task.nu}"


class TestSpotTaskReward:
    """Test that reward computation works with random inputs."""

    def test_reward_shape(self, spot_task_and_nu: Any) -> None:
        """Verify reward output shape."""
        task, _ = spot_task_and_nu
        batch_size = 4
        T = 5
        nq = task.model.nq
        nv = task.model.nv
        nsensor = task.model.nsensordata
        nu = task.nu

        states = np.random.randn(batch_size, T, nq + nv)
        sensors = np.random.randn(batch_size, T, nsensor)
        controls = np.random.randn(batch_size, T, nu)

        reward = task.reward(states, sensors, controls)
        assert reward.shape == (batch_size,), (
            f"{task.__class__.__name__}: expected reward shape ({batch_size},), got {reward.shape}"
        )

    def test_reward_finite(self, spot_task_and_nu: Any) -> None:
        """Verify reward contains only finite values."""
        task, _ = spot_task_and_nu
        batch_size = 2
        T = 3
        nq = task.model.nq
        nv = task.model.nv
        nsensor = task.model.nsensordata
        nu = task.nu

        states = np.random.randn(batch_size, T, nq + nv) * 0.1
        sensors = np.random.randn(batch_size, T, nsensor) * 0.1
        controls = np.random.randn(batch_size, T, nu) * 0.1

        reward = task.reward(states, sensors, controls)
        assert np.all(np.isfinite(reward)), f"{task.__class__.__name__}: reward contains non-finite values"


class TestSpotTaskResetPose:
    """Test that reset_pose returns valid qpos arrays."""

    def test_reset_pose_shape(self, spot_task_and_nu: Any) -> None:
        """Verify reset_pose shape matches model nq."""
        task, _ = spot_task_and_nu
        if not hasattr(task, "reset_pose"):
            pytest.skip(f"{task.__class__.__name__} has no reset_pose")
        pose = task.reset_pose
        assert pose.shape == (task.model.nq,), (
            f"{task.__class__.__name__}: expected reset_pose shape ({task.model.nq},), got {pose.shape}"
        )

    def test_reset_pose_finite(self, spot_task_and_nu: Any) -> None:
        """Verify reset_pose contains only finite values."""
        task, _ = spot_task_and_nu
        if not hasattr(task, "reset_pose"):
            pytest.skip(f"{task.__class__.__name__} has no reset_pose")
        pose = task.reset_pose
        assert np.all(np.isfinite(pose)), f"{task.__class__.__name__}: reset_pose contains non-finite values"
