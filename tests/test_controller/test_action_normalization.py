# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import numpy as np

from judo.controller import Controller, ControllerConfig
from judo.optimizers import CrossEntropyMethod, CrossEntropyMethodConfig
from judo.tasks import CylinderPush, CylinderPushConfig
from judo.utils.normalization import IdentityNormalizer, MinMaxNormalizer

# ##### #
# TESTS #
# ##### #


def test_identity_normalizer() -> None:
    """Test that identity normalizer correctly passes through values."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    controller = Controller(
        ControllerConfig(action_normalizer="identity"),
        task,
        task_config,
        optimizer,
        optimizer_config,
        rollout_backend="mujoco",
    )

    # Test with random actions
    actions = np.random.randn(10, task.nu)
    normalized = controller.action_normalizer.normalize(actions)
    denormalized = controller.action_normalizer.denormalize(normalized)

    np.testing.assert_array_almost_equal(actions, normalized)
    np.testing.assert_array_almost_equal(actions, denormalized)


def test_min_max_normalizer_behavior() -> None:
    """Test that min_max normalizer correctly scales values to [-1, 1] range."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    controller = Controller(
        ControllerConfig(action_normalizer="min_max"),
        task,
        task_config,
        optimizer,
        optimizer_config,
        rollout_backend="mujoco",
    )

    # Test with actions at the bounds
    min_actions = task.actuator_ctrlrange[:, 0]
    max_actions = task.actuator_ctrlrange[:, 1]

    # Normalize min and max values
    min_normalized = controller.action_normalizer.normalize(min_actions)
    max_normalized = controller.action_normalizer.normalize(max_actions)

    # Should be close to -1 and 1 respectively
    np.testing.assert_array_almost_equal(min_normalized, -np.ones_like(min_normalized))
    np.testing.assert_array_almost_equal(max_normalized, np.ones_like(max_normalized))

    # Test round-trip normalization
    actions = np.random.uniform(min_actions, max_actions, (10, task.nu))
    normalized = controller.action_normalizer.normalize(actions)
    denormalized = controller.action_normalizer.denormalize(normalized)

    np.testing.assert_array_almost_equal(actions, denormalized)

    # Check that normalized values are in [-1, 1] range
    assert np.all(normalized >= -1.0 - 1e-6)
    assert np.all(normalized <= 1.0 + 1e-6)


def test_running_mean_std_normalizer() -> None:
    """Test that running_mean_std normalizer correctly updates and normalizes."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    controller = Controller(
        ControllerConfig(action_normalizer="running_mean_std"),
        task,
        task_config,
        optimizer,
        optimizer_config,
        rollout_backend="mujoco",
    )

    # Test with some random actions
    actions = np.random.randn(10, task.nu)
    controller.action_normalizer.update(actions)

    # Check that statistics were updated
    assert controller.action_normalizer.count == 10
    assert not np.allclose(controller.action_normalizer.mean, np.zeros(task.nu))
    assert not np.allclose(controller.action_normalizer.std, np.ones(task.nu))

    # Test normalization
    normalized = controller.action_normalizer.normalize(actions)
    denormalized = controller.action_normalizer.denormalize(normalized)

    np.testing.assert_array_almost_equal(actions, denormalized)


def test_normalizer_type_change() -> None:
    """Test that normalizer is re-initialized when type changes."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    controller = Controller(
        ControllerConfig(action_normalizer="identity"),
        task,
        task_config,
        optimizer,
        optimizer_config,
        rollout_backend="mujoco",
    )

    # Initially should be identity normalizer
    assert isinstance(controller.action_normalizer, IdentityNormalizer)

    # Change the normalizer type in config
    controller.controller_cfg.action_normalizer = "min_max"

    # Update with some actions and rewards
    actions = np.random.randn(10, task.nu)
    rewards = np.random.randn(10)
    controller.update_action_normalizer(actions, rewards)

    # Should now be min_max normalizer
    assert isinstance(controller.action_normalizer, MinMaxNormalizer)


def test_normalizer_in_update_action_loop() -> None:
    """Test that normalizers work in the update_action loop."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    # Test with different normalizer types
    for normalizer_type in ["identity", "min_max", "running_mean_std"]:
        controller = Controller(
            ControllerConfig(action_normalizer=normalizer_type, max_opt_iters=2),
            task,
            task_config,
            optimizer,
            optimizer_config,
            rollout_backend="mujoco",
        )

        curr_state = np.random.rand(controller.task.model.nq + controller.task.model.nv)
        curr_time = 0.0

        # This should run without error
        controller.update_action(curr_state, curr_time)
