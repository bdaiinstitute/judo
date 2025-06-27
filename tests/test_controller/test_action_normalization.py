# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from contextlib import nullcontext as does_not_raise

import numpy as np

from judo.controller import Controller, ControllerConfig
from judo.optimizers import CrossEntropyMethod, CrossEntropyMethodConfig
from judo.tasks import CylinderPush, CylinderPushConfig
from judo.utils.normalization import IdentityNormalizer, MinMaxNormalizer, RunningMeanStdNormalizer

# ##### #
# TESTS #
# ##### #


def test_identity_normalizer() -> None:
    """Test that IdentityNormalizer correctly passes through values."""
    dim = 3
    normalizer = IdentityNormalizer(dim)

    # Test with random data
    data = np.random.randn(10, dim)
    normalized = normalizer.normalize(data)
    denormalized = normalizer.denormalize(normalized)

    # Check that the data is unchanged
    np.testing.assert_array_almost_equal(data, normalized)
    np.testing.assert_array_almost_equal(data, denormalized)


def test_min_max_normalizer() -> None:
    """Test that MinMaxNormalizer correctly scales values to [-1, 1] range."""
    dim = 3
    rand_vals = np.random.randn(dim, 2)
    min_vals = np.min(rand_vals, axis=1)
    max_vals = np.max(rand_vals, axis=1)

    normalizer = MinMaxNormalizer(dim, min_vals, max_vals)

    # Normalize min and max values
    min_normalized = normalizer.normalize(min_vals)
    max_normalized = normalizer.normalize(max_vals)

    # Should be close to -1 and 1 respectively
    np.testing.assert_array_almost_equal(min_normalized, -np.ones_like(min_normalized))
    np.testing.assert_array_almost_equal(max_normalized, np.ones_like(max_normalized))

    # Test with random data
    data = np.random.uniform(min_vals, max_vals, (10, dim))
    normalized = normalizer.normalize(data)
    denormalized = normalizer.denormalize(normalized)

    # Check that denormalizing the normalized data returns the original data
    np.testing.assert_array_almost_equal(data, denormalized)

    # Check that normalized values are in [-1, 1] range
    assert np.all(normalized >= -1.0 - 1e-6)
    assert np.all(normalized <= 1.0 + 1e-6)


def test_running_mean_std_normalizer() -> None:
    """Test that RunningMeanStdNormalizer correctly updates and normalizes."""
    dim = 3
    normalizer = RunningMeanStdNormalizer(dim)
    batch_size = 10
    n_iters = 3
    data = np.random.randn(batch_size * n_iters, dim)

    # Test iterative calls
    for i in range(n_iters):
        batch = data[i * batch_size : (i + 1) * batch_size]
        normalizer.update(batch)

        # Check that statistics were updated correctly
        count = (i + 1) * batch_size
        assert normalizer.count == count
        np.testing.assert_array_almost_equal(normalizer.mean, np.mean(data[:count], axis=0))
        np.testing.assert_array_almost_equal(normalizer.std, np.std(data[:count], axis=0))

    normalized = normalizer.normalize(data)
    denormalized = normalizer.denormalize(normalized)

    # Check that normalized data has mean 0 and std 1
    np.testing.assert_array_almost_equal(np.mean(normalized, axis=0), np.zeros(dim), decimal=5)
    np.testing.assert_array_almost_equal(np.std(normalized, axis=0), np.ones(dim), decimal=5)

    # Check that denormalizing the normalized data returns the original data
    np.testing.assert_array_almost_equal(data, denormalized, decimal=5)


def test_running_mean_std_normalizer_3d_data() -> None:
    """Test that RunningMeanStdNormalizer correctly updates with data with more than 2 dimensions."""
    dim = 3
    normalizer = RunningMeanStdNormalizer(dim)
    batch_size1, batch_size2 = 10, 2
    n_iters = 2
    data = np.random.randn(batch_size1 * n_iters, batch_size2, dim)

    # Test iterative calls
    for i in range(n_iters):
        batch = data[i * batch_size1 : (i + 1) * batch_size1]
        normalizer.update(batch)

        # Check that statistics were updated correctly
        count1 = (i + 1) * batch_size1
        assert normalizer.count == count1 * batch_size2
        np.testing.assert_array_almost_equal(normalizer.mean, np.mean(data[:count1], axis=(0, 1)))
        np.testing.assert_array_almost_equal(normalizer.std, np.std(data[:count1], axis=(0, 1)))

    normalized = normalizer.normalize(data)
    denormalized = normalizer.denormalize(normalized)

    # Check that normalized data has mean 0 and std 1
    np.testing.assert_array_almost_equal(np.mean(normalized, axis=(0, 1)), np.zeros(dim), decimal=5)
    np.testing.assert_array_almost_equal(np.std(normalized, axis=(0, 1)), np.ones(dim), decimal=5)

    # Check that denormalizing the normalized data returns the original data
    np.testing.assert_array_almost_equal(data, denormalized, decimal=5)


def test_normalizer_type_change() -> None:
    """Test that normalizer is re-initialized when type changes."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    controller = Controller(
        ControllerConfig(action_normalizer="none"),
        task,
        task_config,
        optimizer,
        optimizer_config,
        rollout_backend="mujoco",
    )

    # Initially should be IdentityNormalizer
    assert isinstance(controller.action_normalizer, IdentityNormalizer)

    # Change the normalizer type in config to MinMaxNormalizer
    controller.controller_cfg.action_normalizer = "min_max"

    # Run action update loop once
    curr_state = np.random.rand(controller.task.model.nq + controller.task.model.nv)
    curr_time = 0.0
    controller.update_action(curr_state, curr_time)

    # Should now be MinMaxNormalizer
    assert isinstance(controller.action_normalizer, MinMaxNormalizer)


def test_normalizer_in_update_action_loop() -> None:
    """Test that normalizers work in the update_action loop."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    # Test with different normalizer types
    for normalizer_type in ["none", "min_max", "running"]:
        controller = Controller(
            ControllerConfig(action_normalizer=normalizer_type, max_opt_iters=1),
            task,
            task_config,
            optimizer,
            optimizer_config,
            rollout_backend="mujoco",
        )

        curr_state = np.random.rand(controller.task.model.nq + controller.task.model.nv)
        curr_time = 0.0

        # This should run without error
        with does_not_raise():
            controller.update_action(curr_state, curr_time)


def test_min_max_normalizer_with_task_control_ranges() -> None:
    """Test that MinMaxNormalizer correctly uses task's actuator control ranges."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    controller = Controller(
        ControllerConfig(action_normalizer="min_max", max_opt_iters=1),
        task,
        task_config,
        optimizer,
        optimizer_config,
        rollout_backend="mujoco",
    )

    assert isinstance(controller.action_normalizer, MinMaxNormalizer)

    # Check that the normalizer is initialized with correct control ranges
    np.testing.assert_array_almost_equal(controller.action_normalizer.min, controller.task.actuator_ctrlrange[:, 0])
    np.testing.assert_array_almost_equal(controller.action_normalizer.max, controller.task.actuator_ctrlrange[:, 1])

    # Run optimization loop
    curr_state = np.random.rand(controller.task.model.nq + controller.task.model.nv)
    curr_time = 0.0
    controller.update_action(curr_state, curr_time)

    # Check that all candidate actions are within the control range bounds
    assert np.all(controller.candidate_knots >= controller.task.actuator_ctrlrange[:, 0] - 1e-6)
    assert np.all(controller.candidate_knots <= controller.task.actuator_ctrlrange[:, 1] + 1e-6)

    # Check that normalized actions are within the normalized control range bounds
    min_normalized = controller.action_normalizer.normalize(controller.task.actuator_ctrlrange[:, 0])
    max_normalized = controller.action_normalizer.normalize(controller.task.actuator_ctrlrange[:, 1])
    candidate_knots_normalized = controller.action_normalizer.normalize(controller.candidate_knots)
    assert np.all(candidate_knots_normalized >= min_normalized - 1e-6)
    assert np.all(candidate_knots_normalized <= max_normalized + 1e-6)


def test_running_normalizer_updates_with_optimizer_data() -> None:
    """Test that running normalizer correctly updates with optimizer data."""
    task_config = CylinderPushConfig()
    task = CylinderPush()
    optimizer_config = CrossEntropyMethodConfig()
    optimizer = CrossEntropyMethod(optimizer_config, task.nu)

    controller = Controller(
        ControllerConfig(action_normalizer="running", max_opt_iters=1),
        task,
        task_config,
        optimizer,
        optimizer_config,
        rollout_backend="mujoco",
    )

    # Check initial state
    assert isinstance(controller.action_normalizer, RunningMeanStdNormalizer)
    assert controller.action_normalizer.count == 0

    curr_state = np.random.rand(controller.task.model.nq + controller.task.model.nv)
    curr_time = 0.0

    # Run optimization loop
    controller.update_action(curr_state, curr_time)

    # Check that the normalizer was updated with the correct number of samples
    assert controller.action_normalizer.count == controller.optimizer.num_rollouts * controller.optimizer.num_nodes

    # Check that the normalizer was updated with the correct mean and std of candidate knots
    np.testing.assert_array_almost_equal(
        controller.action_normalizer.mean, np.mean(controller.candidate_knots, axis=(0, 1))
    )
    np.testing.assert_array_almost_equal(
        controller.action_normalizer.std, np.std(controller.candidate_knots, axis=(0, 1))
    )
