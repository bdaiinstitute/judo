# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Callable

import numpy as np

from judo.tasks.spot import SpotBase, SpotTireUpright


def test_spot_base_init() -> None:
    """Test SpotBase task initializes correctly."""
    task = SpotBase()
    assert task.name == "spot_base"
    assert task.physics_substeps == 2
    assert task.locomotion_policy_path is not None
    # Base + arm = 3 + 7 = 10
    assert task.nu == 10


def test_spot_tire_upright_init() -> None:
    """Test SpotTireUpright task initializes correctly."""
    task = SpotTireUpright()
    assert task.name == "spot_tire_upright"
    assert task.physics_substeps == 2
    assert task.locomotion_policy_path is not None
    # Base + arm + legs + leg_selection = 3 + 7 + 6 + 1 = 17
    assert task.nu == 17


def test_spot_tire_upright_reset(temp_np_seed: Callable) -> None:
    """Test SpotTireUpright reset generates valid poses."""
    task = SpotTireUpright()
    with temp_np_seed(42):
        task.reset()
    assert task.data.qpos.shape[0] == task.model.nq
    assert task.data.qvel.shape[0] == task.model.nv


def test_spot_tire_upright_task_to_sim_ctrl() -> None:
    """Test task_to_sim_ctrl maps compact controls to 25-dim policy command."""
    task = SpotTireUpright()
    command = np.zeros(task.nu)
    sim_ctrl = task.task_to_sim_ctrl(command)
    assert sim_ctrl.shape[-1] == 25


def test_spot_tire_upright_reward() -> None:
    """Test reward function runs with correct shapes."""
    task = SpotTireUpright()
    nq = task.model.nq
    nv = task.model.nv
    nsensor = task.model.nsensordata
    batch_size = 4
    T = 3

    states = np.zeros((batch_size, T, nq + nv))
    # Set reasonable body height to avoid divide-by-zero
    states[..., task.body_pose_idx + 2] = 0.52
    states[..., task.object_pose_idx : task.object_pose_idx + 3] = [1.0, 0.0, 0.17]

    sensors = np.zeros((batch_size, T, nsensor))
    controls = np.zeros((batch_size, T, task.nu))

    reward = task.reward(states, sensors, controls)
    assert reward.shape == (batch_size,)


def test_spot_base_actuator_ctrlrange() -> None:
    """Test actuator_ctrlrange shape matches nu."""
    task = SpotBase()
    ctrlrange = task.actuator_ctrlrange
    assert ctrlrange.shape == (task.nu, 2)
