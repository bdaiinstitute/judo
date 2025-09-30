# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import numpy as np

from judo.app.data.controller_data import ControllerData
from judo.optimizers.cem import CrossEntropyMethod
from judo.optimizers.mppi import MPPI, MPPIConfig
from judo.tasks import Cartpole, CartpoleConfig


def test_controller_data_basics() -> None:
    """Test the controller data."""
    controller_data = ControllerData(init_task="cylinder_push", init_optimizer="cem")
    assert controller_data.task is not None
    assert controller_data.optimizer is not None
    assert controller_data.controller is not None
    controller_data.pause()
    assert controller_data.paused


def test_controller_data_update_task() -> None:
    """Test the controller data update task."""
    controller_data = ControllerData(init_task="cylinder_push", init_optimizer="cem")
    assert controller_data.task is not None
    assert controller_data.optimizer is not None
    assert controller_data.controller is not None
    assert isinstance(controller_data.optimizer, CrossEntropyMethod)
    res = controller_data.available_optimizers.get("mppi")
    assert res is not None
    mppi_opt, _ = res
    controller_data.update_task(
        task=Cartpole(),
        task_config=CartpoleConfig(),
        optimizer=mppi_opt(controller_data.optimizer_config, controller_data.task.nu),
    )
    assert isinstance(controller_data.optimizer, MPPI)


def test_controller_data_reset_task() -> None:
    """Test the controller data reset task."""
    np.random.seed(17)
    controller_data = ControllerData(init_task="cylinder_push", init_optimizer="cem")
    original_states = np.copy(controller_data.states)
    original_time = controller_data.curr_time
    controller_data.reset_task()
    assert not np.allclose(controller_data.states, original_states)
    assert controller_data.curr_time == original_time


def test_step() -> None:
    """Test the controller data step."""
    np.random.seed(24)
    controller_data = ControllerData(init_task="cylinder_push", init_optimizer="cem")
    original_plan_time = controller_data.last_plan_time
    original_states = np.copy(controller_data.controller.states)
    controller_data.step()
    assert not np.allclose(controller_data.controller.states, original_states)
    assert controller_data.last_plan_time > original_plan_time


def test_update_optimizer() -> None:
    """Test the controller data update optimizer."""
    controller_data = ControllerData(init_task="cylinder_push", init_optimizer="cem")
    assert isinstance(controller_data.optimizer, CrossEntropyMethod)
    controller_data.update_optimizer(
        optimizer=MPPI(controller_data.optimizer_config, controller_data.task.nu),
        optimizer_config_cls=MPPIConfig,
        optimizer_config=MPPIConfig(),
        optimizer_cls=MPPI,
    )
    assert isinstance(controller_data.optimizer, MPPI)
    assert isinstance(controller_data.optimizer_config, MPPIConfig)
    assert isinstance(controller_data.controller.optimizer, MPPI)
    assert controller_data.optimizer_cls == MPPI
    assert controller_data.optimizer_config_cls == MPPIConfig


def test_spline_data() -> None:
    """Test the controller data spline data."""
    controller_data = ControllerData(init_task="cylinder_push", init_optimizer="cem")
    assert controller_data.spline_data is not None
