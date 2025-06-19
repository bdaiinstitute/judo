# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import numpy as np

from judo.controller import Controller, ControllerConfig
from judo.optimizers import Optimizer, OptimizerConfig, get_registered_optimizers
from judo.tasks import CylinderPush, CylinderPushConfig

# ##### #
# TESTS #
# ##### #


def test_update_action() -> None:
    """Tests the update_action method with different optimizers."""

    def _setup_controller(opt_cls: type[Optimizer], opt_cfg: OptimizerConfig) -> Controller:
        """Helper function to set up the controller."""
        task = CylinderPush()
        task_config = CylinderPushConfig()
        opt = opt_cls(opt_cfg, task.nu)
        controller = Controller(
            ControllerConfig(max_opt_iters=2),
            task,
            task_config,
            opt,
            opt_cfg,
            rollout_backend="mujoco",
        )
        return controller

    # test with all registered optimizers
    for _opt_name, (opt_cls, opt_cfg) in get_registered_optimizers().items():
        controller = _setup_controller(opt_cls, opt_cfg)
        curr_state = np.random.rand(controller.task.model.nq + controller.task.model.nv)
        curr_time = 0.0

        # check that update_action runs without error
        controller.update_action(curr_state, curr_time)

        # check that the nominal knots have the correct shape
        assert controller.nominal_knots.shape == (controller.optimizer.num_nodes, controller.optimizer.nu)

        # check that the candidate knots have the correct shape
        assert controller.candidate_knots.shape == (
            controller.optimizer.num_rollouts,
            controller.optimizer.num_nodes,
            controller.optimizer.nu,
        )
