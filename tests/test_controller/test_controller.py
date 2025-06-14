from typing import Callable

import numpy as np

from judo.controller import Controller, ControllerConfig
from judo.optimizers import Optimizer, OptimizerConfig
from judo.tasks import CylinderPush, CylinderPushConfig

# ##### #
# MOCKS #
# ##### #

class MockOptimizerTrackNominalKnots(Optimizer):
    """A mock optimizer to track the history of nominal_knots."""

    def __init__(self, opt_config: OptimizerConfig, nu: int) -> None:
        """Initializes the mock optimizer."""
        super().__init__(opt_config, nu)
        self.received_knots_history: list[np.ndarray] = []

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots and tracks the history."""
        self.received_knots_history.append(nominal_knots.copy())
        num_rollouts = self.num_rollouts
        sampled_knots = nominal_knots + np.random.randn(num_rollouts, self.config.num_nodes, self.nu)
        return sampled_knots

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Selects something."""
        return sampled_knots[0]

# ##### #
# TESTS #
# ##### #

def test_max_opt_iters(temp_np_seed: Callable) -> None:
    """Tests that max_opt_iters correctly applies multiple iterations of optimization to a solution."""

    def _setup_controller(max_opt_iters: int) -> tuple[MockOptimizerTrackNominalKnots, Controller]:
        """Helper function to set up the controller."""
        task = CylinderPush()
        task_config = CylinderPushConfig()
        ps_config = OptimizerConfig()
        opt = MockOptimizerTrackNominalKnots(ps_config, task.nu)
        controller = Controller(
            ControllerConfig(max_opt_iters=max_opt_iters),
            task,
            task_config,
            opt,
            ps_config,
            rollout_backend="mujoco",
        )
        return opt, controller

    # generate a solution using max_opt_iters=1
    with temp_np_seed(42):
        opt1, controller1 = _setup_controller(max_opt_iters=1)
        curr_state1 = np.random.rand(controller1.task.model.nq + controller1.task.model.nv)
        curr_time = 0.0
        controller1.update_action(curr_state1, curr_time)

    # generate a solution using max_opt_iters=2
    with temp_np_seed(42):
        opt2, controller2 = _setup_controller(max_opt_iters=2)
        curr_state2 = np.random.rand(controller2.task.model.nq + controller2.task.model.nv)
        curr_time = 0.0
        controller2.update_action(curr_state2, curr_time)

    # check that the initial knots match in the optimization iterations
    assert np.array_equal(opt1.received_knots_history[0], opt2.received_knots_history[0])

    # check that the final knot in opt2 is not the same as the initial knot, matches the nominal knots of controller1
    assert not np.array_equal(opt2.received_knots_history[-1], opt2.received_knots_history[0])
    assert np.array_equal(opt2.received_knots_history[-1], controller1.nominal_knots)
