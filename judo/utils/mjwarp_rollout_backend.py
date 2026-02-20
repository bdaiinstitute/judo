# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""GPU-accelerated rollout backend using mujoco_warp (NVIDIA Warp)."""

import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco import MjData, MjModel

from judo.controller.batched_spot_locomotion import BatchedSpotLocomotion
from judo.utils.rollout_backend import RolloutBackend
from judo.utils.timer import Timer

NCONMAX = 256
NJMAX = 900
DEVICE = "cuda:0"


@wp.kernel
def _copy_target_to_ctrl(
    target_q: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    n_controlled: int,
):
    """Copy target_q into the first n_controlled columns of ctrl."""
    i = wp.tid()
    for j in range(n_controlled):
        ctrl[i, j] = target_q[i, j]


class PassThroughController:
    """Passes commands through as target joint positions (no locomotion policy)."""

    def compute_batch(self, cmd, qpos, qvel, previous_actions):
        return cmd, None

    def reset(self):
        pass

    @property
    def target_frequency(self):
        return float("inf")


class MJWarpRolloutBackend(RolloutBackend):
    """GPU-accelerated rollout backend using mujoco_warp.

    Supports two modes:
    - Direct control: controls directly set actuator commands (default)
    - Hierarchical control: controls are processed through locomotion policy + PD controller
    """

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        num_problems: int = 1,
        locomotion_controller: BatchedSpotLocomotion | None = None,
        device: str = DEVICE,
    ) -> None:
        """Initialize the backend with optional hierarchical control.

        Args:
            model: MuJoCo model.
            num_threads: Number of parallel rollouts per problem.
            num_problems: Number of independent problems.
            locomotion_controller: Locomotion policy (e.g. BatchedSpotLocomotion) or None.
            device: Device for GPU operations.
        """
        assert device != "cpu", "RolloutBackend requires a GPU device."

        self.device = device
        self.model = model
        self.data = MjData(self.model)
        self.num_threads = num_threads
        self.num_problems = num_problems
        self.num_worlds = num_threads * num_problems

        with wp.ScopedDevice(device):
            self.mjw_model = mjw.put_model(self.model)
            self.mjw_model.opt.ccd_iterations = 60
            self.mjw_data = mjw.put_data(
                self.model,
                self.data,
                nworld=self.num_worlds,
                nconmax=NCONMAX,
                njmax=NJMAX,
            )

        # Warm up kernels before CUDA graph capture (compilation not allowed during capture)
        mjw.step(self.mjw_model, self.mjw_data)
        wp.synchronize()

        # Capture CUDA graph for step function
        with wp.ScopedCapture() as capture:
            mjw.step(self.mjw_model, self.mjw_data)
        self.mjw_step_graph = capture.graph

        # Initialize hierarchical control (optional)
        self.locomotion_controller = locomotion_controller if locomotion_controller else PassThroughController()

        # Calculate policy decimation
        physics_dt = model.opt.timestep
        self.policy_decimation = max(
            1, int(1.0 / (self.locomotion_controller.target_frequency * physics_dt))
        )
        self.global_step_counter = 0

        # Initialize timers for performance measurement
        self.timer_cpu_to_gpu = Timer("CPU->GPU", unit="ms")
        self.timer_rollout = Timer("Rollout ", unit="ms")
        self.timer_gpu_to_cpu = Timer("GPU->CPU", unit="ms")

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: wp.array | None = None,
    ) -> tuple[np.ndarray, np.ndarray, wp.array | None]:
        """Conduct a GPU-accelerated rollout using mujoco_warp.

        Supports both single-problem and multi-problem modes:
        - Single problem: x0 shape (nq + nv,), controls shape (num_threads, horizon, nu)
        - Multi-problem: x0 shape (num_problems, nq + nv), controls shape (num_problems*num_threads, horizon, nu)

        Args:
            x0: Initial state(s) as [qpos, qvel] (no time).
            controls: Control inputs.
            last_policy_output: Previous policy outputs as warp array on GPU,
                shape (num_worlds, policy_dim), or None.

        Returns:
            Tuple of (states, sensors, policy_output):
            - states: shape (num_worlds, horizon, nq + nv)
            - sensors: shape (num_worlds, horizon, nsensordata)
            - policy_output: Final policy outputs as warp array on GPU, or None.
        """
        nq = self.model.nq
        nv = self.model.nv
        nu = self.model.nu
        nsensordata = self.model.nsensordata
        horizon = controls.shape[1]

        if x0.ndim == 1:
            x0_batched = np.tile(x0, (self.num_threads, 1))
        else:
            x0_batched = np.repeat(x0, self.num_threads, axis=0)

        num_worlds = x0_batched.shape[0]

        full_states = np.concatenate([np.zeros((num_worlds, 1)), x0_batched], axis=-1)

        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[0] == num_worlds

        # CPU -> GPU copy
        self.timer_cpu_to_gpu.tic()
        full_states_wp = wp.array(full_states, dtype=wp.float32)
        controls_wp = wp.array(controls, dtype=wp.float32)

        out_qpos_wp = wp.zeros((num_worlds, horizon, nq), dtype=wp.float32)
        out_qvel_wp = wp.zeros((num_worlds, horizon, nv), dtype=wp.float32)
        out_sensors_wp = wp.zeros((num_worlds, horizon, nsensordata), dtype=wp.float32)

        wp.copy(self.mjw_data.time, full_states_wp[:, 0])
        wp.copy(self.mjw_data.qpos, full_states_wp[:, 1 : nq + 1])
        wp.copy(self.mjw_data.qvel, full_states_wp[:, 1 + nq : nq + nv + 1])
        wp.synchronize()
        self.timer_cpu_to_gpu.toc()

        # GPU rollout loop
        self.timer_rollout.tic()

        qpos_wp = wp.zeros((num_worlds, nq), dtype=wp.float32)
        qvel_wp = wp.zeros((num_worlds, nv), dtype=wp.float32)
        ctrl_wp = wp.zeros((num_worlds, nu), dtype=wp.float32)
        target_q_wp = None

        previous_actions_wp = last_policy_output

        for t in range(horizon):
            wp.synchronize()

            cmd_wp = controls_wp[:, t, :]

            # Update locomotion policy
            if (
                target_q_wp is None
                or self.global_step_counter % self.policy_decimation == 0
            ):
                target_q_wp, previous_actions_wp = self.locomotion_controller.compute_batch(
                    cmd_wp, qpos_wp, qvel_wp, previous_actions_wp
                )

            # Copy target_q into ctrl (pad remaining actuators with zeros)
            n_controlled = target_q_wp.shape[1]
            wp.launch(
                _copy_target_to_ctrl,
                dim=num_worlds,
                inputs=[target_q_wp, ctrl_wp, n_controlled],
                device=self.device,
            )
            wp.copy(self.mjw_data.ctrl, ctrl_wp)

            wp.capture_launch(self.mjw_step_graph)

            self.global_step_counter += 1

            wp.copy(qpos_wp, self.mjw_data.qpos)
            wp.copy(qvel_wp, self.mjw_data.qvel)
            wp.copy(out_qpos_wp[:, t, :], self.mjw_data.qpos)
            wp.copy(out_qvel_wp[:, t, :], self.mjw_data.qvel)
            wp.copy(out_sensors_wp[:, t, :], self.mjw_data.sensordata)

        wp.synchronize()
        self.timer_rollout.toc()

        # GPU -> CPU copy
        self.timer_gpu_to_cpu.tic()
        out_states = np.zeros((num_worlds, horizon, nq + nv), dtype=np.float32)
        out_states[:, :, :nq] = out_qpos_wp.numpy()
        out_states[:, :, nq : nq + nv] = out_qvel_wp.numpy()
        out_sensors = out_sensors_wp.numpy()
        self.timer_gpu_to_cpu.toc()

        return out_states, out_sensors, previous_actions_wp

    def update(self, num_threads: int, num_problems: int = 1) -> None:
        """Update the backend with a new number of threads."""
        self.num_threads = num_threads
        self.num_problems = num_problems
        self.num_worlds = num_threads * num_problems

        with wp.ScopedDevice(DEVICE):
            self.mjw_model = mjw.put_model(self.model)
            self.mjw_model.opt.ccd_iterations = 60
            self.mjw_data = mjw.put_data(
                self.model,
                self.data,
                nworld=self.num_worlds,
                nconmax=NCONMAX,
                njmax=NJMAX,
            )

        # Warm up kernels before CUDA graph capture
        mjw.step(self.mjw_model, self.mjw_data)
        wp.synchronize()

        with wp.ScopedCapture() as capture:
            mjw.step(self.mjw_model, self.mjw_data)
        self.mjw_step_graph = capture.graph

    def print_timer_stats(self) -> None:
        """Print timing statistics for all rollout operations."""
        self.timer_cpu_to_gpu.print_stats()
        self.timer_rollout.print_stats()
        self.timer_gpu_to_cpu.print_stats()

    def reset_timers(self) -> None:
        """Reset all timing statistics."""
        self.timer_cpu_to_gpu.reset()

    def reset(self) -> None:
        """Reset internal state (controllers and step counter)."""
        self.locomotion_controller.reset()
        self.global_step_counter = 0
        self.timer_rollout.reset()
        self.timer_gpu_to_cpu.reset()
