# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco import MjData, MjModel
from mujoco_warp._src.types import IntegratorType

from judo.rollout.base import AbstractRolloutBackend
from judo.utils.patch import patch_mj_ccd_iterations

# TEMPORARY: Patch MJ_CCD_ITERATIONS globally for the session
# Once this PR is resolved, remove the monkey patch logic: https://github.com/google-deepmind/mujoco_warp/issues/456
patch_mj_ccd_iterations(24)


@wp.kernel
def broadcast_rows(
    mat: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    vec: wp.array(dtype=wp.float32),  # type: ignore
) -> None:
    """Broadcast a vector to each row of a matrix."""
    r, c = wp.tid()  # type: ignore
    mat[r, c] = vec[c]


class MjwarpRolloutBackend(AbstractRolloutBackend):
    """The rollout backend using MuJoCo Warp."""

    def __init__(self, model: MjModel, num_threads: int, num_timesteps: int) -> None:
        """Initialize the backend with a number of threads."""
        super().__init__(num_threads)
        self.mjm = model
        self.mjd = MjData(model)
        self.mwm = mjw.put_model(self.mjm)
        self.mwm.opt.ls_parallel = True  # critical for speed
        self.mwm.opt.integrator = IntegratorType.IMPLICITFAST
        self.setup_mjwarp_backend(num_threads, num_timesteps)
        self.mwd.time = self.mjd.time  # ensure time is initialized correctly after warmups

    def setup_mjwarp_backend(self, num_threads: int, num_timesteps: int) -> None:
        """Setup the mujoco warp backend."""
        self.num_threads = num_threads
        self.num_steps = num_timesteps
        self.mwd = mjw.put_data(
            self.mjm,
            self.mjd,
            nworld=num_threads,
            nconmax=10000,
            njmax=40000,
        )  # TODO: expose these options as parameters in a rollout backend
        self.reset_graph = self.create_reset_graph(self.mwm, self.mwd)
        self.rollout_graph = self.create_rollout_graph(self.mwm, self.mwd)

    def create_reset_graph(self, m: mjw.Model, d: mjw.Data) -> wp.context.Graph:
        """Create a CUDA graph for resetting MuJoCo Warp data.

        Args:
            m: Mujoco Warp model.
            d: Mujoco Warp data.
        """
        # create qpos and qvel buffers
        self.qpos_buffer = wp.empty(m.nq, dtype=wp.float32)  # type: ignore
        self.qvel_buffer = wp.empty(m.nv, dtype=wp.float32)  # type: ignore

        # warmup
        mjw.forward(m, d)
        mjw.forward(m, d)

        # capture reset graph
        with wp.ScopedCapture() as capture:
            wp.launch(
                broadcast_rows,
                dim=(self.mwd.qpos.shape[0], self.mwd.qpos.shape[1]),
                inputs=[self.mwd.qpos, self.qpos_buffer],
                device=self.mwd.qpos.device,
            )
            wp.launch(
                broadcast_rows,
                dim=(self.mwd.qvel.shape[0], self.mwd.qvel.shape[1]),
                inputs=[self.mwd.qvel, self.qvel_buffer],
                device=self.mwd.qvel.device,
            )
            mjw.forward(m, d)
            self.mwd.qfrc_constraint.zero_()  # zero out constraint forces
        return capture.graph

    def create_rollout_graph(self, m: mjw.Model, d: mjw.Data) -> wp.context.Graph:
        """Create a CUDA graph for the entire rollout, including recording.

        Args:
            m: Mujoco Warp model.
            d: Mujoco Warp data.
        """
        # create buffers
        self.states_buffer = wp.empty(
            (self.num_threads, self.num_steps, m.nq + m.nv),
            dtype=wp.float32,  # type: ignore
        )
        self.sensors_buffer = wp.empty(
            (self.num_threads, self.num_steps, m.nsensordata),
            dtype=wp.float32,  # type: ignore
        )
        self.controls_buffer = wp.empty(
            (self.num_threads, self.num_steps, m.nu),
            dtype=wp.float32,  # type: ignore
        )

        # warmup before graph capture
        mjw.step(m, d)
        mjw.step(m, d)

        with wp.ScopedCapture() as capture:
            for step in range(self.num_steps):
                # perform a control step
                wp.copy(d.ctrl, self.controls_buffer[:, step, :])
                mjw.step(m, d)

                # record the result of the step
                wp.copy(self.states_buffer[:, step, : m.nq], d.qpos)  # type: ignore
                wp.copy(self.states_buffer[:, step, m.nq :], d.qvel)  # type: ignore
                wp.copy(self.sensors_buffer[:, step, :], d.sensordata)  # type: ignore
        return capture.graph

    def rollout(self, x0: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout using the MuJoCo Warp backend."""
        # copy qpos, vel, and controls to pre-allocated buffers
        wp.copy(self.qpos_buffer, wp.array(x0[: self.mjm.nq], dtype=wp.float32))
        wp.copy(self.qvel_buffer, wp.array(x0[self.mjm.nq :], dtype=wp.float32))
        wp.copy(self.controls_buffer, wp.array(controls, dtype=wp.float32))

        # reset and rollout
        wp.capture_launch(self.reset_graph)
        wp.capture_launch(self.rollout_graph)

        return self.states_buffer.numpy(), self.sensors_buffer.numpy()

    def update(self, num_threads: int, num_timesteps: int | None = None) -> None:
        """Update the backend with a new number of threads."""
        self.setup_mjwarp_backend(num_threads, num_timesteps or self.num_steps)
