# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco import MjData, MjModel

from judo.rollout.base import AbstractRolloutBackend
from judo.utils.patch import patch_mj_ccd_iterations

# TEMPORARY: Patch MJ_CCD_ITERATIONS globally for the session
# Once this PR is resolved, remove the monkey patch logic: https://github.com/google-deepmind/mujoco_warp/issues/456
patch_mj_ccd_iterations(32)


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

    def __init__(self, model: MjModel, num_threads: int) -> None:
        """Initialize the backend with a number of threads."""
        super().__init__(num_threads)
        self.mjm = model
        self.mjd = MjData(model)
        self.mwm = mjw.put_model(self.mjm)
        self.setup_mjwarp_backend(num_threads)

    def setup_mjwarp_backend(self, num_threads: int) -> None:
        """Setup the mujoco warp backend."""
        self.mwd = mjw.put_data(
            self.mjm,
            self.mjd,
            nworld=num_threads,
            nconmax=100000,
            njmax=400000,
        )  # TODO: expose these options as parameters in a rollout backend
        self.forward_graph = self.create_mjw_forward_graph(self.mwm, self.mwd)
        self.step_graph = self.create_mjw_step_graph(self.mwm, self.mwd)
        self.num_threads = num_threads

    def create_mjw_forward_graph(self, m: mjw.Model, d: mjw.Data) -> wp.context.Graph:
        """Create a CUDA graph for MuJoCo Warp forward function.

        Args:
            m: Mujoco Warp model.
            d: Mujoco Warp data.

        Returns:
            forward_graph: A CUDA graph that can be used to perform multiple forward passes efficiently.
        """
        mjw.forward(m, d)
        mjw.forward(m, d)
        with wp.ScopedCapture() as capture:
            mjw.forward(m, d)
        forward_graph = capture.graph
        return forward_graph

    def create_mjw_step_graph(self, m: mjw.Model, d: mjw.Data) -> wp.context.Graph:
        """Create a CUDA graph for MuJoCo Warp step function.

        Args:
            m: Mujoco Warp model.
            d: Mujoco Warp data.

        Returns:
            step_graph: A CUDA graph that can be used to perform multiple steps efficiently.
        """
        mjw.step(m, d)
        mjw.step(m, d)
        with wp.ScopedCapture() as capture:
            mjw.step(m, d)
        step_graph = capture.graph
        return step_graph

    def reset_mjw_data(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Resets MuJoCo Warp data to clear any intermediates.

        Args:
            qpos: Position vector to set for all worlds of shape (nq,).
            qvel: Velocity vector to set for all worlds of shape (nv,).
        """
        wp.launch(
            broadcast_rows,
            dim=(self.mwd.qpos.shape[0], self.mwd.qpos.shape[1]),
            inputs=[self.mwd.qpos, wp.array(qpos, dtype=wp.float32)],
            device=self.mwd.qpos.device,
        )
        wp.launch(
            broadcast_rows,
            dim=(self.mwd.qvel.shape[0], self.mwd.qvel.shape[1]),
            inputs=[self.mwd.qvel, wp.array(qvel, dtype=wp.float32)],
            device=self.mwd.qvel.device,
        )
        wp.capture_launch(self.forward_graph)  # perform forward pass
        self.mwd.qfrc_constraint.zero_()  # zero out constraint forces

    def rollout(self, x0: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout depending on the backend."""
        num_threads, num_steps, _ = controls.shape  # type: ignore
        states = wp.empty(
            (num_threads, num_steps, self.mjm.nq + self.mjm.nv),
            dtype=wp.float32,  # type: ignore  # typed as float in source code...
        )
        sensors = wp.empty(
            (num_threads, num_steps, self.mjm.nsensordata),
            dtype=wp.float32,  # type: ignore  # typed as float in source code...
        )

        # set data
        self.reset_mjw_data(x0[: self.mwm.nq], x0[self.mwm.nq :])
        us = wp.array(controls, dtype=wp.float32)  # type: ignore

        for step in range(num_steps):
            wp.copy(self.mwd.ctrl, us[:, step, :])  # type: ignore
            wp.capture_launch(self.step_graph)  # perform step

            # record result of step
            wp.copy(states[:, step, : self.mjm.nq], self.mwd.qpos)  # type: ignore
            wp.copy(states[:, step, self.mjm.nq :], self.mwd.qvel)  # type: ignore
            wp.copy(sensors[:, step, :], self.mwd.sensordata)  # type: ignore

        return states.numpy(), sensors.numpy()

    def update(self, num_threads: int) -> None:
        """Update the backend with a new number of threads."""
        self.setup_mjwarp_backend(num_threads)
