# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import numpy as np
from mujoco import MjData, MjModel
from mujoco.rollout import Rollout

from judo.rollout.base import AbstractRolloutBackend


class MujocoRolloutBackend(AbstractRolloutBackend):
    """The rollout backend using multithreaded Mujoco C."""

    def __init__(self, model: MjModel, num_threads: int, num_steps: int) -> None:
        """Initialize the backend with a number of threads."""
        super().__init__(num_threads, num_steps)
        self.model = model
        self.setup_mujoco_backend(num_threads)

    def make_model_data_pairs(self, model: MjModel, num_pairs: int) -> list[tuple[MjModel, MjData]]:
        """Create model/data pairs for mujoco threaded rollout."""
        models = [deepcopy(model) for _ in range(num_pairs)]
        datas = [MjData(m) for m in models]
        model_data_pairs = list(zip(models, datas, strict=True))
        return model_data_pairs

    def setup_mujoco_backend(self, num_threads: int) -> None:
        """Setup the mujoco backend."""
        self.rollout_obj = Rollout(nthread=num_threads)
        self.mj_rollout_func = lambda m, d, x0, u: self.rollout_obj.rollout(m, d, x0, u)
        self.model_data_pairs = self.make_model_data_pairs(self.model, num_threads)
        self.num_threads = num_threads

    def rollout(self, x0: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout depending on the backend."""
        # unpack models into a list of models and data
        ms, ds = zip(*self.model_data_pairs, strict=True)
        ms = list(ms)
        ds = list(ds)

        # getting shapes
        nq = ms[0].nq
        nv = ms[0].nv
        nu = ms[0].nu

        # the state passed into mujoco's rollout function includes the time
        # shape = (num_rollouts, num_states + 1)
        x0_batched = np.tile(x0, (len(ms), 1))
        full_states = np.concatenate([time.time() * np.ones((len(ms), 1)), x0_batched], axis=-1)
        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert controls.ndim == 3
        assert controls.shape[-1] == nu
        assert controls.shape[0] == full_states.shape[0]

        _states, _out_sensors = self.mj_rollout_func(ms, ds, full_states, controls)
        out_states = np.asarray(_states)[..., 1:]  # remove time from state
        out_sensors = np.asarray(_out_sensors)

        return out_states, out_sensors

    def update(self, num_threads: int, num_steps: int) -> None:
        """Update the backend with a new number of threads."""
        if num_threads != self.num_threads:  # ignore num_steps for Mujoco backend
            self.rollout_obj.close()
            self.setup_mujoco_backend(num_threads)
