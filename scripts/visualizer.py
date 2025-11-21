# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from mujoco import mj_forward
from viser import GuiEvent, ViserServer

from judo.tasks import get_registered_tasks
from judo.visualizers.model import ViserMjModel


class BenchmarkRunVisualizer:
    """Interactive viewer for benchmark runs stored in an HDF5 file.

    Structure expected:
        /<task>/<optimizer>/episode_<i>/qpos_traj   (T x nq)
        /<task>/<optimizer>/episode_<i>/mocap_pos_traj (T x num_mocap x 3)  (optional)
        /<task>/<optimizer>/episode_<i>/mocap_quat_traj (T x num_mocap x 4) (optional)
    """

    PLAY_LABEL = "Start playback"
    PAUSE_LABEL = "Pause playback"

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the visualizer with the given HDF5 file path."""
        self.file_path = Path(file_path)
        self.server = ViserServer()
        self.available_tasks = get_registered_tasks()

        # Playback state
        self.running = False
        self.curr_episode = 0
        self.curr_episode_length = 0

        # MuJoCo artifacts
        self.task = None
        self.spec = None
        self.model = None
        self.data = None
        self.dt = 1 / 60.0  # sensible default; overwritten after task instantiation
        self.viser_mj_model: ViserMjModel | None = None

        # HDF5 handle (closed in run())
        self.h5: h5py.File | None = None

        # GUI controls (initialized in _build_gui)
        self.pause_button = None
        self.task_dropdown = None
        self.optimizer_dropdown = None
        self.episode_slider = None
        self.time_slider = None

    # ############ #
    # HDF5 Helpers #
    # ############ #

    def _current_group(self) -> h5py.Group:
        """Returns the current HDF5 group based on the selected task and optimizer."""
        assert self.h5 is not None
        task_name = self.task_dropdown.value
        opt_name = self.optimizer_dropdown.value
        return self.h5[task_name][opt_name]

    def _episode_key(self, episode_idx: int) -> str:
        """Returns the HDF5 key for a specific episode."""
        return f"episode_{int(episode_idx)}"

    def _qpos_traj(self, episode_idx: int) -> h5py.Dataset:
        """Returns the qpos trajectory dataset for a specific episode."""
        grp = self._current_group()
        ep_key = self._episode_key(episode_idx)
        if ep_key not in grp:
            raise KeyError(f"Episode '{ep_key}' not found under {grp.name}")
        if "qpos_traj" not in grp[ep_key]:
            raise KeyError(f"'qpos_traj' not found at {grp.name}/{ep_key}")
        return grp[ep_key]["qpos_traj"]

    def _num_episodes(self) -> int:
        """Returns the number of episodes under the current task/optimizer."""
        grp = self._current_group()
        # Count groups named like "episode_*"
        return sum(1 for k in grp.keys() if k.startswith("episode_"))

    def _mocap_pos_traj(self, episode_idx: int) -> h5py.Dataset | None:
        """(T x num_mocap x 3) or None."""
        grp = self._current_group()
        ep_key = self._episode_key(episode_idx)
        if ep_key not in grp:
            raise KeyError(f"Episode '{ep_key}' not found under {grp.name}")
        return grp[ep_key].get("mocap_pos_traj", None)

    def _mocap_quat_traj(self, episode_idx: int) -> h5py.Dataset | None:
        """(T x num_mocap x 4) or None. Assumed (qw, qx, qy, qz)."""
        grp = self._current_group()
        ep_key = self._episode_key(episode_idx)
        if ep_key not in grp:
            raise KeyError(f"Episode '{ep_key}' not found under {grp.name}")
        return grp[ep_key].get("mocap_quat_traj", None)

    def _episode_length(self, episode_idx: int) -> int:
        """Length considering qpos and whichever mocap streams exist (min across present)."""
        T = int(self._qpos_traj(episode_idx).shape[0])

        pos_ds = self._mocap_pos_traj(episode_idx)
        if pos_ds is not None:
            T = min(T, int(pos_ds.shape[0]))

        quat_ds = self._mocap_quat_traj(episode_idx)
        if quat_ds is not None:
            T = min(T, int(quat_ds.shape[0]))

        return T

    # ############## #
    # MuJoCo Helpers #
    # ############## #

    def _instantiate_task(self, task_name: str) -> None:
        """Create a fresh task/spec/model/data for a given task name."""
        self.task = self.available_tasks[task_name][0]()
        self.spec = self.task.spec
        self.model = self.spec.compile()
        self.data = self.task.data
        assert self.h5 is not None
        self.dt = self.h5.attrs["viz_dt"]
        self.viser_mj_model = ViserMjModel(self.server, self.spec)

    def _clamp_episode_to_current_group(self) -> int:
        """Clamp current (UI) episode value to valid range for current task/optimizer."""
        n = self._num_episodes()
        new_max = max(0, n - 1)
        desired = int(self.episode_slider.value) if self.episode_slider is not None else 0
        clamped = min(desired, new_max)
        if self.episode_slider is not None:
            self.episode_slider.max = new_max
            self.episode_slider.value = float(clamped)
        self.curr_episode = clamped
        return clamped

    def _reset_or_clamp_time_slider(self) -> None:
        """Ensure time slider fits current episode length and starts at 0."""
        self.curr_episode_length = int(self._qpos_traj(self.curr_episode).shape[0])
        if self.time_slider is not None:
            self.time_slider.max = max(0, self.curr_episode_length - 1)
            self.time_slider.value = 0.0

    def _apply_mocap_pos(self, mocap_pos_t: Any | None) -> None:
        """Write mocap positions if available; shape (num_mocap, 3)."""
        if mocap_pos_t is None or self.data is None:
            return
        if self.data.mocap_pos is None or self.data.mocap_pos.shape[0] == 0:
            return
        arr = np.asarray(mocap_pos_t)
        nmocap = min(self.data.mocap_pos.shape[0], arr.shape[0])
        self.data.mocap_pos[:nmocap, :] = arr[:nmocap, :3]

    def _apply_mocap_quat(self, mocap_quat_t: Any | None) -> None:
        """Write mocap quaternions if available; shape (num_mocap, 4) in (qw,qx,qy,qz)."""
        if mocap_quat_t is None or self.data is None:
            return
        if self.data.mocap_quat is None or self.data.mocap_quat.shape[0] == 0:
            return
        arr = np.asarray(mocap_quat_t)
        nmocap = min(self.data.mocap_quat.shape[0], arr.shape[0])
        # If your file stores xyzw, swap to qwxyz here before assignment.
        self.data.mocap_quat[:nmocap, :] = arr[:nmocap, :4]

    def _apply_and_render(self, qpos: Any, mocap_pos: Any | None = None, mocap_quat: Any | None = None) -> None:
        """Apply qpos and optional mocap pos/quat, then push to Viser."""
        self.data.qpos[:] = qpos
        self._apply_mocap_pos(mocap_pos)
        self._apply_mocap_quat(mocap_quat)
        mj_forward(self.model, self.data)
        self.viser_mj_model.set_data(self.data)

    def _reset_or_clamp_time_slider(self) -> None:
        """Ensure time slider fits current episode length and starts at 0."""
        self.curr_episode_length = self._episode_length(self.curr_episode)
        if self.time_slider is not None:
            self.time_slider.max = max(0, self.curr_episode_length - 1)
            self.time_slider.value = 0.0

    def _refresh_episode_info(self) -> None:
        """Update episode length and reset the time slider to 0."""
        self.curr_episode_length = self._episode_length(self.curr_episode)
        if self.time_slider is not None:
            self.time_slider.value = 0.0
            self.time_slider.max = max(0, self.curr_episode_length - 1)

    # ### #
    # GUI #
    # ### #

    def _build_gui(self) -> None:
        # Pause / play button
        self.pause_button = self.server.gui.add_button(self.PLAY_LABEL)

        @self.pause_button.on_click
        def _(_: GuiEvent) -> None:
            self._toggle_running()

        # Determine initial HDF5-derived choices
        assert self.h5 is not None
        data_tasks = list(self.h5.keys())
        if not data_tasks:
            raise ValueError("No tasks found in HDF5 file.")
        initial_task = data_tasks[0]
        data_optimizers = list(self.h5[initial_task].keys())
        if not data_optimizers:
            raise ValueError(f"No optimizers under task '{initial_task}'.")

        # Task dropdown
        self.task_dropdown = self.server.gui.add_dropdown("task", data_tasks, initial_value=initial_task)

        @self.task_dropdown.on_update
        def _(_: GuiEvent) -> None:
            # Re-instantiate for the new task
            task_name = self.task_dropdown.value
            self._instantiate_task(task_name)

            # update optimizer choices for the selected task
            new_opts = list(self.h5[task_name].keys())
            try:
                self.optimizer_dropdown.choices = new_opts
            except Exception:
                pass
            if self.optimizer_dropdown.value not in new_opts:
                self.optimizer_dropdown.value = new_opts[0]

            # Keep the user's episode index if possible (clamp to new range)
            self._clamp_episode_to_current_group()
            self._reset_or_clamp_time_slider()

            # Apply frame at t=0 for the selected episode
            qpos0 = self._qpos_traj(self.curr_episode)[0]
            pos_ds = self._mocap_pos_traj(self.curr_episode)
            quat_ds = self._mocap_quat_traj(self.curr_episode)
            mocap_pos0 = pos_ds[0] if pos_ds is not None and len(pos_ds) > 0 else None
            mocap_quat0 = quat_ds[0] if quat_ds is not None and len(quat_ds) > 0 else None
            self._apply_and_render(qpos0, mocap_pos0, mocap_quat0)

        # Optimizer dropdown
        self.optimizer_dropdown = self.server.gui.add_dropdown(
            "optimizer", data_optimizers, initial_value=data_optimizers[0]
        )

        @self.optimizer_dropdown.on_update
        def _(_: GuiEvent) -> None:
            self._clamp_episode_to_current_group()
            self._reset_or_clamp_time_slider()
            qpos0 = self._qpos_traj(self.curr_episode)[0]
            pos_ds = self._mocap_pos_traj(self.curr_episode)
            quat_ds = self._mocap_quat_traj(self.curr_episode)
            mocap_pos0 = pos_ds[0] if pos_ds is not None and len(pos_ds) > 0 else None
            mocap_quat0 = quat_ds[0] if quat_ds is not None and len(quat_ds) > 0 else None
            self._apply_and_render(qpos0, mocap_pos0, mocap_quat0)

        # Instantiate model/vis for initial task
        self._instantiate_task(initial_task)

        # Episode slider
        curr_num_episodes = self._num_episodes()
        self.episode_slider = self.server.gui.add_slider(
            "episode", 0, max(0, curr_num_episodes - 1), 1, initial_value=0
        )

        @self.episode_slider.on_update
        def _(_: GuiEvent) -> None:
            self.curr_episode = int(self.episode_slider.value)
            self._reset_or_clamp_time_slider()
            qpos0 = self._qpos_traj(self.curr_episode)[0]
            pos_ds = self._mocap_pos_traj(self.curr_episode)
            quat_ds = self._mocap_quat_traj(self.curr_episode)
            mocap_pos0 = pos_ds[0] if pos_ds is not None and len(pos_ds) > 0 else None
            mocap_quat0 = quat_ds[0] if quat_ds is not None and len(quat_ds) > 0 else None
            self._apply_and_render(qpos0, mocap_pos0, mocap_quat0)

        # Time slider (frame within episode)
        self._reset_or_clamp_time_slider()
        self.time_slider = self.server.gui.add_slider(
            "time", 0, max(0, self.curr_episode_length - 1), 1, initial_value=0
        )

        @self.time_slider.on_update
        def _(_: GuiEvent) -> None:
            t = int(self.time_slider.value)
            qpos_t = self._qpos_traj(self.curr_episode)[t]
            pos_ds = self._mocap_pos_traj(self.curr_episode)
            quat_ds = self._mocap_quat_traj(self.curr_episode)
            mocap_pos_t = pos_ds[t] if pos_ds is not None and t < len(pos_ds) else None
            mocap_quat_t = quat_ds[t] if quat_ds is not None and t < len(quat_ds) else None
            self._apply_and_render(qpos_t, mocap_pos_t, mocap_quat_t)

        # Set initial frame
        qpos0 = self._qpos_traj(self.curr_episode)[0]
        pos_ds = self._mocap_pos_traj(self.curr_episode)
        quat_ds = self._mocap_quat_traj(self.curr_episode)
        mocap_pos0 = pos_ds[0] if pos_ds is not None and len(pos_ds) > 0 else None
        mocap_quat0 = quat_ds[0] if quat_ds is not None and len(quat_ds) > 0 else None
        self._apply_and_render(qpos0, mocap_pos0, mocap_quat0)

    # ################# #
    # Playback Controls #
    # ################# #

    def _toggle_running(self) -> None:
        self.running = not self.running
        self.pause_button.label = self.PAUSE_LABEL if self.running else self.PLAY_LABEL

    # ######### #
    # Main Loop #
    # ######### #

    def run(self) -> None:
        """Open HDF5, build UI, and enter the playback loop (blocking)."""
        with h5py.File(self.file_path, "r") as h5:
            self.h5 = h5
            self._build_gui()

            try:
                # Ensure the slider is at t=0 before starting loop
                self.time_slider.value = 0.0
                while True:
                    start_time = time.time()
                    if self.running:
                        # advance a frame; wrap and auto-pause at the end like original
                        t = int(self.time_slider.value)
                        t = min(self.curr_episode_length - 1, t + 1)
                        self.time_slider.value = float(t)
                        if t == self.curr_episode_length - 1:
                            self.time_slider.value = 0.0
                            self._toggle_running()  # pause at end
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    time.sleep(self.dt - elapsed_time if self.dt > elapsed_time else 0.0)
            except KeyboardInterrupt:
                print("Closing viser...")


def visualize_benchmark_runs(file_path: str | Path) -> None:
    """Visualize benchmark runs from an HDF5 file (public entry point)."""
    BenchmarkRunVisualizer(file_path).run()
