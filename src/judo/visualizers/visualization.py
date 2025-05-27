# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
import time
from abc import abstractmethod
from typing import Protocol

from mujoco import MjData
from viser import ViserServer

from judo.tasks.task import Task
from judo.viser_app.io import ControlBufferKeys, IOContext, StateBufferKeys
from judo.visualizers.model import ViserMjModel

# TODO(pculbertson): make this configurable.
VISUALIZATION_FREQ = 60


class MjVisualization:
    """Container for updating the viser model with current state info."""

    def __init__(
        self, task: Task, server: ViserServer, context: IOContext, handles: dict
    ):
        self.data = MjData(task.model)
        self.traces = None
        self.server = server
        self.text_handles = handles
        self.context = context

        # Create Viser object for Mujoco model.
        self.viser_mjmodel = ViserMjModel(self.server, task.model)

    def _update_stats(self) -> None:
        """Adds text with the statistics of performance in the GUI"""
        with self.context.profiling_lock:
            for key, value in self.context.profiling_stats_dict.items():
                if value is not None:
                    self.text_handles[key].value = str(value)

    def update_visualization(self) -> None:
        """Update model state and traces.

        Update model state by reading FK results from state buffer.
        Update traces by reading control buffer.
        """
        start_time = time.time()
        if not self.context.task_updated_event.is_set():
            with self.context.state_lock:
                self.data.xpos[:] = self.context.state_buffer.get(
                    StateBufferKeys.xpos, None
                )
                self.data.xquat[:] = self.context.state_buffer.get(
                    StateBufferKeys.xquat, None
                )

            with self.context.control_lock:
                self.traces = self.context.control_buffer.get(
                    ControlBufferKeys.traces, None
                )
                all_traces_rollout_size = self.context.control_buffer.get(
                    ControlBufferKeys.all_traces_rollout_size, None
                )
            self.viser_mjmodel.set_data(self.data)
            # Figure out which traces are good and which aren't
            self.viser_mjmodel.set_traces(self.traces, all_traces_rollout_size)
        self._update_stats()
        time_elapsed = time.time() - start_time
        time.sleep(max(1 / VISUALIZATION_FREQ - time_elapsed, 0))

    def remove(self) -> None:
        """Removes all model geometries from the GUI."""
        self.viser_mjmodel.remove()
        self.server.flush()


class Visualization(Protocol):
    """Container for updating the viser model with current state info."""

    @abstractmethod
    def _update_stats(self) -> None:
        """Adds text with the statistics of performance in the GUI."""

    @abstractmethod
    def update_visualization(self) -> None:
        """Update model state and traces."""

    @abstractmethod
    def remove(self) -> None:
        """Removes all model geometries from the GUI."""
