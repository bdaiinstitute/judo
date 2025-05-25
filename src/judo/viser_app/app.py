# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import json
import logging
import multiprocessing
import time
import traceback
from copy import copy
from dataclasses import asdict
from datetime import datetime
from multiprocessing.context import Process
from pathlib import Path
from typing import Callable, Optional, Tuple, Type, no_type_check

import numpy as np
import tyro
from dacite import from_dict
from viser import GuiEvent, Icon, MeshHandle, ViserServer

from judo.controllers import get_registered_controllers
from judo.controllers.controller import (
    Controller,
    ControllerConfig,
)
from judo.tasks import get_registered_tasks
from judo.tasks.task import Task, TaskConfig
from judo.viser_app.gui import create_gui_elements
from judo.viser_app.io import (
    ControlBufferKeys,
    IOContext,
    SimulationIOContext,
    StateBufferKeys,
)
from judo.viser_app.json_serializer import ConfigEncoder
from judo.viser_app.path_utils import PACKAGE_ROOT
from judo.viser_app.profiler import ViserProfiler, ViserProfilerConfig
from judo.visualizers.visualization import MjVisualization


class SimulationProcess(Process):
    """Container for the simulation thread in the viser app."""

    def __init__(self, task: Task, context: SimulationIOContext):
        self.task = task
        self.context = context

        # Write default states to buffer to ensure it's not empty.
        self.write_states()
        super().__init__(target=self.simulation_loop)
        # TODO(pculbertson): implement sim parameter changes.

    def simulation_loop(self) -> None:
        """Main simulation loop for SimulationProcess."""
        while True:
            if self.context.simulation_reset_event.is_set():
                self.reset()
                self.context.simulation_reset_event.clear()

            if self.context.simulation_running.is_set():
                start_time = time.time()
                # Read current control.
                if self.context.controller_running.is_set():
                    with self.context.control_lock:
                        control = self.context.control_buffer[ControlBufferKeys.spline]
                else:
                    control = None

                # Take mujoco physics step and write to buffer.
                self.task.sim_step(control)
                self.write_states()

                # Force loop to run at 1x realtime (or as close as possible).
                time_elapsed = time.time() - start_time
                time.sleep(max(self.task.dt - time_elapsed, 0))

    def write_states(self) -> None:
        """Write current sim states to context."""
        with self.context.state_lock:
            self.context.state_buffer[StateBufferKeys.qpos] = self.task.data.qpos.copy()
            self.context.state_buffer[StateBufferKeys.qvel] = self.task.data.qvel.copy()
            self.context.state_buffer[StateBufferKeys.time] = self.task.data.time
            self.context.state_buffer[StateBufferKeys.xpos] = self.task.data.xpos.copy()
            self.context.state_buffer[StateBufferKeys.xquat] = (
                self.task.data.xquat.copy()
            )
            self.context.state_buffer[StateBufferKeys.additional_info] = copy(
                self.task.additional_task_info
            )

    def reset(self) -> None:
        """Resets the simulation to a new initial state."""
        self.task.reset()


class ControlProcess(multiprocessing.Process):
    """Process that executes a controller at a fixed rate."""

    def __init__(
        self,
        controller: Controller,
        server: ViserServer,
        context: IOContext,
        profiler: Optional[ViserProfiler] = None,
    ):
        super().__init__()
        self.controller = controller
        self.server = server
        self.context = context
        self.exception_queue = self.context.manager.Queue()
        self.profiler = profiler

        # Overwrite context with configs for current controller.
        with self.context.control_config_lock:
            self.context.control_config_dict = self.context.manager.dict(
                asdict(self.controller.config)
            )

        with self.context.reward_config_lock:
            self.context.reward_config_dict = self.context.manager.dict(
                asdict(self.controller.reward_config)
            )

        # Create gui elements for controller/reward parameters.
        control_folder = self.server.gui.add_folder("Controller parameters")
        with control_folder:
            self.control_config_gui_elements = create_gui_elements(
                self.server,
                self.controller.config,
                self.context.control_config_dict,
                self.context.control_config_updated_event,
                self.context.control_config_lock,
            )
        self.control_config_gui_elements.append(control_folder)

        reward_folder = self.server.gui.add_folder("Reward parameters")
        with reward_folder:
            self.reward_config_gui_elements = create_gui_elements(
                self.server,
                self.controller.reward_config,
                self.context.reward_config_dict,
                self.context.reward_config_updated_event,
                self.context.reward_config_lock,
            )
        self.reward_config_gui_elements.append(reward_folder)

        self.write_controls()
        if self.profiler is not None:
            self.set_profiler_recording(self.profiler.recording)
        else:
            self.control_step = self._control_step

    def run(self) -> None:
        """Main control flow for the ControllerProcess.

        When running, the Process will try to run the control loop. If an exception is encountered,
        the Process stores the exception in a Queue, prints the exception, and removes its GUI elements.
        """
        try:
            self.control_loop()
        except Exception as e:
            self.exception_queue.put(e)
            traceback.print_exc()
        finally:
            self.remove()

    def remove(self) -> None:
        """Helper function to clean up GUI elements for this control instance."""
        for element in (
            self.control_config_gui_elements + self.reward_config_gui_elements
        ):
            # We don't need to remove the mesh handles; that's handled on the visualization.reset()
            if not isinstance(element, MeshHandle):
                element.remove()

    def terminate(self) -> None:
        """Cleans up GUI elements on process termination."""
        self.remove()
        super().terminate()

    def control_loop(self) -> Exception | None:
        """Main outer control loop. Run controller at fixed control frequency."""
        while True:
            # Handle event where profiler has cycled on/off.
            if self.context.profiler_updated_event.is_set():
                if self.profiler is not None:
                    self.set_profiler_recording(not self.profiler.recording)
                else:
                    logging.warning("No profiler attached to control thread!")
                self.context.profiler_updated_event.clear()

            # Run controller.
            if self.context.controller_running.is_set():
                start_time = time.time()
                self.control_step()

                # Force controller to run at fixed rate specified by control_freq.
                time.sleep(
                    max(
                        0,
                        (1 / self.controller.config.control_freq)
                        - (time.time() - start_time),
                    )
                )

    def set_profiler_recording(self, recording: bool = False) -> None:
        """Turns on/off profiler recording and updates inner control loop."""
        if self.profiler is None:
            return
        self.profiler.recording = recording
        self.control_step = (
            self.profiler.benchmark_wrapper(self._control_step)
            if self.profiler.recording
            else self._control_step
        )

    def _control_step(self) -> None:
        """Inner control step. Reads state, updates config, updates controls, writes controls."""
        # Read current state.
        with self.context.state_lock:
            qpos = self.context.state_buffer[StateBufferKeys.qpos]
            qvel = self.context.state_buffer[StateBufferKeys.qvel]
            curr_time = self.context.state_buffer[StateBufferKeys.time]
            additional_info = self.context.state_buffer[StateBufferKeys.additional_info]

        # Update config dictionaries.
        if self.context.control_config_updated_event.is_set():
            with self.context.control_config_lock:
                self.controller.config = from_dict(
                    type(self.controller.config), self.context.control_config_dict
                )
            self.context.control_config_updated_event.clear()

        if self.context.reward_config_updated_event.is_set():
            with self.context.reward_config_lock:
                self.controller.reward_config = from_dict(
                    type(self.controller.reward_config), self.context.reward_config_dict
                )
            self.context.reward_config_updated_event.clear()

        # Run controller update and write back to state buffer.
        self.controller.update_action(
            np.concatenate([qpos, qvel]), curr_time, additional_info
        )

        # Write the profiler information back to the context
        if self.profiler is not None:
            with self.context.profiling_lock:
                for key, value in self.profiler.agg_stats.items():
                    self.context.profiling_stats_dict[key] = value

        self.write_controls()
        self.write_traces()

    def write_controls(self) -> None:
        """Write control result out to buffer."""
        with self.context.control_lock:
            self.context.control_buffer[ControlBufferKeys.spline] = (
                self.controller.spline
            )

    def write_traces(self) -> None:
        """Write traces out to buffer."""
        with self.context.control_lock:
            self.context.control_buffer[ControlBufferKeys.traces] = (
                self.controller.traces
            )
            self.context.control_buffer[ControlBufferKeys.all_traces_rollout_size] = (
                self.controller.all_traces_rollout_size
            )


class ViserApp:
    """Main class for running the viser app."""

    def __init__(
        self,
        init_controller: str = "predictive_sampling",
        init_task: str = "cartpole",
        port: int = 8008,
        benchmark_dir: Optional[Path] = PACKAGE_ROOT / "log/controller",
    ):
        # Create viser server for app frontend.
        self.server = ViserServer(port=port)
        self.server.flush()

        # Create datetime string and benchmark dir for profiling.
        now = datetime.now()
        self.dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.benchmark_dir = benchmark_dir

        # Load all registered controllers/tasks.
        self.available_controllers = get_registered_controllers()
        self.available_tasks = get_registered_tasks()

        assert (
            init_controller in self.available_controllers
        ), f"Controller {init_controller} is not registered!"
        assert init_task in self.available_tasks, f"Task {init_task} is not registered!"

        self.current_controller = init_controller
        self.current_task = init_task

        # Add GUI elements for top-level control of app.
        # Create button for turning on / off controller.
        self.start_controller_label = "Start controller"
        self.stop_controller_label = "Stop controller"

        self.control_cycle_button = self.server.gui.add_button(
            self.stop_controller_label
        )
        self.control_cycle_button.on_click(self.control_cycle_callback)

        # Create button for resetting the physics simulation.
        self.simulation_reset_button = self.server.gui.add_button("Reset simulation")
        self.simulation_reset_button.on_click(self.simulation_reset_callback)

        # Create button for pausing / starting simulation.
        self.simulation_cycle_button = self.server.gui.add_button("Pause simulation")
        self.simulation_cycle_button.on_click(self.simulation_cycle_callback)

        # Create button for downloading the current task / controller config.
        self.config_download_button = self.server.gui.add_button(
            "Download configuration", icon=Icon.BOOK_DOWNLOAD
        )
        self.config_download_button.on_click(self.config_download_callback)

        # Create buffers for app + start sim.
        self.setup_context()

        # Add dropdown to enable switching between available controllers.
        self.control_picker = self.server.gui.add_dropdown(
            "Controller",
            list(self.available_controllers.keys()),
            self.current_controller,
        )
        self.control_picker.on_update(self.control_selection_callback)

        # Add dropdown for switching between available tasks.
        self.task_picker = self.server.gui.add_dropdown(
            "Task", list(self.available_tasks.keys()), self.current_task
        )
        self.task_picker.on_update(self.task_selection_callback)

        _, controller_config = self._get_control_class_config()

        self.enable_profiling_label = "Enable profiling"
        self.disable_profiling_label = "Disable profiling"

        self.enable_profiler_button = self.server.gui.add_button(
            self.enable_profiling_label
        )
        self.enable_profiler_button.on_click(self.enable_profiling_callback)

        self.text_handles = {}
        viser_profile_config = ViserProfilerConfig()
        for tracked_key in viser_profile_config.tracked_fields.keys():
            self.text_handles[tracked_key] = self.server.gui.add_text(
                label=str(tracked_key), initial_value="N/A", disabled=True
            )

        self.setup_task()

        try:
            while True:
                controller_exception = None
                # Spin infinitely.
                with self.context.visualization_lock:
                    self.visualization.update_visualization()

                # Exception handling for the controller.
                if not self.controller.exception_queue.empty():
                    controller_exception = self.controller.exception_queue.get(
                        block=False
                    )

                # If exception encountered, create a viser notification to alert the user, restart.
                if controller_exception:
                    for client in self.server.get_clients().values():
                        client.add_notification(
                            title="Oops! 😵‍💫",
                            body=f"Controller has crashed with exception: '{controller_exception}'",
                            auto_close=10000,
                            color="red",
                        )
                    self.controller.remove()
                    self.enable_profiler_button.label = self.enable_profiling_label
                    self.setup_controller(
                        self.controller.controller.config,
                        self.controller.controller.reward_config,
                    )

        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            if self.physics.is_alive():
                self.physics.terminate()
            if self.controller.is_alive():
                self.controller.terminate()
            self.server.stop()

    def flip_profile_button_label(self) -> None:
        """Flips the button label for the disable/enable profile"""
        if self.enable_profiler_button.label == self.enable_profiling_label:
            self.enable_profiler_button.label = self.disable_profiling_label
        else:
            self.enable_profiler_button.label = self.enable_profiling_label

    @no_type_check
    @staticmethod
    def _unset_sim_and_run(func: Callable) -> Callable:
        """Function decorator that unsets the running events to make changes safer. Assumes private visibility"""

        def wrapper(*args: str, **kwargs: int) -> Callable:
            args[0].context.simulation_running.clear()
            value = func(*args, **kwargs)
            args[0].context.simulation_running.set()
            return value

        return wrapper

    @no_type_check
    @staticmethod
    def _unset_sim_control_and_run(func: Callable) -> Callable:
        """Function decorator that unsets the running events to make changes safer. Assumes private visibility"""

        def wrapper(*args: str, **kwargs: int) -> Callable:
            # Wanted to reuse the code from above, but Python doesn't like calling two static methods in each other
            initial_control_state = args[0].context.controller_running.is_set()
            args[0].context.controller_running.clear()
            args[0].context.simulation_running.clear()
            value = func(*args, **kwargs)
            args[0].context.simulation_running.set()
            # If the controller was running, we reset it to run. Otherwise we leave it
            if initial_control_state:
                args[0].context.controller_running.set()
            return value

        return wrapper

    def setup_task(self) -> None:
        """Task setup for task selected currently in GUI; spawns simulation, control, and visualization processes."""
        # TODO(pculbertson): generalize to include hardware.
        # Instantiate current task, reset to random state.
        task_class, _ = self.available_tasks[self.current_task]
        task = task_class()
        task.reset()

        # Spawn simulation and visualization processes.
        self.setup_physics(task)
        self.setup_visualization()

        # Run controller setup for new task.
        self.setup_controller()

        # Start simulation
        self.physics.start()
        self.context.simulation_running.set()

    def setup_visualization(self) -> None:
        """Creates the visualization stack"""
        with self.context.visualization_lock:
            self.visualization: MjVisualization = MjVisualization(
                self.physics.task, self.server, self.context, self.text_handles
            )

    def setup_context(self) -> None:
        """Set the context field and set the initial controller state.

        This method can be overwritten for hardware deployment.
        """
        self.context = SimulationIOContext(multiprocessing.Manager())
        # Set controller to begin running by default.
        self.context.controller_running.set()

    def setup_physics(self, task: Task) -> None:
        """Set the physics field.

        This method can be overwritten for hardware deployment.
        """
        self.physics: SimulationProcess = SimulationProcess(task, self.context)

    def _get_control_class_config(
        self,
    ) -> Tuple[Type[Controller], Type[ControllerConfig]]:
        """Gets the controller class type and configuration"""
        return self.available_controllers[self.current_controller]

    def setup_controller(
        self,
        control_config: ControllerConfig | None = None,
        task_config: TaskConfig | None = None,
    ) -> None:
        """Spawns controller process for controller method currently selected in GUI."""
        # Instantiate current controller.
        _, task_config_class = self.available_tasks[self.current_task]
        control_class, control_config_class = self.available_controllers[
            self.current_controller
        ]

        if control_config is not None:
            assert (
                task_config is not None
            ), "Must pass both task and control config defaults together!"
            assert isinstance(control_config, control_config_class)
            assert isinstance(task_config, task_config_class)
        else:
            control_config = control_config_class()
            task_config = task_config_class()

        self.profiler = ViserProfiler(
            logfile=(
                None
                if self.benchmark_dir is None
                else str(
                    self.benchmark_dir / f"{self.current_task}_{self.dt_string}.txt"
                )
            ),
            record=False,
        )
        self.enable_profiler_button.label = self.enable_profiling_label

        # Spawn controller process for current control method / task.
        self.controller: ControlProcess = ControlProcess(
            control_class(self.physics.task, control_config, task_config),
            self.server,
            self.context,
            self.profiler,
        )
        self.controller.start()

        if self.context.controller_running.is_set():
            self.control_cycle_button.label = self.stop_controller_label
        else:
            self.control_cycle_button.label = self.stop_controller_label

    @_unset_sim_and_run
    def control_cycle_callback(self, _: GuiEvent) -> None:
        """Logic for turning on/off the controller via GUI button."""
        if self.context.controller_running.is_set():
            self.context.controller_running.clear()
            self.control_cycle_button.label = self.start_controller_label
        else:
            self.context.controller_running.set()
            self.control_cycle_button.label = self.stop_controller_label

    def simulation_cycle_callback(self, _: GuiEvent) -> None:
        """Logic for turning on/off the simulation via GUI button."""
        if self.context.simulation_running.is_set():
            self.context.simulation_running.clear()
            self.simulation_cycle_button.label = "Start simulation"
        else:
            self.context.simulation_running.set()
            self.simulation_cycle_button.label = "Pause simulation"

    @_unset_sim_control_and_run
    def simulation_reset_callback(self, _: GuiEvent) -> None:
        """Resets the simulation via GUI button."""
        self.context.simulation_reset_event.set()
        if self.controller.is_alive():
            self.controller.terminate()
        previous_config = from_dict(
            type(self.controller.controller.config), self.context.control_config_dict
        )
        self.setup_controller(previous_config, self.controller.controller.reward_config)

    @_unset_sim_control_and_run
    def control_selection_callback(self, event: GuiEvent) -> None:
        """Callback for changing controllers. Does not reset sim/vis, but sets up new controller."""
        if self.controller.is_alive():
            self.controller.terminate()
        else:
            self.controller.remove()
        self.enable_profiler_button.label = self.enable_profiling_label

        self.current_controller = event.target.value
        self.setup_controller()

    @_unset_sim_control_and_run
    def task_selection_callback(self, event: GuiEvent) -> None:
        """Callback for changing tasks. Changes target task and runs a clean task setup."""
        # Sets the task update event to prevent the visualization from running
        self.context.task_updated_event.set()
        if self.physics.is_alive():
            self.physics.terminate()
        if self.controller.is_alive():
            self.controller.terminate()
        with self.context.visualization_lock:
            self.visualization.remove()

        self.current_task = event.target.value

        self.setup_task()
        # Clears the task update event
        self.context.task_updated_event.clear()

    def enable_profiling_callback(self, event: GuiEvent) -> None:
        """Callback to enabling/disabling the performance profiler. Initializes the function in the controller"""
        # Bottom needs to be backwards because the profiler will flip. The goal is to keep the actual not here
        self.context.profiler_updated_event.set()
        self.flip_profile_button_label()

    def config_download_callback(self, event: GuiEvent) -> None:
        """Callback for downloading the current controller/task configs."""
        with self.context.control_config_lock:
            controller_config = dict(self.context.control_config_dict)
        with self.context.reward_config_lock:
            task_config = dict(self.context.reward_config_dict)

        full_config = {
            "controller": self.current_controller,
            "controller_config": controller_config,
            "task": self.current_task,
            "task_config": task_config,
        }

        json_str = json.dumps(full_config, cls=ConfigEncoder)
        json_bytes = json_str.encode("utf-8")

        self.server.send_file_download("config.json", json_bytes)


def main() -> None:
    """Helper main method to make app installable as script via pyproject.toml."""
    tyro.cli(ViserApp)


if __name__ == "__main__":
    """Entry point for Viser app."""
    main()
