# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import threading
import warnings

import mujoco
import numpy as np
import pyarrow as pa
import viser
from dora_utils.dataclasses import from_arrow, to_arrow
from dora_utils.node import DoraNode, on_event
from omegaconf import DictConfig
from PIL import Image
from viser import GuiFolderHandle, GuiImageHandle, GuiInputHandle, IcosphereHandle, MeshHandle

from judo import PACKAGE_ROOT
from judo.app.structs import MujocoState
from judo.app.utils import register_optimizers_from_cfg, register_tasks_from_cfg
from judo.config import set_config_overrides
from judo.controller import ControllerConfig
from judo.gui import create_gui_elements
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks
from judo.visualizers.model import ViserMjModel

ElementType = GuiImageHandle | GuiInputHandle | GuiFolderHandle | MeshHandle | IcosphereHandle


class VisualizationNode(DoraNode):
    """The visualization node."""

    def __init__(
        self,
        node_id: str = "visualization",
        max_workers: int | None = None,
        init_task: str = "cylinder_push",
        init_optimizer: str = "cem",
        task_registration_cfg: DictConfig | None = None,
        optimizer_registration_cfg: DictConfig | None = None,
        controller_override_cfg: DictConfig | None = None,
        optimizer_override_cfg: DictConfig | None = None,
        sim_pause_button: bool = True,
        geom_exclude_substring: str = "collision",
    ) -> None:
        """Initialize the visualization node."""
        super().__init__(node_id=node_id, max_workers=max_workers)

        # handling custom task and optimizer registration
        if task_registration_cfg is not None:
            register_tasks_from_cfg(task_registration_cfg)
        if optimizer_registration_cfg is not None:
            register_optimizers_from_cfg(optimizer_registration_cfg)

        # starting the server
        self.server = viser.ViserServer()
        self.available_tasks = get_registered_tasks()
        self.available_optimizers = get_registered_optimizers()
        self.geom_exclude_substring = geom_exclude_substring

        self.task_name = ""
        self.optimizer_name = ""

        self.task_lock = threading.Lock()
        self.task_updated = threading.Event()

        self.sim_pause_button = sim_pause_button
        if sim_pause_button:
            self.sim_pause_lock = threading.Lock()
            self.sim_pause_updated = threading.Event()

        # handling task-specific config overrides
        if controller_override_cfg is not None:
            self.register_controller_config_overrides(controller_override_cfg)
        if optimizer_override_cfg is not None:
            self.register_optimizer_config_overrides(optimizer_override_cfg)

        self.set_task(init_task, init_optimizer)

    def register_controller_config_overrides(self, controller_override_cfg: DictConfig) -> None:
        """Register task-specific controller config overrides.

        We do this in the Visualization node because this is the "master" node in the dora stack.
        """
        # as opposed to the optimizer, there is only one Controller class, so for each task, we simply need to specify
        # the controller parameter overrides
        cfg_cls = ControllerConfig
        for task_name in controller_override_cfg.keys():
            field_override_values = controller_override_cfg.get(task_name, {})
            set_config_overrides(str(task_name), cfg_cls, field_override_values)

    def register_optimizer_config_overrides(self, optimizer_override_cfg: DictConfig) -> None:
        """Register task-specific optimizer config overrides.

        We do this in the Visualization node because this is the "master" node in the dora stack.
        """
        for task_name in optimizer_override_cfg.keys():  # task name in registry
            for optimizer_name in optimizer_override_cfg[task_name].keys():  # optimizer name in registry
                _, cfg_cls = get_registered_optimizers().get(optimizer_name, (None, None))
                if cfg_cls is None:
                    raise ValueError(f"Optimizer {optimizer_name} not found in the registry!")
                field_override_values = optimizer_override_cfg[task_name].get(optimizer_name, {})
                set_config_overrides(str(task_name), cfg_cls, field_override_values)

    def set_task(self, task: str, optimizer: str) -> None:
        """Helper to initialize task from task name."""
        self.task_name = task
        self.optimizer_name = optimizer

        task_entry = self.available_tasks.get(task)
        if task_entry is None:
            raise ValueError(f"Task {task} not found in the task registry.")

        task_cls, self.task_config_cls = task_entry
        self.task = task_cls()
        self.data = mujoco.MjData(self.task.model)
        self.viser_model = ViserMjModel(
            self.server,
            self.task.spec,
            geom_exclude_substring=self.geom_exclude_substring,
        )

        optimizer_entry = self.available_optimizers.get(optimizer)
        if optimizer_entry is None:
            raise ValueError(f"Optimizer {optimizer} not found in optimizer registry.")
        _, optimizer_config_cls = optimizer_entry

        self.controller_config_lock = threading.Lock()
        self.controller_config_updated = threading.Event()
        self.controller_config_cls = ControllerConfig
        self.controller_config = self.controller_config_cls()
        self.controller_config.set_override(task)

        self.optimizer_lock = threading.Lock()
        self.optimizer_updated = threading.Event()
        self.optimizer_config = optimizer_config_cls()
        self.optimizer_config.set_override(task)
        self.optimizer_config_lock = threading.Lock()
        self.optimizer_config_updated = threading.Event()

        self.task_config = self.task_config_cls()
        self.task_config_lock = threading.Lock()
        self.task_config_updated = threading.Event()

        self.task_reset_updated = threading.Event()

        self.setup_gui()

        # send the configs to the other nodes
        self.controller_config_updated.set()
        self.task_config_updated.set()
        self.optimizer_config_updated.set()

    def setup_gui(self) -> None:
        """Set up the GUI for the visualization node."""
        self.gui_elements = {}

        # add the Judo logo
        logo_path = PACKAGE_ROOT / "app" / "asset" / "viser-logo-light.png"
        logo = self.server.gui.add_image(np.array(Image.open(logo_path)))
        self.gui_elements["logo"] = logo

        # create the dropdown to select the task
        task_dropdown = self.server.gui.add_dropdown(
            "task",
            list(self.available_tasks.keys()),
            initial_value=self.task_name,
        )
        self.gui_elements["task"] = task_dropdown

        # create the dropdown to select the optimizer
        optimizer_dropdown = self.server.gui.add_dropdown(
            "optimizer",
            list(self.available_optimizers.keys()),
            initial_value=self.optimizer_name,
        )
        self.gui_elements["optimizer"] = optimizer_dropdown

        # create a task reset button
        reset_button = self.server.gui.add_button("Reset Task")
        self.gui_elements["reset_button"] = reset_button

        @reset_button.on_click
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the reset button is clicked."""
            self.task_reset_updated.set()

        # create a config reset button
        config_reset_button = self.server.gui.add_button("Reset Configs")
        self.gui_elements["config_reset_button"] = config_reset_button

        @config_reset_button.on_click
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the reset button is clicked."""
            # reset configs to (task) defaults
            self.controller_config.set_override(self.task_name)
            self.optimizer_config.set_override(self.task_name)
            self.task_config = self.task_config_cls()

            # reset gui elements
            for handle in self.gui_elements["controller_params"]:
                self.remove_handles(handle)
            for handle in self.gui_elements["optimizer_params"]:
                self.remove_handles(handle)
            for handle in self.gui_elements["task_params"]:
                self.remove_handles(handle)
            with self.gui_elements["controller_tab"]:
                self.gui_elements["controller_params"] = create_gui_elements(
                    self.server,
                    self.controller_config,
                    self.controller_config_updated,
                    self.controller_config_lock,
                )
            with self.gui_elements["optimizer_tab"]:
                self.gui_elements["optimizer_params"] = create_gui_elements(
                    self.server,
                    self.optimizer_config,
                    self.optimizer_config_updated,
                    self.optimizer_config_lock,
                )
            with self.gui_elements["task_tab"]:
                self.gui_elements["task_params"] = create_gui_elements(
                    self.server,
                    self.task_config,
                    self.task_config_updated,
                    self.task_config_lock,
                )

            # send the configs to the other nodes
            self.controller_config_updated.set()
            self.optimizer_config_updated.set()
            self.task_config_updated.set()

        # create a sim pause button if requested
        if self.sim_pause_button:
            sim_pause_button = self.server.gui.add_button("Pause Simulation")
            self.gui_elements["sim_pause_button"] = sim_pause_button

            @sim_pause_button.on_click
            def _(_: viser.GuiEvent) -> None:
                """Callback for when the pause button is clicked."""
                if sim_pause_button.label == "Pause Simulation":
                    sim_pause_button.label = "Resume Simulation"
                    self.sim_pause_updated.set()
                else:
                    sim_pause_button.label = "Pause Simulation"
                    self.sim_pause_updated.set()

        # create a display for plan time
        self.gui_elements["plan_time_display"] = self.server.gui.add_number(
            "plan time (ms)",
            initial_value=0.0,
            step=0.01,
            disabled=True,
        )

        # main tab containing parameter GUI elements
        tab_group = self.server.gui.add_tab_group()
        self.gui_elements["tab_group"] = tab_group

        controller_tab = tab_group.add_tab("Controller", viser.Icon.DEVICE_GAMEPAD)
        task_tab = tab_group.add_tab("Task", viser.Icon.CHECKLIST)
        optimizer_tab = tab_group.add_tab("Optimizer", viser.Icon.BINARY_TREE_2)

        # create controller GUI elements
        self.gui_elements["controller_tab"] = controller_tab
        with controller_tab:
            self.gui_elements["controller_params"] = create_gui_elements(
                self.server,
                self.controller_config,
                self.controller_config_updated,
                self.controller_config_lock,
            )

        # create the task GUI elements
        self.gui_elements["task_tab"] = task_tab
        with task_tab:
            self.gui_elements["task_params"] = create_gui_elements(
                self.server,
                self.task_config,
                self.task_config_updated,
                self.task_config_lock,
            )

        # create the optimizer GUI elements
        self.gui_elements["optimizer_tab"] = optimizer_tab
        with optimizer_tab:
            # create the rest of the optimizer GUI elements
            self.gui_elements["optimizer_params"] = create_gui_elements(
                self.server,
                self.optimizer_config,
                self.optimizer_config_updated,
                self.optimizer_config_lock,
            )

        # create task and optimizer callbacks after tabs are created
        @optimizer_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the optimizer dropdown is updated."""
            # first, send the name of the new optimizer
            self.optimizer_updated.set()
            self.optimizer_name = optimizer_dropdown.value

            # update config
            optimizer_entry = self.available_optimizers.get(optimizer_dropdown.value)
            assert optimizer_entry is not None
            _optimizer, optimizer_config_cls = optimizer_entry
            self.optimizer_config = optimizer_config_cls()
            self.optimizer_config.set_override(self.task_name)
            self.optimizer_config_updated.set()

            # replace optimizer param gui elements
            for v in self.gui_elements["optimizer_params"]:
                v.remove()
            with optimizer_tab:
                self.gui_elements["optimizer_params"] = create_gui_elements(
                    self.server,
                    self.optimizer_config,
                    self.optimizer_config_updated,
                    self.optimizer_config_lock,
                )

            # because the optimizer will be updated, we need to sent the task parameters back to it so it doesn't start
            # with defaults
            self.task_config_updated.set()

        @task_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            """Callback for when the task dropdown is updated."""
            # first, send the name of the new task
            self.task_name = task_dropdown.value
            self.task_updated.set()

            # replace gui elements
            self._remove_gui_elements()

            # set up the entire visualizer from scratch
            with self.task_lock:
                self.set_task(self.task_name, self.optimizer_name)

    def remove_handles(self, handles: list[ElementType] | ElementType) -> None:
        """Remove GUI handles from the visualization node."""
        if isinstance(handles, ElementType):
            if not handles._impl.removed:
                handles.remove()
        else:
            assert isinstance(handles, list), "handles must be a list or a single handle."
            for handle in handles:
                if isinstance(handle, list):
                    self.remove_handles(handle)
                elif not handle._impl.removed:
                    handle.remove()

    def write_sim_pause(self) -> None:
        """Write the sim pause signal to the GUI."""
        with self.sim_pause_lock:
            self.node.send_output("sim_pause", pa.array([1]))  # dummy value
            self.sim_pause_updated.clear()

    def write_task(self) -> None:
        """Write the task name to the GUI."""
        with self.task_lock:
            self.node.send_output("task", pa.array([self.task_name]))

    def write_task_reset(self) -> None:
        """Write the task reset signal to the GUI."""
        with self.task_lock:
            self.node.send_output("task_reset", pa.array([1]))  # dummy value

    def write_optimizer(self) -> None:
        """Write the optimizer name to the GUI."""
        with self.optimizer_lock:
            self.node.send_output("optimizer", pa.array([self.optimizer_name]))

    def write_controller_config(self) -> None:
        """Write the controller config to the GUI."""
        with self.controller_config_lock:
            self.node.send_output("controller_config", *to_arrow(self.controller_config))

    def write_optimizer_config(self) -> None:
        """Write the optimizer config to the GUI."""
        with self.optimizer_config_lock:
            self.node.send_output("optimizer_config", *to_arrow(self.optimizer_config))

    def write_task_config(self) -> None:
        """Write the task config to the GUI."""
        with self.task_config_lock:
            self.node.send_output("task_config", *to_arrow(self.task_config))

    @on_event("INPUT", "states")
    def update_states(self, event: dict) -> None:
        """Callback to update states on receiving a new state measurement."""
        if self.controller_config.spline_order == "cubic" and self.optimizer_config.num_nodes < 4:
            warnings.warn("Cubic splines require at least 4 nodes. Setting num_nodes=4.", stacklevel=2)
            for e in self.gui_elements["optimizer_params"]:
                if e.label == "num_nodes":
                    e.value = 4
                    break
            self.optimizer_config_updated.set()

        state_msg = from_arrow(event["value"], event["metadata"], MujocoState)
        try:
            with self.task_lock:
                self.data.xpos[:] = state_msg.xpos
                self.data.xquat[:] = state_msg.xquat
                self.viser_model.set_data(self.data)
        except ValueError:
            # we're switching tasks and the new task has a different number of xpos/xquat
            return

    @on_event("INPUT", "traces")
    def update_traces(self, event: dict) -> None:
        """Callback to update traces on receiving a new trace measurement."""
        traces_flat = event["value"].to_numpy()
        all_traces_rollout_size = int(event["metadata"]["all_traces_rollout_size"])
        shape = event["metadata"]["shape"]
        traces = traces_flat.reshape(*shape)
        with self.task_lock:
            self.viser_model.set_traces(traces, all_traces_rollout_size)

    @on_event("INPUT", "plan_time")
    def update_plan_time(self, event: dict) -> None:
        """Callback to update plan time on receiving a new plan time measurement."""
        plan_time_s = event["value"].to_numpy(zero_copy_only=False)[0]
        self.gui_elements["plan_time_display"].value = plan_time_s * 1000  # ms

    def _remove_gui_elements(self) -> None:
        """Remove GUI elements from the visualization node."""
        for v in self.gui_elements.values():
            if isinstance(v, list):
                for handle in v:
                    self.remove_handles(handle)
            else:
                v.remove()
        self.viser_model.remove()

    def cleanup(self) -> None:
        """Cleanup the visualization node."""
        self._remove_gui_elements()
        self.server.flush()
        self.server.stop()
        super().cleanup()

    def spin(self) -> None:
        """Spin logic for the visualization node."""
        for event in self.node:
            if self.sim_pause_updated.is_set():
                self.write_sim_pause()
                self.sim_pause_updated.clear()
            if self.task_updated.is_set():
                self.write_task()
                self.task_updated.clear()
            if self.task_reset_updated.is_set():
                self.write_task_reset()
                self.task_reset_updated.clear()
            if self.optimizer_updated.is_set():
                self.write_optimizer()
                self.optimizer_updated.clear()
            if self.controller_config_updated.is_set():
                self.write_controller_config()
                self.controller_config_updated.clear()
            if self.optimizer_config_updated.is_set():
                self.write_optimizer_config()
                self.optimizer_config_updated.clear()
            if self.task_config_updated.is_set():
                self.write_task_config()
                self.task_config_updated.clear()

            self.handle(event)
