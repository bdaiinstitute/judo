# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from dataclasses import MISSING, Field, field, fields, is_dataclass, make_dataclass
from threading import Event, Lock
from typing import Any, Callable, List, Literal, get_args, get_origin

import numpy as np
from viser import (
    GuiCheckboxHandle,
    GuiDropdownHandle,
    GuiEvent,
    GuiFolderHandle,
    GuiInputHandle,
    GuiSliderHandle,
    MeshHandle,
    ViserServer,
)

DEFAULT_SLIDER_STEP_FLOAT = 0.01
DEFAULT_SLIDER_STEP_INT = 1
GOAL_RADIUS = 0.05


def slider(
    parameter_name: str,
    min: int | float,
    max: int | float,
    step: float | None = None,
) -> Callable[[type], type]:
    """Decorator that adds slider metadata to desired dataclass fields.

    Args:
        parameter_name: target parameter to annotate.
        min: minimum value for slider.
        max: maximum value for slider.
        step: step for slider handle. If not set, defaults to constant 0.01 for floats and 1 for ints.
    """
    if step is None:
        step = DEFAULT_SLIDER_STEP_INT if isinstance(min, int) else DEFAULT_SLIDER_STEP_FLOAT

    def wrapper(cls: type) -> type:
        assert is_dataclass(cls), "@slider decorator can only be applied to dataclasses!"

        new_fields = []
        for f in fields(cls):
            meta = dict(f.metadata)
            if f.name == parameter_name:
                meta["ui_config"] = (min, max, step)
            new_fields.append(
                (
                    f.name,
                    f.type,
                    field(
                        init=f.init,
                        repr=f.repr,
                        hash=f.hash,
                        compare=f.compare,
                        default=f.default if f.default is not MISSING else MISSING,
                        default_factory=f.default_factory if f.default_factory is not MISSING else MISSING,
                        metadata=meta,
                        kw_only=f.kw_only,
                    ),
                )
            )

        # in order to reflect changes to the fields, we must instantiate a new dataclass instance, because fields are
        # frozen and cannot be modified in place!
        new_cls = make_dataclass(cls.__name__, new_fields, bases=cls.__bases__, namespace=dict(cls.__dict__))
        return new_cls

    return wrapper


def _get_gui_element(
    server: ViserServer, config: Any, field: Field
) -> GuiInputHandle | MeshHandle | GuiFolderHandle | list | None:
    """Helper function that creates a GUI element for a particular dataclass field.

    Args:
        server: ViserServer to which GUI elements will be added.
        config: Dataclass containing parameters for which GUI elements should be created.
        field: Current field that a GUI element will be added.
    """
    assert is_dataclass(config), "GUI elements can only be added for dataclasses!"
    init_value = getattr(config, field.name)

    # Create a slider for an integer-valued param.
    if field.type is int:
        if "ui_config" not in field.metadata:
            if init_value < 0:
                min_int, max_int, step_int = 2 * init_value, -1, DEFAULT_SLIDER_STEP_INT
            elif init_value == 0:
                min_int, max_int, step_int = -5, 5, DEFAULT_SLIDER_STEP_INT
            else:
                min_int, max_int, step_int = 1, 2 * init_value, DEFAULT_SLIDER_STEP_INT
        else:
            min_int, max_int, step_int = field.metadata["ui_config"]

        # if the init_value is outside of the bounds, update the bounds
        if init_value < min_int or init_value > max_int:
            min_float = min(min_int, init_value)
            max_float = max(max_int, init_value)

        return server.gui.add_slider(
            field.name,
            min=min_int,
            max=max_int,
            step=step_int,
            initial_value=init_value,
        )

    # Create a slider for a float-valued param.
    elif field.type is float:
        if "ui_config" not in field.metadata:
            if init_value < 0.0:
                min_float, max_float, step_float = 2 * init_value, 0.0, DEFAULT_SLIDER_STEP_FLOAT
            elif init_value == 0.0:
                min_float, max_float, step_float = -5.0, 5.0, DEFAULT_SLIDER_STEP_FLOAT
            else:
                min_float, max_float, step_float = 0.0, 2 * init_value, DEFAULT_SLIDER_STEP_FLOAT
        else:
            min_float, max_float, step_float = field.metadata["ui_config"]

        # if the init_value is outside of the bounds, update the bounds
        if init_value < min_float or init_value > max_float:
            min_float = min(min_float, init_value)
            max_float = max(max_float, init_value)

        return server.gui.add_slider(
            field.name,
            min=min_float,
            max=max_float,
            step=step_float,
            initial_value=init_value,
        )

    # Create a checkbox for a boolean param.
    elif field.type is bool:
        return server.gui.add_checkbox(field.name, initial_value=init_value)

    elif field.type is np.ndarray:
        # TODO(ahl): generalize this logic to allow for non-int/float
        if init_value.dtype.kind not in ["i", "f"]:  # checks if the dtype is int or float
            return None

        folder = server.gui.add_folder(field.name)
        if "ui_array_config" in field.metadata:
            names = field.metadata["ui_array_config"]["names"]
            mins = field.metadata["ui_array_config"]["mins"]
            maxs = field.metadata["ui_array_config"]["maxs"]
            steps = field.metadata["ui_array_config"]["steps"]
            vis = field.metadata["ui_array_config"]["vis"]
        else:
            names = [f"{i}" for i in range(len(init_value))]
            mins = [0.5 * i for i in init_value]
            maxs = [1.5 * i for i in init_value]
            steps = [(maxs[i] - mins[i]) / 100.0 for i in range(len(init_value))]
            vis = None
        slider_handles = []
        with folder:
            for i in range(len(init_value)):
                slider_handle = server.gui.add_slider(
                    names[i],
                    min=mins[i],
                    max=maxs[i],
                    step=steps[i],
                    initial_value=init_value[i],
                )
                slider_handles.append(slider_handle)

        # check whether we have a visualization specification for the np array.
        # if so, return both the folder and a mesh handle corresponding to some icosphere.
        if vis is not None:
            # compute the initial position of the mesh handle.
            xyz_vis_indices = vis["xyz_vis_indices"]
            xyz_vis_defaults = vis["xyz_vis_defaults"]
            position = np.array(
                [
                    init_value[xyz_vis_indices[i]] if xyz_vis_indices[i] is not None else xyz_vis_defaults[i]
                    for i in range(len(xyz_vis_indices))
                ]
            )

            # make a mesh, which we can move with slider callbacks
            mesh_handle = server.scene.add_icosphere(
                vis["name"],
                radius=GOAL_RADIUS,
                color=(0.0, 0.0, 1.0),
                position=position,
            )

            # register slider callbacks to update the mesh frame position
            for i in xyz_vis_indices:
                if i is None:  # when None is specified in one position, we don't update it, e.g., planar visualizations
                    continue
                slider_handle = slider_handles[i]

                @slider_handle.on_update
                def _(_: GuiEvent, index: int = i, slider_handle: GuiSliderHandle = slider_handle) -> None:
                    pos = mesh_handle.position
                    new_pos = np.copy(pos)
                    new_pos[index] = slider_handle.value
                    with server.atomic():
                        mesh_handle.position = new_pos

            return [folder, mesh_handle]
        else:
            return folder

    # Create a dropdown for a Literal-typed param.
    elif get_origin(field.type) == Literal:
        return server.gui.add_dropdown(field.name, get_args(field.type), initial_value=init_value)

    else:
        warnings.warn(
            f"Warning: Field {field.name} with type {field.type} is not a supported type for GUI creation.",
            stacklevel=1,
        )
        return None


def _get_callback(
    element: GuiInputHandle,
    element_name: str,
    config: Any,
    config_event: Event,
    config_lock: Lock,
    array_name: str | None = None,
    array_index: int | None = None,
) -> Callable[[GuiEvent], None]:
    """Factory method for creating a GUI callback for a particular GUI element.

    Args:
        element: GuiInputHandle that callback will be created for.
        element_name: Config dict key corresponding to GUI element.
        config: the dataclass to write into.
        config_event: Event flagging that this config param has been updated.
        config_lock: multiprocessing Lock that ensures thread safety when reading / writing config values.
        array_name: Name of the folder that contains this element, if any. Folder elements correspond to array-like
            config entries, so this lets us know which index of the array we are updating.
        array_index: Index of the array that this element corresponds to, if any. This is used to update the correct
            index of the array in the config dataclass.
    """

    def gui_callback(_: GuiEvent) -> None:
        """Updates the config with the new value received from viser."""
        if isinstance(element, GuiSliderHandle):
            # if value exceeds the bounds, update the bounds
            if element.value < element.min or element.value > element.max:
                element.min = min(element.min, element.value)
                element.max = max(element.max, element.value)
            value = element.value
        else:
            value = element.value

        # Use lock for thread safety.
        with config_lock:
            if array_name is not None and array_index is not None:
                array_value = getattr(config, array_name)
                array_value[array_index] = value
            else:
                setattr(config, element_name, value)

        # Set config_event so other Processes know to update this value.
        config_event.set()

    return gui_callback


def create_gui_elements(
    server: ViserServer,
    config: Any,
    config_updated_event: Event,
    config_lock: Lock,
) -> List[GuiInputHandle]:
    """Recursively iterates through a dataclass, adding GUI elements / callbacks for its fields to a Viser server.

    Args:
        server: current `ViserServer` to which the GUI elements are added.
        config: dataclass containing params which we want to update via viser.
        config_dict: `multiprocessing.DictProxy` where param values are shared between threads.
        config_updated_event: `multiprocessing.Event` that flags when config params are updated.
        config_lock: `multiprocessing.Lock` that prevents multiple processes from accessing the same param.

    Returns a list of GuiHandles that can be used to remove all elements created here on destruction.
    """
    gui_elements = []

    # Create GUI elements for each field in the dataclass.
    for config_parameter in fields(config):
        # If the dataclass is nested, traverse it recursively.
        if is_dataclass(config_parameter.type):
            config_folder = server.gui.add_folder(config_parameter.name)
            gui_elements.append(config_folder)
            with config_folder:
                new_elements = create_gui_elements(
                    server,
                    getattr(config, config_parameter.name),
                    config_updated_event,
                    config_lock,
                )
            gui_elements.append(new_elements)

        # Otherwise add an atomic GUI element for this parameter.
        else:
            element = _get_gui_element(server, config, config_parameter)
            if element is None:
                continue
            elif isinstance(element, list):
                for e in element:
                    register_gui_element_callback(
                        e,
                        config_parameter,
                        config,
                        config_updated_event,
                        config_lock,
                    )
                    gui_elements.append(e)
            else:
                register_gui_element_callback(
                    element,
                    config_parameter,
                    config,
                    config_updated_event,
                    config_lock,
                )
                gui_elements.append(element)
    return gui_elements


def register_gui_element_callback(
    element: Any,
    config_parameter: Field,
    config: Any,
    config_updated_event: Event,
    config_lock: Lock,
) -> None:
    """Register a callback for a GUI element.

    Args:
        element: The GUI element to register the callback for.
        config_parameter: The field in the dataclass that corresponds to the GUI element.
        config: The dataclass to write into.
        config_updated_event: The event to set when the config is updated.
        config_lock: The lock to use for thread safety.
    """
    if isinstance(element, (GuiSliderHandle, GuiDropdownHandle, GuiCheckboxHandle)):
        element.on_update(
            _get_callback(
                element,
                config_parameter.name,
                config,
                config_updated_event,
                config_lock,
            )
        )
    elif isinstance(element, GuiFolderHandle):
        for i, handle in enumerate(element._children.values()):
            handle.on_update(
                _get_callback(
                    handle,
                    f"{handle.label}",
                    config,
                    config_updated_event,
                    config_lock,
                    array_name=element.label,
                    array_index=i,
                )
            )
