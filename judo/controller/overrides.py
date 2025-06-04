# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.config import set_config_overrides
from judo.controller.controller import ControllerConfig


def set_default_cylinder_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cylinder push task."""
    set_config_overrides(
        "cylinder_push",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "zero",
        },
    )


def set_default_cartpole_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cartpole task."""
    set_config_overrides(
        "cartpole",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "zero",
        },
    )


def set_default_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube task."""
    set_config_overrides(
        "leap_cube",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_leap_cube_down_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube down task."""
    set_config_overrides(
        "leap_cube_down",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_caltech_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the caltech leap cube task."""
    set_config_overrides(
        "caltech_leap_cube",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "cubic",
            "max_num_traces": 1,
        },
    )


def set_default_fr3_pick_overrides() -> None:
    """Sets the default task-specific controller config overrides for the fr3 pick task."""
    set_config_overrides(
        "fr3_pick",
        ControllerConfig,
        {
            "horizon": 1.0,
            "spline_order": "linear",
            "max_num_traces": 3,
            "control_freq": 20.0,
        },
    )
