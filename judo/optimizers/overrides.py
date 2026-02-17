# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.config import set_config_overrides
from judo.optimizers.base import OptimizerConfig
from judo.optimizers.cem import CrossEntropyMethodConfig
from judo.optimizers.mppi import MPPIConfig
from judo.optimizers.ps import PredictiveSamplingConfig


def set_default_cylinder_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cylinder push task."""
    set_config_overrides(
        "cylinder_push",
        PredictiveSamplingConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
        },
    )
    set_config_overrides(
        "cylinder_push",
        CrossEntropyMethodConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "num_elites": 2,
            "use_noise_ramp": True,
        },
    )
    set_config_overrides(
        "cylinder_push",
        MPPIConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
        },
    )


def set_default_cartpole_overrides() -> None:
    """Sets the default task-specific controller config overrides for the cartpole task."""
    set_config_overrides(
        "cartpole",
        PredictiveSamplingConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
        },
    )
    set_config_overrides(
        "cartpole",
        CrossEntropyMethodConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "num_elites": 2,
            "use_noise_ramp": True,
        },
    )
    set_config_overrides(
        "cartpole",
        MPPIConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
        },
    )


def set_default_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube task."""
    set_config_overrides(
        "leap_cube",
        PredictiveSamplingConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.2,
        },
    )
    set_config_overrides(
        "leap_cube",
        CrossEntropyMethodConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "num_elites": 3,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
        },
    )
    set_config_overrides(
        "leap_cube",
        MPPIConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.2,
            "temperature": 0.0025,
        },
    )


def set_default_leap_cube_down_overrides() -> None:
    """Sets the default task-specific controller config overrides for the leap cube down task."""
    set_config_overrides(
        "leap_cube_down",
        PredictiveSamplingConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.2,
        },
    )
    set_config_overrides(
        "leap_cube_down",
        CrossEntropyMethodConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 64,
            "num_elites": 3,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
        },
    )
    set_config_overrides(
        "leap_cube_down",
        MPPIConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 64,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.2,
            "temperature": 0.0025,
        },
    )


def set_default_caltech_leap_cube_overrides() -> None:
    """Sets the default task-specific controller config overrides for the caltech leap cube task."""
    set_config_overrides(
        "caltech_leap_cube",
        PredictiveSamplingConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.2,
        },
    )
    set_config_overrides(
        "caltech_leap_cube",
        CrossEntropyMethodConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "num_elites": 3,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
        },
    )
    set_config_overrides(
        "caltech_leap_cube",
        MPPIConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 32,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.2,
            "temperature": 0.0025,
        },
    )


def _set_spot_optimizer_overrides(task_name: str) -> None:
    """Sets the default optimizer config overrides for a Spot task."""
    _spot_base = {
        "num_rollouts": 24,
        "num_nodes": 3,
        "use_noise_ramp": True,
        "noise_ramp": 3.5,
    }
    set_config_overrides(task_name, OptimizerConfig, _spot_base)
    set_config_overrides(task_name, PredictiveSamplingConfig, _spot_base)
    set_config_overrides(task_name, CrossEntropyMethodConfig, {**_spot_base, "num_elites": 3})
    set_config_overrides(task_name, MPPIConfig, _spot_base)


def set_default_spot_tire_upright_overrides() -> None:
    """Sets the default task-specific optimizer config overrides for the spot_tire_upright task."""
    _set_spot_optimizer_overrides("spot_tire_upright")


def set_default_fr3_pick_overrides() -> None:
    """Sets the default task-specific controller config overrides for the fr3 pick task."""
    set_config_overrides(
        "fr3_pick",
        PredictiveSamplingConfig,
        {
            "num_nodes": 8,
            "num_rollouts": 64,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.2,
        },
    )
    set_config_overrides(
        "fr3_pick",
        CrossEntropyMethodConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 64,
            "num_elites": 3,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma_min": 0.01,
            "sigma_max": 0.3,
        },
    )
    set_config_overrides(
        "fr3_pick",
        MPPIConfig,
        {
            "num_nodes": 4,
            "num_rollouts": 64,
            "use_noise_ramp": True,
            "noise_ramp": 4.0,
            "sigma": 0.01,
            "temperature": 0.002,
        },
    )
