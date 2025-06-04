# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Type

from judo.optimizers.base import Optimizer, OptimizerConfig
from judo.optimizers.cem import CrossEntropyMethod, CrossEntropyMethodConfig
from judo.optimizers.mppi import MPPI, MPPIConfig
from judo.optimizers.overrides import (
    set_default_caltech_leap_cube_overrides,
    set_default_cartpole_overrides,
    set_default_cylinder_push_overrides,
    set_default_fr3_pick_overrides,
    set_default_leap_cube_down_overrides,
    set_default_leap_cube_overrides,
)
from judo.optimizers.ps import PredictiveSampling, PredictiveSamplingConfig

set_default_caltech_leap_cube_overrides()
set_default_cartpole_overrides()
set_default_cylinder_push_overrides()
set_default_fr3_pick_overrides()
set_default_leap_cube_overrides()
set_default_leap_cube_down_overrides()

_registered_optimizers: dict[str, tuple[Type[Optimizer], Type[OptimizerConfig]]] = {
    "cem": (CrossEntropyMethod, CrossEntropyMethodConfig),
    "mppi": (MPPI, MPPIConfig),
    "ps": (PredictiveSampling, PredictiveSamplingConfig),
}


def get_registered_optimizers() -> dict[str, tuple[Type[Optimizer], Type[OptimizerConfig]]]:
    """Get the registered optimizer."""
    return _registered_optimizers


def register_optimizer(
    name: str,
    controller_type: Type[Optimizer],
    controller_config_type: Type[OptimizerConfig],
) -> None:
    """Register a new optimizer."""
    _registered_optimizers[name] = (controller_type, controller_config_type)


__all__ = [
    "get_registered_optimizers",
    "register_optimizer",
    "CrossEntropyMethod",
    "CrossEntropyMethodConfig",
    "MPPI",
    "MPPIConfig",
    "Optimizer",
    "OptimizerConfig",
    "PredictiveSampling",
    "PredictiveSamplingConfig",
]
