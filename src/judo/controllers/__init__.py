# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.controllers.controller import (
    Controller,
    ControllerConfig,
)
from judo.controllers.sampling.cmaes import CMAES, CMAESConfig
from judo.controllers.sampling.cross_entropy_method import (
    CrossEntropyConfig,
    CrossEntropyMethod,
)
from judo.controllers.sampling.mppi import MPPI, MPPIConfig
from judo.controllers.sampling.predictive_sampling import (
    PredictiveSampling,
    PredictiveSamplingConfig,
)

_registered_controllers: Dict[str, Tuple[Type[Controller], Type[ControllerConfig]]] = {
    "cross_entropy_method": (CrossEntropyMethod, CrossEntropyConfig),
    "MPPI": (MPPI, MPPIConfig),
    "predictive_sampling": (PredictiveSampling, PredictiveSamplingConfig),
    "cmaes": (CMAES, CMAESConfig),
}


def get_registered_controllers() -> (
    Dict[str, Tuple[Type[Controller], Type[ControllerConfig]]]
):
    return _registered_controllers


def register_controller(
    name: str,
    controller_type: Type[Controller],
    controller_config_type: Type[ControllerConfig],
) -> None:
    _registered_controllers[name] = (controller_type, controller_config_type)
