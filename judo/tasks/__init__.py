# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.spot.spot_box import SpotBox, SpotBoxConfig
from judo.tasks.spot.spot_yellow_chair import SpotYellowChair, SpotYellowChairConfig
from judo.tasks.spot.spot_yellow_chair_ramp import SpotYellowChairRamp, SpotYellowChairRampConfig

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    SpotBox.name: (SpotBox, SpotBoxConfig),
    SpotYellowChair.name: (SpotYellowChair, SpotYellowChairConfig),
    SpotYellowChairRamp.name: (SpotYellowChairRamp, SpotYellowChairRampConfig),
    CylinderPush.name: (CylinderPush, CylinderPushConfig),
    Cartpole.name: (Cartpole, CartpoleConfig),
    FR3Pick.name: (FR3Pick, FR3PickConfig),
    LeapCube.name: (LeapCube, LeapCubeConfig),
    LeapCubeDown.name: (LeapCubeDown, LeapCubeDownConfig),
    CaltechLeapCube.name: (CaltechLeapCube, CaltechLeapCubeConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    """Registers a new task."""
    _registered_tasks[name] = (task_type, task_config_type)


__all__ = [
    "get_registered_tasks",
    "register_task",
    "Task",
    "TaskConfig",
    "CaltechLeapCube",
    "CaltechLeapCubeConfig",
    "Cartpole",
    "CartpoleConfig",
    "CylinderPush",
    "CylinderPushConfig",
    "FR3Pick",
    "FR3PickConfig",
    "LeapCube",
    "LeapCubeConfig",
    "LeapCubeDown",
    "LeapCubeDownConfig",
    "SpotBox",
    "SpotBoxConfig",
    "SpotYellowChair",
    "SpotYellowChairConfig",
    "SpotYellowChairRamp",
    "SpotYellowChairRampConfig",
]
