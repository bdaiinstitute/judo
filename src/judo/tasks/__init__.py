# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.tasks.acrobot import Acrobot, AcrobotConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import (
    CylinderPush,
    CylinderPushConfig,
)
from judo.tasks.task import Task, TaskConfig

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    "acrobot": (Acrobot, AcrobotConfig),
    "cylinder_push": (CylinderPush, CylinderPushConfig),
    "cartpole": (Cartpole, CartpoleConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    return _registered_tasks


def register_task(
    name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]
) -> None:
    _registered_tasks[name] = (task_type, task_config_type)
