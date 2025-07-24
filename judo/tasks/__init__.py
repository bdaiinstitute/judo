# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import importlib
import inspect
import json
import threading
from importlib.util import module_from_spec, spec_from_file_location
from typing import Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.utils.registration import get_run_id, make_ephemeral_path

_registered_tasks: dict[str, tuple[Type[Task], Type[TaskConfig]]] = {
    "cylinder_push": (CylinderPush, CylinderPushConfig),
    "cartpole": (Cartpole, CartpoleConfig),
    "fr3_pick": (FR3Pick, FR3PickConfig),
    "leap_cube": (LeapCube, LeapCubeConfig),
    "leap_cube_down": (LeapCubeDown, LeapCubeDownConfig),
    "caltech_leap_cube": (CaltechLeapCube, CaltechLeapCubeConfig),
}
_builtin_names = set(_registered_tasks.keys())

# set a run ID for this process that is used to persist programmatic registrations
_run_id = get_run_id()
_CUSTOM_REGISTRY_PATH = make_ephemeral_path("judo_tasks", _run_id)
_registry_lock = threading.Lock()


def _load_ephemeral_registry() -> None:
    """On import, pull in any user-registered tasks from the temp file."""
    if not _CUSTOM_REGISTRY_PATH.is_file():
        return
    try:
        data = json.loads(_CUSTOM_REGISTRY_PATH.read_text())
    except Exception:
        return

    for name, info in data.items():
        # load Task class
        task_mod = info.get("task_mod")
        if task_mod and not task_mod.startswith("__main__"):
            mod = importlib.import_module(task_mod)  # package import preserves relative imports in that module
            task_cls = getattr(mod, info["task_qn"])
        else:
            # fallback: load by path
            spec = spec_from_file_location(f"_judo_task_{name}", info["task_src"])
            assert spec and spec.loader, f"Could not load task module {info['task_src']}"
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            task_cls = getattr(mod, info["task_qn"])

        # load Config class
        cfg_mod = info.get("cfg_mod")
        if cfg_mod and not cfg_mod.startswith("__main__"):
            mod2 = importlib.import_module(cfg_mod)
            cfg_cls = getattr(mod2, info["cfg_qn"])
        else:
            spec2 = spec_from_file_location(f"_judo_cfg_{name}", info["cfg_src"])
            assert spec2 and spec2.loader, f"Could not load config module {info['cfg_src']}"
            mod2 = module_from_spec(spec2)
            spec2.loader.exec_module(mod2)
            cfg_cls = getattr(mod2, info["cfg_qn"])

        _registered_tasks[name] = (task_cls, cfg_cls)


def _persist_ephemeral_registry() -> None:
    """After any register_task(), write the current state of the registry to disk."""
    with _registry_lock:
        out: dict[str, dict[str, str]] = {}
        for name, (task_cls, cfg_cls) in _registered_tasks.items():
            if name in _builtin_names:
                continue
            task_src = inspect.getsourcefile(task_cls)
            cfg_src = inspect.getsourcefile(cfg_cls)
            if task_src and cfg_src:
                out[name] = {
                    "task_src": task_src,
                    "task_qn": task_cls.__qualname__,
                    "cfg_src": cfg_src,
                    "cfg_qn": cfg_cls.__qualname__,
                }
        _CUSTOM_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CUSTOM_REGISTRY_PATH.write_text(json.dumps(out, indent=2))


# load any ephemeral registrations on import
_load_ephemeral_registry()


def get_registered_tasks() -> dict[str, tuple[Type[Task], Type[TaskConfig]]]:
    """Get the currently registered tasks."""
    return _registered_tasks


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    """Register a new task with the Judo framework for this run only."""
    _registered_tasks[name] = (task_type, task_config_type)
    _persist_ephemeral_registry()


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
]
