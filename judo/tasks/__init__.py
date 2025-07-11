# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import atexit
import inspect
import json
import os
import threading
import uuid
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Tuple, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    "cylinder_push": (CylinderPush, CylinderPushConfig),
    "cartpole": (Cartpole, CartpoleConfig),
    "fr3_pick": (FR3Pick, FR3PickConfig),
    "leap_cube": (LeapCube, LeapCubeConfig),
    "leap_cube_down": (LeapCubeDown, LeapCubeDownConfig),
    "caltech_leap_cube": (CaltechLeapCube, CaltechLeapCubeConfig),
}
_builtin_names = set(_registered_tasks.keys())

# set a run ID for this process that is used to persist programmatic registrations
_run_id = os.environ.get("JUDO_TASK_RUN_ID")
_new_run = _run_id is None
if _new_run:
    _run_id = uuid.uuid4().hex
    os.environ["JUDO_TASK_RUN_ID"] = _run_id
_CUSTOM_REGISTRY_PATH = Path(f"/tmp/judo_tasks_{_run_id}.json")

# remove any potential stale judo_tasks_*.json files
for old in _CUSTOM_REGISTRY_PATH.parent.glob("judo_tasks_*.json"):
    if old.name != _CUSTOM_REGISTRY_PATH.name:
        try:
            old.unlink()
        except OSError:
            pass


# register a cleanup function to remove the file on exit
def _cleanup_registry_file() -> None:
    """Remove the ephemeral registry file on exit of the main."""
    try:
        _CUSTOM_REGISTRY_PATH.unlink()
    except FileNotFoundError:
        pass


atexit.register(_cleanup_registry_file)

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
        spec = spec_from_file_location(f"_judo_task_{name}", info["task_src"])
        assert spec is not None, f"Could not load task module {info['task_src']}"
        mod = module_from_spec(spec)
        assert spec.loader is not None, f"Could not load task module {info['task_src']}"
        spec.loader.exec_module(mod)
        task_cls = getattr(mod, info["task_qn"])

        # load Config class
        spec2 = spec_from_file_location(f"_judo_cfg_{name}", info["cfg_src"])
        assert spec2 is not None, f"Could not load config module {info['cfg_src']}"
        mod2 = module_from_spec(spec2)
        assert spec2.loader is not None, f"Could not load config module {info['cfg_src']}"
        spec2.loader.exec_module(mod2)
        cfg_cls = getattr(mod2, info["cfg_qn"])

        _registered_tasks[name] = (task_cls, cfg_cls)


def _persist_ephemeral_registry() -> None:
    """After any register_task(), write the current state of the registry to disk."""
    with _registry_lock:
        out: Dict[str, Dict[str, str]] = {}
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


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
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
