# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import importlib

from omegaconf import DictConfig

from judo.optimizers import register_optimizer
from judo.tasks import register_task


def get_class_from_string(class_path: str) -> type:
    """Get a class from a string path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def register_tasks_from_cfg(task_registration_cfg: DictConfig) -> None:
    """Register custom tasks."""
    for task_name in task_registration_cfg.keys():
        task_dict = task_registration_cfg.get(task_name, {})
        assert set(task_dict.keys()) == {"task", "config"}, (
            "Task registration must be a dict with keys 'task' and 'config'."
        )
        assert isinstance(task_dict["task"], str), "Task must be a string path to the task class."
        assert isinstance(task_dict["config"], str), "Task config must be a string path to the config class."
        task_cls = get_class_from_string(task_dict["task"])
        task_config_cls = get_class_from_string(task_dict["config"])
        register_task(str(task_name), task_cls, task_config_cls)


def register_optimizers_from_cfg(optimizer_registration_cfg: DictConfig) -> None:
    """Register custom optimizers."""
    for optimizer_name in optimizer_registration_cfg.keys():
        optimizer_dict = optimizer_registration_cfg.get(optimizer_name, {})
        assert set(optimizer_dict.keys()) == {"optimizer", "config"}, (
            "Optimizer registration must be a dict with keys 'optimizer' and 'config'."
        )
        assert isinstance(optimizer_dict["optimizer"], str), "Optimizer must be a string path to the optimizer class."
        assert isinstance(optimizer_dict["config"], str), "Optimizer config must be a string path to the config class."
        optimizer_cls = get_class_from_string(optimizer_dict["optimizer"])
        optimizer_config_cls = get_class_from_string(optimizer_dict["config"])
        register_optimizer(str(optimizer_name), optimizer_cls, optimizer_config_cls)
