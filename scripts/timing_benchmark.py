# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from dataclasses import fields

import numpy as np
from mujoco import mj_step
from tqdm import tqdm

from judo.controller import Controller, ControllerConfig
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks


def summarize_dataclass(dataclass_instance) -> None:  # noqa: ANN001
    """Prints the fields and values of a dataclass instance."""
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)
        print(f"  {field.name}: {value}")


def time_one_task_optimizer_pair(
    task_name: str,
    optimizer_name: str,
    num_steps: int = 1000,
) -> None:
    """Times one task-optimizer pair.

    Args:
        task_name: The name of the task to benchmark.
        optimizer_name: The name of the optimizer to benchmark.
        num_steps: The number of steps to run the benchmark for.

    Returns:
        The average time per step in seconds.
    """
    # load the controller
    available_tasks = get_registered_tasks()
    available_optimizers = get_registered_optimizers()
    if task_name not in available_tasks.keys():
        raise ValueError(f"Task '{task_name}' is not registered.")
    if optimizer_name not in available_optimizers.keys():
        raise ValueError(f"Optimizer '{optimizer_name}' is not registered.")

    task_class, task_config_class = available_tasks[task_name]
    optimizer_class, optimizer_config_class = available_optimizers[optimizer_name]

    task = task_class()
    task_config = task_config_class()
    optimizer_config = optimizer_config_class()
    optimizer_config.set_override(task_name)
    optimizer = optimizer_class(optimizer_config, task.nu)
    controller_config = ControllerConfig()
    controller_config.set_override(task_name)
    controller = Controller(controller_config, task, task_config, optimizer, optimizer_config)

    # summarize the configs
    print("Task Config:")
    summarize_dataclass(task_config)
    print("Optimizer Config:")
    summarize_dataclass(optimizer_config)
    print("Controller Config:")
    summarize_dataclass(controller_config)

    # getting model and data
    task.reset()
    controller.reset()
    sim_model = task.sim_model
    sim_data = task.data

    # warm up the task by running a synchronous loop for some number of steps to get it into a "typical" state
    # cheat a little by allowing the controller to replan at every step for max performance
    print("Warming up...")
    for _ in tqdm(range(1000)):
        curr_time = sim_data.time
        curr_state = np.concatenate([sim_data.qpos, sim_data.qvel])

        controller.update_action(curr_state, curr_time)
        sim_data.ctrl[:] = controller.action(curr_time)
        task.pre_sim_step()
        mj_step(sim_model, sim_data)
        controller.system_metadata = task.get_sim_metadata()
        task.post_sim_step()

    # timing the controller
    print(f"Timing {task_name} with {optimizer_name} for {num_steps} steps...")
    curr_time = sim_data.time
    curr_state = np.concatenate([sim_data.qpos, sim_data.qvel])
    start_time = time.perf_counter()
    for _ in tqdm(range(num_steps)):
        controller.update_action(curr_state, curr_time)
    end_time = time.perf_counter()
    avg_time_per_step = (end_time - start_time) / num_steps
    print(f"Average time per planning step: {avg_time_per_step:.6f} seconds")


if __name__ == "__main__":
    # time_one_task_optimizer_pair("walker", "ps", num_steps=1000)
    time_one_task_optimizer_pair("leap_cube", "ps", num_steps=1000)
