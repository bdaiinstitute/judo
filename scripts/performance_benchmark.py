# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import datetime
from dataclasses import fields
from fractions import Fraction
from functools import reduce
from math import gcd, isclose
from textwrap import dedent
from typing import Any, Sequence

import h5py
import numpy as np

# Define the Spot SimBackend from the provided code
from mujoco import mj_step
from tqdm import tqdm

from judo.controller import Controller, ControllerConfig
from judo.optimizers import Optimizer, OptimizerConfig, get_registered_optimizers
from judo.tasks import Task, TaskConfig, get_registered_tasks
from judo.utils.mujoco import SimBackendSpot

SPOT_BACKEND_AVAILABLE = True

# ##### #
# UTILS #
# ##### #


def _as_fraction(x: float, max_den: int = 10**4) -> Fraction:
    """Convert a float to a Fraction with limited denominator."""
    return Fraction(x).limit_denominator(max_den)


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _lcm_many(values: Sequence[int]) -> int:
    return reduce(_lcm, values)


def _gcd_many(values: Sequence[int]) -> int:
    return reduce(gcd, values)


def compute_common_step_many(dts: Sequence[float], *, max_den: int = 10**4) -> tuple[float, list[int]]:
    """Pick a loop step that evenly divides all given dts.

    Returns:
        step_dt: the loop step size (rational GCD of all dts).
        loop_steps: list of loop iterations per 1 dt, in the same order as `dts`.
    """
    assert len(dts) >= 1, "Provide at least one dt."
    assert all(dt > 0 for dt in dts), "Timesteps must be positive."

    fracs = [_as_fraction(dt, max_den) for dt in dts]

    # put all fractions over a common denominator L (integer grid)
    L = _lcm_many([f.denominator for f in fracs])
    nums = [f.numerator * (L // f.denominator) for f in fracs]  # each dt as n_i / L

    g = _gcd_many(nums)  # integer GCD on the common grid
    step_frac = Fraction(g, L)  # step_dt as a rational
    step_dt = float(step_frac)

    loop_steps = [n // g for n in nums]  # exact integer counts by construction

    # sanity checks (tolerant to float conversion)
    for dt_f, loops in zip(fracs, loop_steps, strict=False):
        assert loops > 0
        assert isclose(step_dt * loops, float(dt_f), abs_tol=1e-15)

    return step_dt, loop_steps


def summarize_dataclass(dataclass_instance) -> None:  # noqa: ANN001
    """Prints the fields and values of a dataclass instance."""
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)
        print(f"  {field.name}: {value}")


# ############## #
# MAIN FUNCTIONS #
# ############## #


def load_controller(
    task_name: str, optimizer_name: str, verbose: bool = False
) -> tuple[
    Task,
    Optimizer,
    Controller,
    TaskConfig,
    OptimizerConfig,
    ControllerConfig,
]:
    """Loads a controller associated with a task and optimizer."""
    # load task
    task_dict = get_registered_tasks()
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' is not registered.")
    task_cls, task_config_cls = task_dict[task_name]
    task_config = task_config_cls()
    task = task_cls()

    # load optimizer
    optimizer_dict = get_registered_optimizers()
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Optimizer '{optimizer_name}' is not registered.")
    optimizer_cls, optimizer_config_cls = optimizer_dict[optimizer_name]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override(task_name)
    optimizer = optimizer_cls(optimizer_config, task.nu)

    # load controller
    controller_config_cls = ControllerConfig
    controller_config = controller_config_cls()
    controller_config.set_override(task_name)
    controller = Controller(
        controller_config,
        task,
        task_config,
        optimizer,
        optimizer_config,
    )

    # summarize the task, optimizer, and controller configs for debugging
    if verbose:
        print("*" * 80)
        print("Loaded the following configurations...\n")
        print(f"Task ({task_name}):")
        summarize_dataclass(task_config)
        print(f"Optimizer ({optimizer_name}):")
        summarize_dataclass(optimizer_config)
        print("Controller:")
        summarize_dataclass(controller_config)
        print("*" * 80)

    return task, optimizer, controller, task_config, optimizer_config, controller_config


def benchmark_single_task_and_optimizer(
    task_name: str,
    optimizer_name: str,
    num_episodes: int = 10,
    episode_length_s: float = 30.0,
    viz_dt: float = 0.02,
    min_dt: float = 0.0001,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Benchmarks a single task and optimizer combination.

    Args:
        task_name: Name of the task to benchmark.
        optimizer_name: Name of the optimizer to use.
        num_episodes: Number of episodes to run for the benchmark.
        episode_length_s: Maximum length of each episode (in seconds).
        viz_dt: Timestep for visualization/logging (in seconds).
        min_dt: Minimum timestep as a quantum unit for synchronization between simulation and controller.
        verbose: Whether to print detailed configuration information.

    Returns:
        benchmark_results: A list of dictionaries containing results for each episode.
    """
    # load controller
    task, optimizer, controller, task_config, optimizer_config, controller_config = load_controller(
        task_name,
        optimizer_name,
        verbose=verbose,
    )
    sim_model = task.sim_model
    ctrl_model = task.model

    # set up the synchronous loop
    sim_dt = sim_model.opt.timestep
    plan_dt = 1.0 / controller_config.control_freq
    ctrl_dt = ctrl_model.opt.timestep
    assert sim_dt <= ctrl_dt, "Simulation timestep must be less than or equal to controller timestep."
    step_dt, loop_step_counts = compute_common_step_many([sim_dt, plan_dt, ctrl_dt, viz_dt])
    loop_steps_per_sim_step = loop_step_counts[0]  # sim_dt
    loop_steps_per_plan_step = loop_step_counts[1]  # plan_dt
    loop_steps_per_ctrl_step = loop_step_counts[2]  # ctrl_dt
    loop_steps_per_viz_step = loop_step_counts[3]  # viz_dt
    assert step_dt >= min_dt, f"Common step {step_dt} is less than minimum dt {min_dt}."
    num_steps = int(episode_length_s / step_dt) + 1

    # storage for outcomes
    benchmark_results = []

    # Initialize Spot backend if needed
    spot_sim_backend = None
    is_spot_task = hasattr(task, "task_to_sim_ctrl")
    if is_spot_task:
        spot_sim_backend = SimBackendSpot(task_to_sim_ctrl=task.task_to_sim_ctrl)

    # loop through episodes
    for i in range(num_episodes):
        np.random.seed(i)  # reset seed for each task for fairness

        # reset task and controller
        task.reset()
        controller.reset()
        curr_time = 0.0
        task.data.time = 0.0  # manually reset here in case task.reset() doesn't
        curr_state = np.concatenate([task.data.qpos, task.data.qvel])

        # simulating
        curr_episode_results = {
            "rewards": [],
            "qpos_traj": [],
            "mocap_pos_traj": [],
            "mocap_quat_traj": [],
            "success": False,
            "failure": False,
            "length": 0.0,
            "metrics": {},
        }
        did_break = False
        current_action = None  # Store current action for Spot tasks
        for step in tqdm(range(num_steps), desc=f"Episode {i + 1}/{num_episodes}", leave=False):
            curr_time = step * step_dt
            curr_state = np.concatenate([task.data.qpos, task.data.qvel])

            # advancing the simulation, controller, and planner as needed
            if step % loop_steps_per_plan_step == 0:  # planning step (updates spline)
                controller.update_action(curr_state, curr_time)
            if step % loop_steps_per_ctrl_step == 0:  # control step (updates control from spline)
                current_action = controller.action(curr_time)
                # For Spot tasks, we don't set ctrl directly - it's handled in the sim step
                if not is_spot_task:
                    task.data.ctrl[:] = current_action
            if step % loop_steps_per_viz_step == 0:  # visualization step (no-op here, but could be used for logging)
                curr_episode_results["qpos_traj"].append(np.array(task.data.qpos))
                mocap_pos = np.array(task.data.mocap_pos)  # (num_mocap, 3)
                mocap_quat = np.array(task.data.mocap_quat)  # (num_mocap, 4)
                curr_episode_results["mocap_pos_traj"].append(mocap_pos)
                curr_episode_results["mocap_quat_traj"].append(mocap_quat)
            if step % loop_steps_per_sim_step == 0:  # simulation step (updates simulation state)
                task.pre_sim_step()
                if is_spot_task and spot_sim_backend is not None and current_action is not None:
                    # Use Spot-specific simulation backend
                    spot_sim_backend.sim(sim_model, task.data, current_action)
                else:
                    # Use standard MuJoCo step
                    mj_step(sim_model, task.data)
                controller.system_metadata = task.get_sim_metadata()

                # compute the instantaneous reward in the simulation
                curr_sensor = np.asarray(task.data.sensordata)
                curr_control = np.asarray(task.data.ctrl[:])
                reward = task.reward(
                    curr_state[None, None, :],  # (B, T, dim)
                    curr_sensor[None, None, :],
                    curr_control[None, None, :],
                    task_config,
                )[0]  # returns (B,), so we index to get scalar
                curr_episode_results["rewards"].append(reward)

                # check for termination conditions and/or writing metrics
                metadata = {}
                has_success = task.success(sim_model, task.data, task_config, metadata=metadata)
                has_failure = task.failure(sim_model, task.data, task_config, metadata=metadata)
                metrics = task.compute_metrics(sim_model, task.data, task_config, metadata=metadata)

                for k, v in metrics.items():
                    if k not in curr_episode_results["metrics"]:
                        curr_episode_results["metrics"][k] = []
                    curr_episode_results["metrics"][k].append(v)

                # only perform post step AFTER measuring metrics, since that can change the state
                task.post_sim_step()

                if has_failure:  # failures take precedence over successes in rare case of tiebreaker
                    curr_episode_results["failure"] = True
                    curr_episode_results["length"] = curr_time
                    did_break = True
                    break
                elif has_success:
                    curr_episode_results["success"] = True
                    curr_episode_results["length"] = curr_time
                    did_break = True
                    break

        if not did_break:
            curr_episode_results["length"] = curr_time

        # collecting results for this episode
        curr_episode_results["rewards"] = np.asarray(curr_episode_results["rewards"])
        curr_episode_results["qpos_traj"] = np.asarray(curr_episode_results["qpos_traj"])
        if len(curr_episode_results["mocap_pos_traj"]) > 0:
            curr_episode_results["mocap_pos_traj"] = np.asarray(curr_episode_results["mocap_pos_traj"])
        else:
            curr_episode_results["mocap_pos_traj"] = np.empty((0,))
        if len(curr_episode_results["mocap_quat_traj"]) > 0:
            curr_episode_results["mocap_quat_traj"] = np.asarray(curr_episode_results["mocap_quat_traj"])
        else:
            curr_episode_results["mocap_quat_traj"] = np.empty((0,))
        benchmark_results.append(curr_episode_results)

    return benchmark_results


def summarize_benchmark_results(all_results: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    """Prints a summary of all benchmark results.

    Args:
        all_results: A dictionary with the following structure:
            {
                task_name_1: {
                    optimizer_name_1: [
                        {
                            "rewards": np.ndarray of shape (num_steps_1,),
                            "success": bool,
                            "failure": bool,
                            "length": float,
                            "metrics": dict of metric_name to list of float, each of length num_steps_1,
                        },  # results for episode 1...
                        ...
                    ],
                    optimizer_name_2: [ ... ],
                    ...
                },
                task_name_2: {
                    optimizer_name_1: [
                        {
                            "rewards": np.ndarray of shape (num_steps_2,),
                            "success": bool,
                            "failure": bool,
                            "length": float,
                            "metrics": dict of metric_name to list of float, each of length num_steps_2,
                        },  # results for episode 1...
                        ...
                    ],
                    optimizer_name_2: [ ... ],
                },
                ...
            }
    """
    all_tasks = get_registered_tasks()
    print("Summary of Benchmark Results")
    print("=" * 80)
    for task_name, task_results in all_results.items():
        print(f"Task: {task_name}")
        for optimizer_name, episode_results_for_task_opt_pair in task_results.items():
            print(f"  Optimizer: {optimizer_name}")

            all_rewards = []
            all_metrics = {}
            for _, episode_results in enumerate(episode_results_for_task_opt_pair):
                all_rewards.append(episode_results["rewards"])
                for k, v in episode_results["metrics"].items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].extend(v)

            # summarize
            task_cls = all_tasks[task_name][0]
            reduced_metrics = task_cls().reduce_metrics(all_metrics)

            all_rewards = np.concatenate(all_rewards)
            avg_reward = np.mean(all_rewards)
            print(f"    Average reward: {avg_reward:.4f}")
            for metric_name, metric_value in reduced_metrics.items():
                print(f"    {metric_name}: {metric_value:.4f}")
            num_successes = sum(1 for r in episode_results_for_task_opt_pair if r["success"])
            num_failures = sum(1 for r in episode_results_for_task_opt_pair if r["failure"])
            avg_length = np.mean([r["length"] for r in episode_results_for_task_opt_pair])
            print(f"    Successes: {num_successes}/{len(episode_results_for_task_opt_pair)}")
            print(f"    Failures: {num_failures}/{len(episode_results_for_task_opt_pair)}")
            print(f"    Average episode length: {avg_length:.4f} seconds")
    print("=" * 80)


def benchmark_multiple_tasks_and_optimizers(
    task_names: list[str] | None = None,
    optimizer_names: list[str] | None = None,
    num_episodes: int = 10,
    episode_length_s: float | list[float] = 30.0,
    min_dt: float = 0.0001,
    viz_dt: float = 0.02,
    verbose: bool = False,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Benchmarks multiple tasks and optimizers.

    Reports results for the Cartesian product of the provided task and optimizer names. If None are provided for one
    of the lists, all registered tasks or optimizers will be used.

    Args:
        task_names: List of task names to benchmark.
        optimizer_names: List of optimizer names to use.
        num_episodes: Number of episodes to run for each (task, optimizer) pair.
        episode_length_s: Maximum length of each episode (in seconds). If a list is provided, it must match the length
            of task_names.
        min_dt: Minimum timestep as a quantum unit for synchronization between simulation and controller.
        viz_dt: Timestep for visualization/logging (in seconds).
        verbose: Whether to print detailed configuration information.

    Returns:
        all_results: A dictionary mapping (task_name, optimizer_name) pairs to their benchmark results.
    """
    if task_names is None:
        task_names = list(get_registered_tasks().keys())
    if optimizer_names is None:
        optimizer_names = list(get_registered_optimizers().keys())
    if isinstance(episode_length_s, float):
        episode_length_s = [episode_length_s] * len(task_names)
    assert isinstance(episode_length_s, list), "episode_length_s must be a float or a list of floats."
    assert len(episode_length_s) == len(task_names), (
        "If episode_length_s is a list, it must match the length of task_names."
    )

    all_results = {}
    for i, task_name in enumerate(task_names):
        print(f"Benchmarking Task: {task_name}...")
        all_results[task_name] = {}
        for optimizer_name in optimizer_names:
            print(f"  Benchmarking Optimizer: {optimizer_name}...")
            results = benchmark_single_task_and_optimizer(
                task_name,
                optimizer_name,
                num_episodes=num_episodes,
                episode_length_s=episode_length_s[i],
                min_dt=min_dt,
                viz_dt=viz_dt,
                verbose=verbose,
            )
            all_results[task_name][optimizer_name] = results

    # save the benchmark results to a file
    filename = f"benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    with h5py.File(filename, "w") as f:
        # attributes
        f.attrs["viz_dt"] = viz_dt

        for task_name, task_result_dict in all_results.items():
            task_group = f.create_group(task_name)
            for optimizer_name, episodes in task_result_dict.items():
                opt_group = task_group.create_group(optimizer_name)
                for i, episode in enumerate(episodes):
                    ep_group = opt_group.create_group(f"episode_{i}")
                    ep_group.create_dataset("rewards", data=episode["rewards"])
                    ep_group.create_dataset("qpos_traj", data=episode["qpos_traj"])
                    ep_group.create_dataset("mocap_pos_traj", data=episode["mocap_pos_traj"])
                    ep_group.create_dataset("mocap_quat_traj", data=episode["mocap_quat_traj"])
                    ep_group.attrs["success"] = episode["success"]
                    ep_group.attrs["failure"] = episode["failure"]
                    ep_group.attrs["length"] = episode["length"]
                    metrics_group = ep_group.create_group("metrics")
                    for metric_name, metric_values in episode["metrics"].items():
                        metrics_group.create_dataset(metric_name, data=np.asarray(metric_values))

    # print a summary of the results
    summarize_benchmark_results(all_results)
    print_msg = dedent(
        f"""
        Benchmark results saved to '{filename}'.

        To visualize, run:
            python scripts/visualize_benchmark_results.py {filename}
        """
    )
    print(print_msg)
    return all_results


if __name__ == "__main__":
    benchmark_multiple_tasks_and_optimizers(
        task_names=[
            # "cylinder_push",
            # "cartpole",
            # "fr3_pick",
            # "g1_manipulation",
            # "g1_stand",
            # "go2_walk",
            # "leap_cube",
            # "leap_cube_down",
            # "walker",
            "spot_yellow_chair",
            # "spot_yellow_chair_ramp",
        ],
        optimizer_names=None,
        num_episodes=10,
        episode_length_s=[
            # 30.0,  # cylinder_push
            # 10.0,  # cartpole
            # 30.0,  # fr3_pick
            # 30.0,  # g1_manipulation
            # 30.0,  # g1_stand
            # 30.0,  # go2_walk
            # 60.0,  # leap_cube
            # 60.0,  # leap_cube_down
            # 10.0,  # walker
            30.0,  # spot_yellow_chair
            # 120.0,  # spot_yellow_chair_ramp
        ],
        viz_dt=0.02,
    )
