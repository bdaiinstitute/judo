# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import logging
import colorlog
import warnings
from pathlib import Path
import sys
from typing import Any, Iterable
from functools import partial
import signal
import multiprocessing as mp

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from judo.app.data.visualization_data import VisualizationData
from judo.app.data.simulation_data import SimulationData
from judo.app.data.controller_data import ControllerData
from judo.logging_util import setup_logger
from judo.unlocked import Node
from judo.unlocked.policy import Latest, Lossless, Optional
from judo.unlocked.schedule import Threaded

# suppress annoying warning from hydra
warnings.filterwarnings(
    "ignore",
    message=r".*Defaults list is missing `_self_`.*",
    category=UserWarning,
    module="hydra._internal.defaults_list",
)

CONFIG_PATH = (Path(__file__).parent / "configs").resolve()

# ### #
# app #
# ### #

class StopAll:
    """Terminate all schedulers."""

    def __init__(self, stop_it: Iterable) -> None:
        self._stop = list(stop_it)

    def __call__(self, signum: Any, _: Any) -> None:
        for s in self._stop:
            s.set()

def clear_dora_cfg(cfg: DictConfig) -> dict:
    exclude_keys = ["_target_", "node_id", "max_workers"]
    return {k: v for k, v in cfg.items() if k not in exclude_keys}

def log_str_sink(level, msg):
    def log_fn(l: int, m: str, d: Any) -> None:
        logging.log(l, m, d)
    return partial(log_fn, level, msg + "%s")

@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_dora_default", version_base="1.3")
def main_app(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    setup_logger(log_level=logging.INFO, include_thread=True)

    controller_config = clear_dora_cfg(cfg.node_definitions.controller)
    logging.debug("Got controller config:\n%s", controller_config)
    controller = ControllerData(**controller_config)

    simulation_config = clear_dora_cfg(cfg.node_definitions.simulation)
    logging.debug("Got simulation config:\n%s", simulation_config)
    simulation = SimulationData(**simulation_config)

    visualization_config = clear_dora_cfg(cfg.node_definitions.visualization)
    logging.debug("Got visualization config:\n%s", visualization_config)
    visualization = VisualizationData(**visualization_config)

    controller_update_task = Node("cont_update_task", controller.update_task)
    controller_reset_task = Node("cont_reset_task", controller.reset_task)
    controller_update_optimizer = Node("cont_update_optimizer", controller.update_optimizer)
    controller_update_optimizer_config = Node("cont_update_optimizer_config", controller.update_optimizer_config)
    controller_update_controller_config = Node("cont_update_controller_config", controller.update_controller_config)
    controller_update_task_config = Node("cont_update_task_config", controller.update_task_config)
    controller_step = Node("cont_step", controller.step)

    simulation_pause = Node("sim_pause", simulation.pause)
    simulation_reset_task = Node("sim_reset_task", simulation.reset_task)
    simulation_update_task = Node("sim_update_task", simulation.update_task)
    simulation_step = Node("sim_step", simulation.step, frequency=60)

    visualization_sim_pause = Node("vis_sim_pause", visualization.write_sim_pause, frequency=1000)
    visualization_task_name = Node("vis_task_name", visualization.write_task, frequency=1)
    visualization_task_reset = Node("vis_task_reset", visualization.write_task_reset, frequency=10)
    visualization_optimizer_name = Node("vis_optimizer_name", visualization.write_optimizer, frequency=1)
    visualization_controller_config = Node("vis_controller_config", visualization.write_controller_config, frequency=100)
    visualization_optimizer_config = Node("vis_optimizer_config", visualization.write_optimizer_config, frequency=100)
    visualization_task_config = Node("vis_task_config", visualization.write_task_config, frequency=100)
    visualization_update_states = Node("vis_update_states", visualization.update_states)
    visualization_update_traces = Node("vis_update_traces", visualization.update_traces)
    visualization_update_plan_time = Node("vis_update_plan_time", visualization.update_plan_time)

    controller_update_task.input_stage.connect(0, visualization_task_name.output_stage[0], Lossless())
    controller_reset_task.input_stage.connect(0, visualization_task_reset.output_stage[0], Lossless())
    controller_update_optimizer.input_stage.connect(0, visualization_optimizer_name.output_stage[0], Lossless())
    controller_update_optimizer_config.input_stage.connect(0, visualization_optimizer_config.output_stage[0], Lossless())
    controller_update_controller_config.input_stage.connect(0, visualization_controller_config.output_stage[0], Lossless())
    controller_update_task_config.input_stage.connect(0, visualization_task_config.output_stage[0], Lossless())
    controller_step.input_stage.connect(0, simulation_step.output_stage[0], Latest())

    simulation_pause.input_stage.connect(0, visualization_sim_pause.output_stage[0], Lossless())
    simulation_reset_task.input_stage.connect(0, visualization_task_reset.output_stage[0], Lossless())
    simulation_update_task.input_stage.connect(0, visualization_task_name.output_stage[0], Lossless())
    simulation_step.input_stage.connect(0, controller_step.output_stage[0], Optional())

    visualization_update_states.input_stage.connect(0, simulation_step.output_stage[0], Latest())
    visualization_update_traces.input_stage.connect(0, controller_step.output_stage[1], Latest())
    visualization_update_plan_time.input_stage.connect(0, controller_step.output_stage[2], Latest())

    log_str = []
    # log_str.append(Node("log_str_1", log_str_sink(logging.DEBUG, "Got from sim ")))
    # log_str[-1].input_stage.connect(0, simulation_step.output_stage[0], Latest())
    #
    # log_str.append(Node("log_str_2", log_str_sink(logging.DEBUG, "Got from cont")))
    # log_str[-1].input_stage.connect(0, controller_step.output_stage[0], Latest())
    #
    # log_str.append(Node("log_str_3", log_str_sink(logging.WARN, "Got from task name ")))
    # log_str[-1].input_stage.connect(0, visualization_task_name.output_stage[0], Latest())
    #
    # log_str.append(Node("log_str_4", log_str_sink(logging.WARN, "Got from optimizer name ")))
    # log_str[-1].input_stage.connect(0, visualization_optimizer_name.output_stage[0], Latest())

    controller_scheduler = Threaded(
        "cont_sch", (
            controller_update_task,
            controller_reset_task,
            controller_update_optimizer,
            controller_update_optimizer_config,
            controller_update_controller_config,
            controller_update_task_config,
            controller_step,
        ),
        mp.Event())
    simulation_scheduler = Threaded(
        "sim_sch", (
            simulation_pause,
            simulation_reset_task,
            simulation_update_task,
            simulation_step,
        ),
        mp.Event())
    visualization_scheduler = Threaded(
        "vis_sch", (
            visualization_sim_pause,
            visualization_task_name,
            visualization_task_reset,
            visualization_optimizer_name,
            visualization_controller_config,
            visualization_optimizer_config,
            visualization_task_config,
            visualization_update_states,
            visualization_update_traces,
            visualization_update_plan_time,
            *log_str,
        ),
        mp.Event())
    controller_scheduler.start()
    simulation_scheduler.start()
    # visualization_scheduler.start()
    signal.signal(signal.SIGINT, StopAll(
        (
            controller_scheduler._stop,
            simulation_scheduler._stop,
            visualization_scheduler._stop,
        )
    ))
    visualization_scheduler.spin()
    controller_scheduler.join()
    simulation_scheduler.join()
    # visualization_scheduler.join()


def app() -> None:
    """Entry point for the judo CLI."""
    # we store judo_default in the config store so that custom configs located outside of judo can inherit from it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="judo_dora_default")
        cs.store("judo", default_cfg)  # don't name this judo_default so it doesn't clash
    main_app()


# ######### #
# benchmark #
# ######### #


@hydra.main(config_path=str(CONFIG_PATH), config_name="benchmark_default", version_base="1.3")
def main_benchmark(cfg: DictConfig) -> None:
    """Benchmarking hydra call."""
    run(cfg)


def benchmark() -> None:
    """Entry point for benchmarking."""
    # we store benchmark_default in the config store so that custom configs located outside of judo can inherit from it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="benchmark_default")
        cs.store("benchmark", default_cfg)  # don't name this benchmark_default so it doesn't clash
    main_benchmark()
