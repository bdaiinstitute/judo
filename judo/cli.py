# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from pathlib import Path

import hydra
from dora_utils.launch.run import run
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

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


@hydra.main(config_path=str(CONFIG_PATH), config_name="judo_default", version_base="1.3")
def main_app(cfg: DictConfig) -> None:
    """Main function to run judo via a hydra configuration yaml file."""
    run(cfg)


def app() -> None:
    """Entry point for the judo CLI."""
    # we store judo_default in the config store so that custom configs located outside of judo can inherit from it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="judo_default")
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
