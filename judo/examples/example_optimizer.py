# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass

from judo.optimizers import CrossEntropyMethod, CrossEntropyMethodConfig


@dataclass
class MyCrossEntropyMethodConfig(CrossEntropyMethodConfig):
    """Dummy optimizer config."""

    my_custom_param: int = 42


class MyCrossEntropyMethod(CrossEntropyMethod):
    """Dummy custom optimizer."""
