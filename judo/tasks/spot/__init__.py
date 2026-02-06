# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Spot locomotion and manipulation tasks."""

from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_tire_upright import SpotTireUpright, SpotTireUprightConfig

__all__ = [
    "SpotBase",
    "SpotBaseConfig",
    "SpotTireUpright",
    "SpotTireUprightConfig",
]
