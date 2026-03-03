# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Spot locomotion and manipulation tasks."""

from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_box_push import SpotBoxPush, SpotBoxPushConfig
from judo.tasks.spot.spot_navigate import SpotNavigate, SpotNavigateConfig
from judo.tasks.spot.spot_tire_roll import SpotTireRoll, SpotTireRollConfig
from judo.tasks.spot.spot_tire_upright import SpotTireUpright, SpotTireUprightConfig

__all__ = [
    "SpotBase",
    "SpotBaseConfig",
    "SpotBoxPush",
    "SpotBoxPushConfig",
    "SpotNavigate",
    "SpotNavigateConfig",
    "SpotTireRoll",
    "SpotTireRollConfig",
    "SpotTireUpright",
    "SpotTireUprightConfig",
]
