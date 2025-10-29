# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from judo.simulation.base import Simulation
from judo.simulation.mj_simulation import MJSimulation

simulation_registry = {
    "mujoco": MJSimulation,
}


def get_simulation_backend(simulation_backend: str) -> type:
    """Get the simulation class for a given backend."""
    return simulation_registry[simulation_backend]


__all__ = [
    "Simulation",
    "MJSimulation",
]
