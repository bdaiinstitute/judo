# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import contextlib
from typing import Callable, Generator

import numpy as np
import pytest


@pytest.fixture(scope="session")
def temp_np_seed() -> Callable[[int], contextlib._GeneratorContextManager[None]]:
    """Fixture to temporarily set the NumPy random seed for tests."""

    @contextlib.contextmanager
    def _temp_np_seed(seed: int) -> Generator[None, None, None]:
        """Context manager to temporarily set the NumPy random seed."""
        state = np.random.get_state()  # save state before context manager
        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)  # restore state after context manager

    return _temp_np_seed
