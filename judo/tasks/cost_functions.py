# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import numpy as np


def smooth_l1_norm(input: np.ndarray, p: float) -> np.ndarray:
    """Computes the smooth L1 norm of the input."""
    return np.sqrt(np.power(input, 2) + p**2) - p


def quadratic_norm(input: np.ndarray) -> np.ndarray:
    """Computes the quadratic norm of the input."""
    return 0.5 * np.square(input).sum(-1)
