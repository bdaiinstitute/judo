# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Shared utility functions for Spot tasks."""

import numpy as np


def apply_quat_to_vec(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply quaternion rotation to a vector.

    This matches dexterity.utils.math.apply_quat_to_vec from starfish.

    Args:
        quat: Quaternion in (w, x, y, z) format, shape (..., 4).
        vec: Vector to rotate, shape (..., 3).

    Returns:
        Rotated vector, same shape as vec.
    """
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + w * t + np.cross(xyz, t)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (w, x, y, z) format.

    Args:
        q1: First quaternion, shape (..., 4).
        q2: Second quaternion, shape (..., 4).

    Returns:
        Product quaternion, shape (..., 4).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
