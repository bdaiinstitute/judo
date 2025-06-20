# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import numpy as np


def safe_normalize_axis(axis: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safely normalizes a batch of 3D axis vectors, avoiding division by zero.

    If the norm of an axis is less than `eps`, it defaults to [1, 0, 0].

    Args:
        axis: The unnormalized axis vectors. Shape = (..., 3).
        eps: Small threshold to avoid division by zero.

    Returns:
        Normalized axis vectors. Shape = (..., 3).
    """
    norm = np.linalg.norm(axis, axis=-1)
    small_angle_mask = norm < eps
    safe_norm = np.where(small_angle_mask, 1.0, norm)
    normalized = axis / safe_norm[..., None]
    return np.where(small_angle_mask[..., None], np.array([1.0, 0.0, 0.0]), normalized)


def quat_inv(u: np.ndarray) -> np.ndarray:
    """Inverts a quaternion in a way that can broadcast cleanly.

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).

    Returns:
        result: quat in wxyz order. Shape=(*u_dims, 4).
    """
    return u * np.array([1, -1, -1, -1])


def quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions in a way that can broadcast cleanly.

    The leading dimensions of u and v do not have to match - only the trailing dims.

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).
        v: quat in wxyz order. Shape=(*v_dims, 4).

    Returns:
        result: quat in wxyz order. Shape=(*(u_dims or v_dims), 4). The longer leading dims of u or v are used.
    """
    w = u[..., 0] * v[..., 0] - u[..., 1] * v[..., 1] - u[..., 2] * v[..., 2] - u[..., 3] * v[..., 3]
    x = u[..., 0] * v[..., 1] + u[..., 1] * v[..., 0] + u[..., 2] * v[..., 3] - u[..., 3] * v[..., 2]
    y = u[..., 0] * v[..., 2] - u[..., 1] * v[..., 3] + u[..., 2] * v[..., 0] + u[..., 3] * v[..., 1]
    z = u[..., 0] * v[..., 3] + u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1] + u[..., 3] * v[..., 0]
    return np.stack([w, x, y, z], axis=-1)


def quat_diff(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Computes the 'quaternion difference' between two quaternions: u^* \otimes v.

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).
        v: quat in wxyz order. Shape=(*v_dims, 4).

    Returns:
        result: quat in wxyz order. Shape=(*(u_dims or v_dims), 4). The longer leading dims of u or v are used.
    """
    return quat_mul(quat_inv(u), v)


def axis_angle_diff(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Computes the 'axis-angle difference' between two quaternions: 2 * vec(u^* \otimes v).

    Args:
        u: quat in wxyz order. Shape=(*u_dims, 4).
        v: quat in wxyz order. Shape=(*v_dims, 4).

    Returns:
        angle: The angle of rotation in radians. Shape=(*(u_dims or v_dims),).
        axis: The axis of rotation. Shape=(*(u_dims or v_dims), 3).
    """
    diff = quat_diff(u, v)
    axis = diff[..., 1:]
    sin_a_2 = np.linalg.norm(axis, axis=-1)

    # handle division by zero
    axis = safe_normalize_axis(axis, eps=1e-6)
    angle = 2.0 * np.arctan2(sin_a_2, diff[..., 0])

    # use correct angle comparison logic
    mask = angle > np.pi
    angle = np.where(mask, 2 * np.pi - angle, angle)
    axis = np.where(mask[..., None], -axis, axis)
    return angle, axis


def quat_diff_so3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Computes the 'quaternion difference' between two quaternions and then takes the Log map."""
    diff = quat_diff(u, v)
    axis = diff[..., 1:]
    sin_a_2 = np.linalg.norm(axis, axis=-1)
    axis = safe_normalize_axis(axis, eps=1e-6)
    speed = 2.0 * np.arctan2(sin_a_2, diff[..., 0])
    speed = np.where(speed > np.pi, speed - 2 * np.pi, speed)
    output = axis * speed[..., None]
    return output


def quat_vel(u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    r"""Estimates the angular velocity between two quaternions.

    Uses the formula
        omega = (2.0 * u^{-1} * [(v - u) / dt])[1:],
    where [1:] denotes taking the vector part only.

    This is derived from the general differential formula
        omega = 2 * vec(u^* \otimes \dot{q}).

    Source: mariogc.com/post/angular-velocity-quaternions
    """
    return 2.0 * quat_mul(quat_inv(u), (v - u) / dt)[..., 1:]
