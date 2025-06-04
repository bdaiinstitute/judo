# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import field
from typing import Sequence

import numpy as np

IntOrFloat = int | float


def np_1d_field(
    array: np.ndarray,
    names: list[str] | None = None,
    mins: list[IntOrFloat] | IntOrFloat | None = None,
    maxs: list[IntOrFloat] | IntOrFloat | None = None,
    steps: list[IntOrFloat] | IntOrFloat | None = None,
    vis_name: str | None = None,
    xyz_vis_indices: Sequence[int | None] | None = None,
    xyz_vis_defaults: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Create a dataclass field with a default value of a 1D numpy array.

    For all list arguments, the length of the list must match the length of the array.
    If the length of the list is 1, the value will be broadcasted to the length of the array.

    Args:
        array: The default value of the field.
        names: The names of the array elements for visualization.
        mins: The minimum value of the field.
        maxs: The maximum value of the field.
        steps: The step size of the field.
        vis_name: The name of the visualization.
        xyz_vis_indices: The indices of the array elements to visualize in 3D.
        xyz_vis_defaults: The default values for the visualization.

    Returns:
        A dataclass field with the specified default value and metadata.

    Raises:
        ValueError: If the array is not 1D or if the lengths of the list arguments do not match the length of the array.
    """
    if not array.ndim == 1:
        raise ValueError("The array must be 1D.")

    # create lists
    name_list = names if names is not None else [f"{i}" for i in range(array.size)]
    if isinstance(mins, (int, float)):
        min_list = [mins] * array.size
    elif isinstance(mins, list):
        assert len(mins) == array.size, "The length of mins must match the length of the array."
        min_list = mins
    else:
        min_list = list(0.5 * array)

    if isinstance(maxs, (int, float)):
        max_list = [maxs] * array.size
    elif isinstance(maxs, list):
        assert len(maxs) == array.size, "The length of maxs must match the length of the array."
        max_list = maxs
    else:
        max_list = list(1.5 * array)

    if isinstance(steps, (int, float)):
        step_list = [steps] * array.size
    elif isinstance(steps, list):
        assert len(steps) == array.size, "The length of steps must match the length of the array."
        step_list = steps
    else:
        step_list = list((np.array(max_list) - np.array(min_list)) / 100.0)

    # process the visualization specification
    if vis_name is not None and xyz_vis_indices is not None:
        assert len(xyz_vis_indices) == 3, "xyz_vis_indices must be a sequence of length 3."
        assert all(i is None or isinstance(i, int) for i in xyz_vis_indices), (
            "xyz_vis_indices must be a list of integers or None."
        )
        assert all(i is None or 0 <= i < array.size for i in xyz_vis_indices), (
            "xyz_vis_indices must be a list of indices within the range of the array size."
        )

        # create the visualization metadata
        vis = {
            "name": vis_name,
            "xyz_vis_indices": xyz_vis_indices,
            "xyz_vis_defaults": xyz_vis_defaults,
        }
    else:
        vis = None

    # create field
    metadata = {
        "ui_array_config": {
            "names": name_list,
            "mins": min_list,
            "maxs": max_list,
            "steps": step_list,
            "vis": vis,
        }
    }
    # copy the array to avoid mutability issues, especially when resetting the associated config
    return field(default_factory=lambda: array.copy(), metadata=metadata)
