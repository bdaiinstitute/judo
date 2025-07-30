# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import importlib
import sys
from types import ModuleType
from typing import Any, cast


def _apply_patch(mod: ModuleType, value: int) -> ModuleType:
    """Apply the monkey-patch to the given module.

    Write the new value into the target module, then propagate it to every module that already grabbed its own copy via
    `from mujoco_warp._src.io import MJ_CCD_ITERATIONS`.
    """
    cast(Any, mod).MJ_CCD_ITERATIONS = value

    for m in list(sys.modules.values()):
        if m and "MJ_CCD_ITERATIONS" in getattr(m, "__dict__", {}):
            m.__dict__["MJ_CCD_ITERATIONS"] = value
    return mod


def patch_mj_ccd_iterations(value: int = 32) -> None:
    """Monkey-patch MJ_CCD_ITERATIONS globally for the rest of the session.

    Call this exactly once with the desired value. It:
        1. Updates mujoco_warp._src.io.MJ_CCD_ITERATIONS.
        2. Rewrites the constant in every already-imported module that copied it with a 'from … import' statement.
        3. Wraps importlib.reload so any future reload of the io module keeps the patched value.
    """
    io_mod = importlib.import_module("mujoco_warp._src.io")
    _apply_patch(io_mod, value)

    # Ensure future `importlib.reload(mujoco_warp._src.io)` keeps the patch
    _orig_reload = importlib.reload

    def _reload_with_patch(module: ModuleType, *args: Any, **kwargs: Any) -> ModuleType:
        """Reload the module and apply the patch."""
        mod = _orig_reload(module, *args, **kwargs)
        if module.__name__ == "mujoco_warp._src.io":
            _apply_patch(mod, value)
        return mod

    importlib.reload = _reload_with_patch
