# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from pathlib import Path

from judo.utils.patch import patch_mj_ccd_iterations

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_ROOT / "models"

# TEMPORARY: Patch MJ_CCD_ITERATIONS globally for the session
# Once this PR is resolved, remove the monkey patch logic: https://github.com/google-deepmind/mujoco_warp/issues/456
patch_mj_ccd_iterations(32)
