#!/usr/bin/env bash
# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
#
# Repair a judo-rai wheel: set CPython ABI tag, then use auditwheel to
# bundle non-system shared libs (libonnxruntime, libgomp) and set the
# manylinux platform tag.
#
# Usage: repair_wheel.sh <input.whl> [OUT_DIR] [PLAT_TAG] [PYTHON_TAG]
#   input.whl   - path to the wheel file to repair
#   OUT_DIR     - output directory (default: wheelhouse)
#   PLAT_TAG    - manylinux platform tag (default: auto-detect)
#   PYTHON_TAG  - optional CPython tag, e.g. cp311 (sets cpXYZ-cpXYZ ABI tag)

set -euo pipefail

INPUT_WHL="${1:?Usage: repair_wheel.sh <input.whl> [OUT_DIR] [PLAT_TAG] [PYTHON_TAG]}"
OUT_DIR="${2:-wheelhouse}"
PLAT_TAG="${3:-}"
PYTHON_TAG="${4:-}"

SRC_WHL="$(realpath "$INPUT_WHL")"
STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT

# Step 0: Set Python/ABI tag if specified (the .so is CPython-version-specific)
cp "$SRC_WHL" "$STAGE/"
WORK_WHL="$STAGE/$(basename "$SRC_WHL")"

if [ -n "$PYTHON_TAG" ]; then
    echo "==> Setting Python tag: ${PYTHON_TAG}-${PYTHON_TAG}"
    NEW_NAME=$(wheel tags --python-tag "$PYTHON_TAG" --abi-tag "$PYTHON_TAG" --remove "$WORK_WHL")
    WORK_WHL="$STAGE/$NEW_NAME"
fi

# Find libmujoco SONAME to exclude (provided by pip mujoco at runtime)
MUJOCO_SO=$(python -c "
import mujoco, os
d = os.path.dirname(mujoco.__file__)
print(next(f for f in os.listdir(d) if f.startswith('libmujoco.so.')))
")
echo "==> Excluding $MUJOCO_SO (provided by mujoco pip package)"

# Step 1: auditwheel repair
mkdir -p "$OUT_DIR"
PLAT_ARG=""
if [ -n "$PLAT_TAG" ]; then
    PLAT_ARG="--plat $PLAT_TAG"
fi
auditwheel repair --exclude "$MUJOCO_SO" $PLAT_ARG -w "$OUT_DIR" "$WORK_WHL"
