#!/usr/bin/env bash
# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
#
# Repair a judo-rai wheel: bundle shared libs with auditwheel, then
# also bundle libstdc++.so.6 for libonnxruntime (auditwheel skips
# whitelisted libs like libstdc++, but the bundled onnxruntime may
# need a newer version than the target system provides).

set -euo pipefail

DIST_DIR="${1:-dist}"
OUT_DIR="${2:-wheelhouse}"
PLAT_TAG="${3:-}"  # optional, e.g. manylinux_2_35_x86_64

# Find libmujoco SONAME to exclude (provided by pip mujoco at runtime)
MUJOCO_SO=$(python -c "
import mujoco, os
d = os.path.dirname(mujoco.__file__)
print(next(f for f in os.listdir(d) if f.startswith('libmujoco.so.')))
")

echo "==> Excluding $MUJOCO_SO (provided by mujoco pip package)"

# Step 1: auditwheel repair
PLAT_ARG=""
if [ -n "$PLAT_TAG" ]; then
    PLAT_ARG="--plat $PLAT_TAG"
fi
auditwheel repair --exclude "$MUJOCO_SO" $PLAT_ARG -w "$OUT_DIR" "$DIST_DIR"/*.whl

# Step 2: Bundle libstdc++ into the wheel for libonnxruntime
WHL=$(realpath "$OUT_DIR"/*.whl)
echo "==> Bundling libstdc++.so.6 into $WHL"

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

unzip -q "$WHL" -d "$WORK"

LIBS_DIR=$(find "$WORK" -type d -name 'judo_rai.libs' | head -1)
if [ -z "$LIBS_DIR" ]; then
    echo "No judo_rai.libs/ found — nothing to patch"
    exit 0
fi

# Find the bundled libonnxruntime
ORT_SO=$(find "$LIBS_DIR" -name 'libonnxruntime-*.so.*' | head -1)
if [ -z "$ORT_SO" ]; then
    echo "No bundled libonnxruntime found — skipping libstdc++ bundling"
    exit 0
fi

# Check if libonnxruntime needs libstdc++
if ! readelf -d "$ORT_SO" | grep -q 'libstdc++'; then
    echo "libonnxruntime does not link libstdc++ — skipping"
    exit 0
fi

# Find the libstdc++ that libonnxruntime currently resolves to
LIBSTDCXX_PATH=$(python -c "
import ctypes.util, os, subprocess
# Find via the same search path the .so would use
result = subprocess.run(['ldd', '$ORT_SO'], capture_output=True, text=True)
for line in result.stdout.splitlines():
    if 'libstdc++' in line and '=>' in line:
        path = line.split('=>')[1].strip().split()[0]
        print(path)
        break
")

if [ -z "$LIBSTDCXX_PATH" ] || [ ! -f "$LIBSTDCXX_PATH" ]; then
    # Fallback: use CONDA_PREFIX
    LIBSTDCXX_PATH="$CONDA_PREFIX/lib/libstdc++.so.6"
fi

echo "==> Using libstdc++ from: $LIBSTDCXX_PATH"

# Generate hash-based name (matching auditwheel convention)
HASH=$(sha256sum "$LIBSTDCXX_PATH" | cut -c1-8)
RENAMED="libstdc++-${HASH}.so.6"

# Copy into judo_rai.libs/
cp "$LIBSTDCXX_PATH" "$LIBS_DIR/$RENAMED"
chmod 755 "$LIBS_DIR/$RENAMED"

# Patch libonnxruntime to reference the renamed libstdc++
patchelf --replace-needed libstdc++.so.6 "$RENAMED" "$ORT_SO"

# Also patch libgomp if it references libstdc++
for so in "$LIBS_DIR"/libgomp-*.so.*; do
    if [ -f "$so" ] && readelf -d "$so" 2>/dev/null | grep -q 'libstdc++'; then
        patchelf --replace-needed libstdc++.so.6 "$RENAMED" "$so"
        echo "==> Patched $(basename "$so")"
    fi
done

echo "==> Bundled $RENAMED into $(basename "$LIBS_DIR")/"

# Update RECORD file
RECORD=$(find "$WORK" -name RECORD -path '*.dist-info/*')
if [ -n "$RECORD" ]; then
    # Remove old RECORD entry for itself, regenerate
    # wheel format: path,hash,size
    REL_LIBS=$(echo "$LIBS_DIR" | sed "s|$WORK/||")
    FILESIZE=$(stat -c%s "$LIBS_DIR/$RENAMED")
    FILEHASH=$(python -c "
import hashlib, base64
with open('$LIBS_DIR/$RENAMED', 'rb') as f:
    h = hashlib.sha256(f.read())
print('sha256=' + base64.urlsafe_b64encode(h.digest()).rstrip(b'=').decode())
")
    echo "$REL_LIBS/$RENAMED,$FILEHASH,$FILESIZE" >> "$RECORD"
fi

# Repack wheel
rm "$WHL"
cd "$WORK"
zip -q -r "$WHL" .
cd -

echo "==> Repaired wheel: $WHL"
echo "==> Contents of judo_rai.libs/:"
unzip -l "$WHL" | grep 'judo_rai.libs/'
