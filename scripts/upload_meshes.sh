#!/usr/bin/env bash
# Upload judo/models/meshes/ as meshes.zip to the latest GitHub release.
#
# Usage:
#   ./scripts/upload_meshes.sh              # upload to latest release
#   ./scripts/upload_meshes.sh 0.0.6        # upload to specific tag
#
# Requires: gh (GitHub CLI), authenticated via `gh auth login`

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MESHES_DIR="$REPO_ROOT/judo/models/meshes"
ZIP_FILE="$REPO_ROOT/meshes.zip"

if [ ! -d "$MESHES_DIR" ]; then
  echo "Error: meshes directory not found at $MESHES_DIR"
  echo "Make sure mesh files are present locally before uploading."
  exit 1
fi

if ! command -v gh &>/dev/null; then
  echo "Error: gh (GitHub CLI) is required. Install from https://cli.github.com"
  exit 1
fi

# Determine target tag
if [ $# -ge 1 ]; then
  TAG="$1"
else
  TAG=$(gh release list --repo bdaiinstitute/judo --limit 1 --json tagName -q '.[0].tagName')
  if [ -z "$TAG" ]; then
    echo "Error: no releases found. Specify a tag explicitly."
    exit 1
  fi
fi

echo "Zipping $MESHES_DIR ..."
(cd "$REPO_ROOT/judo/models" && zip -qr "$ZIP_FILE" meshes/)

SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "Created meshes.zip ($SIZE)"

echo "Uploading to release $TAG ..."
gh release upload "$TAG" "$ZIP_FILE" --repo bdaiinstitute/judo --clobber

rm -f "$ZIP_FILE"
echo "Done. meshes.zip uploaded to release $TAG."
