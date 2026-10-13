#!/usr/bin/env bash
# write_provenance.sh
# Append one provenance block (date, host, git commit, dirty flag, run settings) to a file.
#
# Usage:
#   bash write_provenance.sh <file> "<settings line>"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_FILE="$1"
SETTINGS="$2"

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "git_dirty=$([[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no -- hybrid_diffusion vla_core robosuite_mujoco_patch.py sitecustomize.py)" ]] && echo 1 || echo 0)"
  echo "$SETTINGS"
  echo
} >> "$OUT_FILE"
