#!/usr/bin/env bash
# run_task3_residual_3seed.sh
# 3-seed replication of the base vs residual comparison on one LIBERO-10 task (default: task 3).
# For each seed: train a residual head with that seed, then evaluate base and residual on the same
# 50 init states with that sampling-noise seed. See README.md for the hypothesis being tested.
#
# Usage:
#   bash run_task3_residual_3seed.sh
#   SEEDS="1 2" bash run_task3_residual_3seed.sh
#
# Safe to re-run after an interruption: finished heads and per-episode records are reused.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"

export TASK_ID="${TASK_ID:-3}"
export ALPHA="${ALPHA:-1.0}"
# Seed 0 is the exploratory run that suggested the effect, so the confirmation uses fresh seeds.
SEEDS="${SEEDS:-1 2 3}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/outputs/task${TASK_ID}_residual}"
export OUT_ROOT
mkdir -p "$OUT_ROOT"

{
  echo "date=$(date -Is)"
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "git_dirty=$([[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no -- hybrid_diffusion vla_core robosuite_mujoco_patch.py sitecustomize.py)" ]] && echo 1 || echo 0)"
  echo "task_id=$TASK_ID alpha=$ALPHA seeds=$SEEDS"
  echo
} >> "$OUT_ROOT/provenance_3seed.txt"

for seed in $SEEDS; do
  echo "=== seed $seed: train + eval_base + eval_residual ($(date -Is)) ==="
  SEED="$seed" STAGES="cache train eval_base eval_residual" bash "$SCRIPT_DIR/run_task3_residual.sh"
done

# -S: reads JSON only, must not import torch/MuJoCo through sitecustomize.
"$PYTHON_BIN" -S "$SCRIPT_DIR/aggregate_seeds.py" --out_root "$OUT_ROOT" --seeds $SEEDS --alpha "$ALPHA"
