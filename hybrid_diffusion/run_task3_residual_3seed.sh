#!/usr/bin/env bash
# run_task3_residual_3seed.sh
# 3-seed replication of the base vs residual comparison on one LIBERO-10 task (default: task 3).
# For each seed: train a residual head with that seed, then evaluate base and residual on the same
# 50 init states with that sampling-noise seed. See README.md for the hypothesis being tested.
#
# Usage:
#   bash run_task3_residual_3seed.sh
#   SEEDS="1 2" bash run_task3_residual_3seed.sh
#   PARALLEL=1 bash run_task3_residual_3seed.sh   # all evals at once (about 2.7 GB GPU memory each)
#
# Safe to re-run after an interruption: finished heads and per-episode records are reused.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"

export TASK_ID="${TASK_ID:-3}"
export ALPHA="${ALPHA:-1.0}"
export EPISODES="${EPISODES:-50}"
export EPOCHS="${EPOCHS:-300}"
export DIFF_HIDDEN_DIM="${DIFF_HIDDEN_DIM:-512}"
export DIFF_LAYERS="${DIFF_LAYERS:-6}"
export COND_NOISE_STD="${COND_NOISE_STD:-0.5}"
# Seed 0 is the exploratory run that suggested the effect, so the confirmation uses fresh seeds.
SEEDS="${SEEDS:-1 2 3}"
# 1 = train the heads, then run every seed's base and residual eval as concurrent processes.
PARALLEL="${PARALLEL:-0}"
export OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/outputs/task${TASK_ID}_residual}"
mkdir -p "$OUT_ROOT"
# Per-seed names are derived by run_task3_residual.sh; a fixed name would make the seeds collide.
unset RUN_NAME CHECKPOINT
RUN_PREFIX="h${DIFF_HIDDEN_DIM}_l${DIFF_LAYERS}_n${COND_NOISE_STD}_e${EPOCHS}"

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "git_dirty=$([[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no -- hybrid_diffusion vla_core robosuite_mujoco_patch.py sitecustomize.py)" ]] && echo 1 || echo 0)"
  echo "task_id=$TASK_ID alpha=$ALPHA seeds=$SEEDS episodes=$EPISODES run_prefix=$RUN_PREFIX parallel=$PARALLEL"
  echo
} >> "$OUT_ROOT/provenance_3seed.txt"

if [[ "$PARALLEL" == "1" ]]; then
  STAGES="cache" bash "$SCRIPT_DIR/run_task3_residual.sh"
  for seed in $SEEDS; do
    SEED="$seed" STAGES="train" bash "$SCRIPT_DIR/run_task3_residual.sh"
  done
  pids=()
  for seed in $SEEDS; do
    for stage in eval_base eval_residual; do
      echo "=== seed $seed: $stage started ($(date -Is)), log: $OUT_ROOT/${stage}_s${seed}.runner.log ==="
      SEED="$seed" STAGES="$stage" bash "$SCRIPT_DIR/run_task3_residual.sh" >> "$OUT_ROOT/${stage}_s${seed}.runner.log" 2>&1 &
      pids+=("$!")
      sleep 5
    done
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    echo "At least one eval process failed; see the *.runner.log files in $OUT_ROOT" >&2
    exit 1
  fi
else
  for seed in $SEEDS; do
    echo "=== seed $seed: train + eval_base + eval_residual ($(date -Is)) ==="
    SEED="$seed" STAGES="cache train eval_base eval_residual" bash "$SCRIPT_DIR/run_task3_residual.sh"
  done
fi

# -S: reads JSON only, must not import torch/MuJoCo through sitecustomize.
"$PYTHON_BIN" -S "$SCRIPT_DIR/aggregate_seeds.py" --out_root "$OUT_ROOT" --seeds $SEEDS --alpha "$ALPHA" \
  --run_prefix "$RUN_PREFIX" --episodes "$EPISODES"
