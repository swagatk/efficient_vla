#!/usr/bin/env bash
# run_base_noise_seeds.sh
# Base SmolVLA on one LIBERO-10 task (default: task 0) under several sampling-noise seeds, on the same
# 50 init states, followed by failure_overlap.py: do failures repeat on the same init states or move
# with the noise? No head is trained or loaded. See README.md for the hypothesis being tested.
#
# Usage:
#   bash run_base_noise_seeds.sh
#   TASK_ID=3 SEEDS="1 2 3 4 5" bash run_base_noise_seeds.sh
#   PARALLEL=1 bash run_base_noise_seeds.sh   # all seeds at once (about 2.7 GB GPU memory each)
#
# Safe to re-run after an interruption: per-episode records are reused.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"

export TASK_ID="${TASK_ID:-0}"
export EPISODES="${EPISODES:-50}"
SEEDS="${SEEDS:-1 2 3}"
# 1 = run every seed's evaluation as a concurrent process.
PARALLEL="${PARALLEL:-0}"
export OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/outputs/task${TASK_ID}_residual}"
mkdir -p "$OUT_ROOT"

bash "$SCRIPT_DIR/write_provenance.sh" "$OUT_ROOT/provenance_base_noise_seeds.txt" \
  "task_id=$TASK_ID seeds=$SEEDS episodes=$EPISODES parallel=$PARALLEL"

if [[ "$PARALLEL" == "1" ]]; then
  pids=()
  for seed in $SEEDS; do
    echo "=== seed $seed: eval_base started ($(date -Is)), log: $OUT_ROOT/eval_base_s${seed}.runner.log ==="
    SEED="$seed" STAGES="eval_base" bash "$SCRIPT_DIR/run_task3_residual.sh" >> "$OUT_ROOT/eval_base_s${seed}.runner.log" 2>&1 &
    pids+=("$!")
    sleep 5
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
    echo "=== seed $seed: eval_base ($(date -Is)) ==="
    SEED="$seed" STAGES="eval_base" bash "$SCRIPT_DIR/run_task3_residual.sh"
  done
fi

# -S: reads JSON only, must not import torch/MuJoCo through sitecustomize.
"$PYTHON_BIN" -S "$SCRIPT_DIR/failure_overlap.py" --out_root "$OUT_ROOT" --seeds $SEEDS --episodes "$EPISODES"
