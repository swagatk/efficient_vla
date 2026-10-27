#!/usr/bin/env bash
# run_step3_noise_sac.sh
# Step 3 of noise_reachability/README.md: train a noise policy with SAC on each task (init states 10-49), then
# evaluate it and the base policy on init states 0-9 x seeds 0-2, paired episode by episode. The command lines
# below are the record of how reported numbers were produced.
#
# Usage (from anywhere):
#   STAGES="train eval" bash noise_reachability/run_step3_noise_sac.sh    # tasks 7 and 0, one after the other
#   TASKS=7 bash noise_reachability/run_step3_noise_sac.sh                # one task (the notebook runs one per task)
#   SMOKE=1 bash noise_reachability/run_step3_noise_sac.sh                # task 7: 3 training episodes, 1 eval episode each
# Override TASKS, STAGES, OUT_ROOT or EXTRA_ARGS (passed to train_noise_sac.py in both stages).
#
# Safe to re-run: training resumes from the last checkpoint and evaluation skips finished episodes.
# Outputs: $OUT_ROOT/task_<t>/ (checkpoint.pt, replay_part_*.npz, train_log.csv, train.log, provenance.jsonl, eval/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
# The repo root must be importable at start-up so sitecustomize loads robosuite_mujoco_patch (FAIL-003).
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" PYTHONFAULTHANDLER=1

STAGES="${STAGES:-train eval}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TASKS="${TASKS:-7}"
  OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/noise_sac/smoke}"
  SETTINGS=(--num_episodes 3 --warmup_episodes 1 --batch_size 32 --updates_per_decision 2 --seeds 0 --init_states 0)
else
  TASKS="${TASKS:-7 0}"
  OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/noise_sac/run}"
  SETTINGS=()
fi
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
cd "$REPO_ROOT"

# Each call does a bounded amount of work and exits with 3 when more remains (FAIL-004); call again until 0.
# A crash (any other exit code) is retried from the last save, at most MAX_CRASHES times in a row.
MAX_CRASHES="${MAX_CRASHES:-3}"
run_until_done() {
  local log="$1"; shift
  local code crashes=0
  while true; do
    set +e
    "$PYTHON_BIN" -u "$SCRIPT_DIR/train_noise_sac.py" "$@" "${SETTINGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$log"
    code=${PIPESTATUS[0]}
    set -e
    if [[ $code -eq 0 ]]; then return 0; fi
    if [[ $code -eq 3 ]]; then
      crashes=0
    else
      crashes=$((crashes + 1))
      echo "[runner] $* exited with code $code (crash $crashes of $MAX_CRASHES in a row)" | tee -a "$log" >&2
      if [[ $crashes -ge $MAX_CRASHES ]]; then return "$code"; fi
    fi
    sleep 2  # stagger EGL process start-up (FAIL-006)
  done
}

for task in $TASKS; do
  out="$OUT_ROOT/task_$task"
  mkdir -p "$out"
  if [[ " $STAGES " == *" train "* ]]; then
    run_until_done "$out/train.log" --mode train --task_id "$task" --output_dir "$out"
  fi
  if [[ " $STAGES " == *" eval "* ]]; then
    run_until_done "$out/eval.log" --mode eval --task_id "$task" --output_dir "$out"
  fi
done
