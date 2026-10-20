#!/usr/bin/env bash
# run_self_filtered_bc.sh
# Self-filtered BC on one LIBERO-10 task: collect base-policy rollouts on init states 10-49, train one LoRA
# adapter per (selection, seed), evaluate each on held-out init states 0-9 x eval seeds 0,1,2, paired with the
# base episodes. The command lines below are the record of how reported numbers were produced (README.md).
#
# Usage (from anywhere):
#   bash self_filtered_bc/run_self_filtered_bc.sh                                   # all stages, task 6
#   STAGES=collect bash self_filtered_bc/run_self_filtered_bc.sh
#   SELECTIONS="success all_matched" SEEDS="0 1 2" STAGES="train eval" bash self_filtered_bc/run_self_filtered_bc.sh
#   SMOKE=1 bash self_filtered_bc/run_self_filtered_bc.sh                           # 2 episodes, 5 steps, 1 eval
#
# Safe to re-run: collection and evaluation skip finished episodes. Training restarts from scratch.
# Outputs: $DATA_DIR (episodes/, collection.csv), $OUT_ROOT/<selection>_seed<s>/ (adapter/, train.log, eval/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
# The repo root must be importable at start-up so sitecustomize loads robosuite_mujoco_patch (FAIL-003).
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" PYTHONFAULTHANDLER=1

TASK_ID="${TASK_ID:-6}"
SEEDS="${SEEDS:-0}"
SELECTIONS="${SELECTIONS:-success}"
STAGES="${STAGES:-collect train eval}"
INIT_STATES="${INIT_STATES:-10-49}"
COLLECT_SEEDS="${COLLECT_SEEDS:-0-2}"
VAL_INIT_STATES="${VAL_INIT_STATES:-45-49}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2}"
EVAL_INIT_STATES="${EVAL_INIT_STATES:-0-9}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/outputs/self_filtered_bc/task_${TASK_ID}/data}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/self_filtered_bc/task_${TASK_ID}}"
# Base episodes of the residual-SAC harness check (same policy, settings and seeds); with no file the base runs.
BASE_CSV="${BASE_CSV:-$REPO_ROOT/week6/sac_residual_task6_pilot1/eval_base/per_episode_base.csv}"
if [[ "${SMOKE:-0}" == 1 ]]; then
  INIT_STATES=10-11 COLLECT_SEEDS=0 VAL_INIT_STATES="" TRAIN_STEPS=5 EVAL_SEEDS=0 EVAL_INIT_STATES=0 BASE_CSV=""
  DATA_DIR="$REPO_ROOT/outputs/self_filtered_bc_smoke/data" OUT_ROOT="$REPO_ROOT/outputs/self_filtered_bc_smoke"
  SELECTIONS=all
fi
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
MAIN="$SCRIPT_DIR/self_filtered_bc.py"
mkdir -p "$OUT_ROOT" "$DATA_DIR"
cd "$REPO_ROOT"

# Exit code 3 = more episodes to run in a fresh process (bounds simulator resources, FAIL-004).
run_until_done() {
  local log="$1"; shift
  local code
  while true; do
    set +e
    "$PYTHON_BIN" -u "$MAIN" --task_id "$TASK_ID" "$@" "${EXTRA[@]}" 2>&1 | tee -a "$log"
    code=${PIPESTATUS[0]}
    set -e
    if [[ $code -eq 0 ]]; then return 0; fi
    if [[ $code -ne 3 ]]; then echo "[runner] failed with exit code $code ($(date -Is))" | tee -a "$log" >&2; return "$code"; fi
    sleep 2  # stagger EGL process start-up (FAIL-006)
  done
}

if [[ " $STAGES " == *" collect "* ]]; then
  echo "=== collect task $TASK_ID ($(date -Is)) ===" | tee -a "$DATA_DIR/collect.log"
  run_until_done "$DATA_DIR/collect.log" --stage collect --data_dir "$DATA_DIR" --init_states "$INIT_STATES" \
    --collect_seeds "$COLLECT_SEEDS" --max_episodes_per_process 40
fi

for selection in $SELECTIONS; do
  for seed in $SEEDS; do
    dir="$OUT_ROOT/${selection}_seed${seed}"
    mkdir -p "$dir"
    if [[ " $STAGES " == *" train "* ]]; then
      echo "=== train $selection seed $seed ($(date -Is)) ===" | tee -a "$dir/train.log"
      run_until_done "$dir/train.log" --stage train --seed "$seed" --selection "$selection" --data_dir "$DATA_DIR" \
        --val_init_states "$VAL_INIT_STATES" --train_steps "$TRAIN_STEPS" --output_dir "$dir"
    fi
    if [[ " $STAGES " == *" eval "* ]]; then
      mkdir -p "$dir/eval"
      if [[ -n "$BASE_CSV" && ! -f "$dir/eval/per_episode_base.csv" ]]; then cp "$BASE_CSV" "$dir/eval/"; fi
      echo "=== evaluate $selection seed $seed ($(date -Is)) ===" | tee -a "$dir/eval/eval.log"
      run_until_done "$dir/eval/eval.log" --stage eval --output_dir "$dir" --eval_policies base,lora \
        --eval_seeds "$EVAL_SEEDS" --eval_init_states "$EVAL_INIT_STATES" --max_episodes_per_process 3
    fi
  done
done
