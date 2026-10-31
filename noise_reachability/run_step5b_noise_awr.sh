#!/usr/bin/env bash
# run_step5b_noise_awr.sh
# Step 5b of noise_reachability/README.md. The command lines below are the record of how reported numbers were
# produced. Finished work is skipped, so a stage can be re-run after an interruption.
#
#   STAGE=harness|prior|train|eval EPOCHS=37 OUT_ROOT=... bash noise_reachability/run_step5b_noise_awr.sh
#   STAGE=pilot OUT_ROOT=... ...                                 # one epoch of training seed 9, timed
#   STAGE=train TRAIN_SEEDS=0 EPOCHS=2 PARTS=1 TRAIN_STATES=10-10 GATE=0 ...      # smoke run
#   STAGE=harness|prior|eval TRAIN_SEEDS=0 EPOCHS=2 SEED_PARTS=16 EVAL_STATES=2 GATE=0 ...   # smoke run
# POLICY_PATH: the snapshot at revision a24f1f3804 (the notebook passes it). harness, and train after seed 0, end with
# their gate: a failure stops the run before more is spent.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
# The repo root must be importable at start-up so sitecustomize loads robosuite_mujoco_patch (FAIL-003).
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" PYTHONFAULTHANDLER=1
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
STAGE="${STAGE:?set STAGE}"
EPOCHS="${EPOCHS:-37}"
PARTS="${PARTS:-4}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
SEED_PARTS="${SEED_PARTS:-16-19 20-23 24-27 28-31}"
TRAIN_STATES="${TRAIN_STATES:-10-49}"
EVAL_STATES="${EVAL_STATES:-0-9}"
GATE="${GATE:-1}"
export POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_libero}"
ENTRY=noise_reachability/train_noise_awr.py
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/logs"

epoch() {  # training seed, epoch: collect its parts, then update
  local t="$1" e="$2" out="$OUT_ROOT/train/seed$1" tag
  tag="$(printf 'seed%s_e%03d' "$t" "$e")"
  mkdir -p "$out"
  [[ -f "$out/$(printf 'actor_e%03d.pt' "$e")" ]] && return 0
  local pids=() k failed=0
  for k in $(seq 0 $((PARTS - 1))); do
    "$PYTHON_BIN" -u "$ENTRY" --mode collect --output_dir "$out" --policy_path "$POLICY_PATH" --train_seed "$t" \
      --train_init_states "$TRAIN_STATES" --epoch "$e" --part "$k" --parts "$PARTS" >> "$OUT_ROOT/logs/${tag}_part$k.log" 2>&1 &
    pids+=($!)
    sleep 5  # staggered EGL start-up (FAIL-006)
  done
  for k in "${pids[@]}"; do wait "$k" || failed=1; done
  if [[ $failed != 0 ]]; then echo "epoch failed: seed $t epoch $e" >&2; return 1; fi
  "$PYTHON_BIN" -u "$ENTRY" --mode update --output_dir "$out" --policy_path "$POLICY_PATH" --train_seed "$t" \
    --train_init_states "$TRAIN_STATES" --epoch "$e" --parts "$PARTS" >> "$OUT_ROOT/logs/${tag}_update.log" 2>&1
}

arm() {  # arm, training seed (or "shared"), seeds of each part: one process per part, repeated until done
  local name="$1" who="$2"; shift 2
  local t=0 k=0 pids=() failed=0 spec
  [[ "$who" != shared ]] && t="$who"
  for spec in "$@"; do
    (
      code=3
      while [[ $code == 3 ]]; do
        code=0
        "$PYTHON_BIN" -u "$ENTRY" --mode eval --arm "$name" --output_dir "$OUT_ROOT/train/seed$t" \
          --policy_path "$POLICY_PATH" --train_seed "$t" --epoch "$((EPOCHS - 1))" --seeds "$spec" --init_states "$EVAL_STATES" \
          --eval_dir "$OUT_ROOT/eval/$( [[ $who == shared ]] && echo shared || echo "seed$t" )/part$k" \
          >> "$OUT_ROOT/logs/eval_${name}_${who}_part$k.log" 2>&1 || code=$?
      done
      exit "$code"
    ) &
    pids+=($!)
    k=$((k + 1))
    sleep 5
  done
  for k in "${pids[@]}"; do wait "$k" || failed=1; done
  [[ $failed == 0 ]]
}

gate() { "$PYTHON_BIN" -I -S noise_reachability/analyze_step5b.py --root "$OUT_ROOT" --epochs "$EPOCHS" --gate "$1"; }

case "$STAGE" in
  pilot)
    t0=$(date +%s)
    epoch 9 0
    echo "$(( $(date +%s) - t0 ))" > "$OUT_ROOT/pilot_epoch_seconds.txt" ;;
  harness)
    # shellcheck disable=SC2086
    arm base shared $SEED_PARTS
    if [[ "$GATE" == 1 ]]; then gate harness; fi ;;
  prior)
    # shellcheck disable=SC2086
    arm prior shared $SEED_PARTS
    arm zero shared 16 ;;
  train)
    for t in $TRAIN_SEEDS; do
      for e in $(seq 0 $((EPOCHS - 1))); do epoch "$t" "$e"; done
      if [[ "$t" == 0 && "$GATE" == 1 ]]; then gate training; fi
    done ;;
  eval)
    for t in $TRAIN_SEEDS; do
      # shellcheck disable=SC2086
      arm actor "$t" $SEED_PARTS
      arm actor_mean "$t" 16
    done ;;
  *) echo "STAGE must be pilot, harness, prior, train or eval" >&2; exit 2 ;;
esac
