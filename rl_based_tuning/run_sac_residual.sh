#!/usr/bin/env bash
# run_sac_residual.sh
# Residual SAC on one LIBERO-10 task: train one residual head per seed on init states 10-49, then evaluate
# the base policy once and each trained residual on held-out init states 0-9 x eval seeds 0,1,2, paired
# episode by episode (same environment and noise seed for both). The command lines below are the record of
# how reported numbers were produced. See README.md in this folder.
#
# Usage (from anywhere):
#   TASK_ID=6 SEEDS="0 1 2" NUM_EPISODES=200 bash rl_based_tuning/run_sac_residual.sh
#   STAGES=eval bash rl_based_tuning/run_sac_residual.sh            # evaluation only
#   EXTRA_ARGS="--hard_task_preset" TASK_ID=0 bash rl_based_tuning/run_sac_residual.sh
#   EXTRA_ARGS="--policy_path HuggingFaceVLA/smolvla_libero --n_action_steps -1 --rename_map {}"   # older base
#
# The default base policy is lerobot/smolvla_libero (see train_online_sac.py --help).
#
# Safe to re-run: training resumes from the latest checkpoint and evaluation skips finished episodes.
# Outputs: $OUT_ROOT/seed_<s>/ (checkpoint, train.log, eval/), $OUT_ROOT/eval_base/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
# The repo root must be importable at start-up so sitecustomize loads robosuite_mujoco_patch (FAIL-003).
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" PYTHONFAULTHANDLER=1

TASK_ID="${TASK_ID:-6}"
SEEDS="${SEEDS:-0 1 2}"
NUM_EPISODES="${NUM_EPISODES:-200}"
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2}"
EVAL_INIT_STATES="${EVAL_INIT_STATES:-0-9}"
STAGES="${STAGES:-train eval}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/sac_residual/task_${TASK_ID}}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
TRAIN="$SCRIPT_DIR/train_online_sac.py"
mkdir -p "$OUT_ROOT"
cd "$REPO_ROOT"

# Evaluation runs at most 10 episodes per process (as lerobot-eval does per run); exit code 3 = call again.
run_eval() {
  local code
  while true; do
    set +e
    "$PYTHON_BIN" -u "$TRAIN" --eval --task_id "$TASK_ID" --eval_seeds "$EVAL_SEEDS" \
      --eval_init_states "$EVAL_INIT_STATES" "$@" "${EXTRA[@]}"
    code=$?
    set -e
    if [[ $code -eq 0 ]]; then return 0; fi
    if [[ $code -ne 3 ]]; then echo "[runner] evaluation failed with exit code $code" >&2; return "$code"; fi
    sleep 2  # stagger EGL process start-up (FAIL-006)
  done
}

if [[ " $STAGES " == *" train "* ]]; then
  for seed in $SEEDS; do
    dir="$OUT_ROOT/seed_$seed"
    mkdir -p "$dir"
    echo "=== train task $TASK_ID seed $seed ($(date -Is)) ===" | tee -a "$dir/train.log"
    set +e
    "$PYTHON_BIN" -u "$TRAIN" --task_id "$TASK_ID" --seed "$seed" --num_episodes "$NUM_EPISODES" \
      --checkpoint_dir "$dir" --eval_init_states "$EVAL_INIT_STATES" --resume_training "${EXTRA[@]}" \
      2>&1 | tee -a "$dir/train.log"
    code=${PIPESTATUS[0]}
    set -e
    if [[ $code -eq 3 ]]; then echo "[runner] seed $seed stopped at its time limit; run again to continue"; exit 3; fi
    if [[ $code -ne 0 ]]; then echo "[runner] training seed $seed failed with exit code $code" >&2; exit "$code"; fi
  done
fi

if [[ " $STAGES " == *" eval "* ]]; then
  echo "=== evaluate base policy ($(date -Is)) ==="
  run_eval --eval_policies base --eval_output_dir "$OUT_ROOT/eval_base"
  for seed in $SEEDS; do
    dir="$OUT_ROOT/seed_$seed"
    mkdir -p "$dir/eval"
    # Reuse the base episodes, so only the residual runs and the comparison is paired with them.
    [[ -f "$dir/eval/per_episode_base.csv" ]] || cp "$OUT_ROOT/eval_base/per_episode_base.csv" "$dir/eval/"
    echo "=== evaluate residual, training seed $seed ($(date -Is)) ==="
    run_eval --eval_policies base,residual --checkpoint_dir "$dir" --eval_output_dir "$dir/eval"
  done
fi
