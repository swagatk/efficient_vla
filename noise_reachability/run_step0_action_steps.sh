#!/usr/bin/env bash
# run_step0_action_steps.sh
# Step 0 of noise_reachability/README.md: lerobot/smolvla_libero at another n_action_steps on the weak tasks,
# through baseline_performance/evaluate_all.py with the protocol of the 171/300 run otherwise. Reported numbers
# come from Kaggle (baseline_performance/kaggle_baseline_lerobot_eval.ipynb, N_ACTION_STEPS and TASKS in its
# step 1); this script runs the same evaluation on a Colab or local GPU.
#
# Usage (from anywhere):
#   N_ACTION_STEPS=10 bash noise_reachability/run_step0_action_steps.sh
#   N_ACTION_STEPS=50 JOBS=4 OUT_DIR=/content/step0_nas50 bash noise_reachability/run_step0_action_steps.sh
#   SMOKE=1 N_ACTION_STEPS=10 bash noise_reachability/run_step0_action_steps.sh   # task 0, seed 0, 1 episode
# Rerunning with the same OUT_DIR resumes. Then:
#   python -S noise_reachability/compare_action_steps.py \
#       week6/smolvla_libero_lerobot_eval_kaggle/per_episode.csv 10=<OUT_DIR>/per_episode.csv --alpha 0.025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
LEROBOT_EVAL="${LEROBOT_EVAL:-$(dirname "$PYTHON_BIN")/lerobot-eval}"

POLICY_REPO="${POLICY_REPO:-lerobot/smolvla_libero}"
POLICY_REVISION="${POLICY_REVISION:-a24f1f380461b665158d52acd140152c79e8b36d}"
N_ACTION_STEPS="${N_ACTION_STEPS:?set N_ACTION_STEPS, e.g. 10 or 50}"
TASKS="${TASKS:-0 4 6 8}"
SEEDS="${SEEDS:-0 1 2}"
N_EPISODES="${N_EPISODES:-10}"
JOBS="${JOBS:-1}"
GPUS="${GPUS:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/noise_reachability/step0_nas${N_ACTION_STEPS}}"

if [[ "${SMOKE:-0}" == 1 ]]; then
  TASKS=0 SEEDS=0 N_EPISODES=1
  OUT_DIR="${OUT_DIR%/}_smoke"
fi

RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
POLICY_PATH="$("$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$POLICY_REPO', revision='$POLICY_REVISION'))" | tail -n 1)"

export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
cd "$REPO_ROOT"
# shellcheck disable=SC2086  # TASKS and SEEDS are space-separated lists
"$PYTHON_BIN" -u baseline_performance/evaluate_all.py --resume_dir "$OUT_DIR" --tasks $TASKS --seeds $SEEDS \
  --n_episodes "$N_EPISODES" --policy_path "$POLICY_PATH" --lerobot_eval "$LEROBOT_EVAL" --order seed \
  --n_action_steps "$N_ACTION_STEPS" --rename_map "$RENAME_MAP" --jobs "$JOBS" ${GPUS:+--gpus "$GPUS"}
