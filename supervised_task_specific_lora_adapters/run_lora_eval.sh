#!/usr/bin/env bash
# run_lora_eval.sh
# Evaluates SmolVLA policy with task-specific LoRA adapters across tasks and seeds.
#
# Usage:
#   bash run_lora_eval.sh --task_id 0 --lora_dir outputs/run_lora_training_20260829_215932 --num_episodes 10 --output_dir lora_eval_results_task0_3seeds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/swagat/anaconda3/envs/lerobot_v040/bin/python}"

# Headless rendering backend for MuJoCo and PyOpenGL
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "Executing LoRA Policy Evaluation with MUJOCO_GL=$MUJOCO_GL ..."
exec "$PYTHON_BIN" "$SCRIPT_DIR/eval_lora_policy.py" "$@"
