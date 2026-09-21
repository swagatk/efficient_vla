#!/usr/bin/env bash
# eval_hybrid_diffusion.sh
# Clean, robust evaluation runner for Hybrid Diffusion / Residual VLA policies.
#
# Usage:
#   bash eval_hybrid_diffusion.sh [OPTIONS]
#
# Options:
#   -o, --output_dir DIR      Directory to store eval results, logs, and configs.
#                             (Default: ./outputs/eval_{POLICY_MODE}_{datetime})
#   -t, --task_id ID          Task ID in LIBERO suite (default: 1)
#   -m, --policy_mode MODE    Policy mode: base | residual | hybrid | mixed (default: residual)
#   -c, --checkpoint PATH     Path to trained checkpoint (.pt)
#   -r, --run_dir DIR         Path to training run dir (auto-finds checkpoint within)
#   -a, --alpha VAL           Residual scaling factor alpha (default: 1.0)
#   -e, --episodes NUM        Number of evaluation episodes (default: 10)
#   -s, --diffusion_steps NUM Diffusion integration steps (default: 10)
#       --replan_each_step 0|1 Replan residual chunk at each step (default: 1)
#       --flip_mode MODE      Camera flip mode (default: vertical_horizontal)
#   -h, --help                Show this help message
#
# Environment variable overrides (also supported):
#   OUTPUT_DIR, TASK_ID, POLICY_MODE, CHECKPOINT_PATH, RUN_DIR, ALPHA, EPISODES, DIFFUSION_STEPS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# --- Python Environment Detection ---
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/swagat/miniconda3/envs/lerobot_env/bin/python" ]]; then
    PYTHON_BIN="/home/swagat/miniconda3/envs/lerobot_env/bin/python"
  elif [[ -x "/home/swagat/anaconda3/envs/lerobot_v040/bin/python" ]]; then
    PYTHON_BIN="/home/swagat/anaconda3/envs/lerobot_v040/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="python3"
  fi
fi
export PYTHON_BIN

# Headless EGL rendering configuration
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Add paths to PYTHONPATH
if [[ -d "$HOME/LIBERO" ]]; then
  export PYTHONPATH="$HOME/LIBERO:$SCRIPT_DIR:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
fi

# Ensure LIBERO config exists
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
mkdir -p "$LIBERO_CONFIG_PATH"

# Default configuration variables (can be set via env vars or CLI flags)
TASK_ID="${TASK_ID:-1}"
BASE_POLICY_PATH="${BASE_POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
EPISODES="${EPISODES:-10}"
POLICY_MODE="${POLICY_MODE:-residual}"
ALPHA="${ALPHA:-1.0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
IMAGE_FLIP_MODE="${IMAGE_FLIP_MODE:-vertical_horizontal}"
MAX_STEPS="${MAX_STEPS:-520}"
REPLAN_EACH_STEP="${REPLAN_EACH_STEP:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RUN_DIR="${RUN_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Parse Command Line Arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -t|--task_id)
      TASK_ID="$2"
      shift 2
      ;;
    -m|--policy_mode)
      POLICY_MODE="$2"
      shift 2
      ;;
    -c|--checkpoint|--checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    -r|--run_dir)
      RUN_DIR="$2"
      shift 2
      ;;
    -a|--alpha|--residual_alpha)
      ALPHA="$2"
      shift 2
      ;;
    -e|--episodes)
      EPISODES="$2"
      shift 2
      ;;
    -s|--diffusion_steps)
      DIFFUSION_STEPS="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --replan_each_step)
      REPLAN_EACH_STEP="$2"
      shift 2
      ;;
    --flip_mode)
      IMAGE_FLIP_MODE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash eval_hybrid_diffusion.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -o, --output_dir DIR       Directory to store eval results and logs (default: ./outputs/eval_{POLICY_MODE}_{datetime})"
      echo "  -t, --task_id ID           Task ID in LIBERO suite (default: 1)"
      echo "  -m, --policy_mode MODE     Policy mode: base | residual | hybrid | mixed (default: residual)"
      echo "  -c, --checkpoint PATH      Path to trained checkpoint (.pt)"
      echo "  -r, --run_dir DIR          Path to training run dir (auto-finds checkpoint within)"
      echo "  -a, --alpha VAL            Residual scaling factor alpha (default: 1.0)"
      echo "  -e, --episodes NUM         Number of evaluation episodes (default: 10)"
      echo "  -s, --diffusion_steps NUM  Diffusion integration steps (default: 10)"
      echo "      --max_steps NUM        Maximum rollout steps per episode (default: 520)"
      echo "      --replan_each_step 0|1 Replan residual chunk at each step (default: 1)"
      echo "      --flip_mode MODE       Camera flip mode (default: vertical_horizontal)"
      echo "  -h, --help                 Show this help message"
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Resolve default output directory: ./outputs/eval_{POLICY_MODE}_{datetime}
if [[ -z "$OUTPUT_DIR" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="./outputs/eval_${POLICY_MODE}_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

# Resolve checkpoint location
if [[ "$POLICY_MODE" != "base" ]]; then
  if [[ -n "$RUN_DIR" ]]; then
    # Search within RUN_DIR for checkpoints
    FOUND_CKPT=$(find "$RUN_DIR" -name "latest_checkpoint.pt" -o -name "*.pt" | head -1)
    if [[ -n "$FOUND_CKPT" ]]; then
      CHECKPOINT_PATH="$FOUND_CKPT"
      echo "[Run Dir Checkpoint Found]: $CHECKPOINT_PATH"
    else
      echo "Error: No checkpoint (.pt) found inside RUN_DIR: $RUN_DIR" >&2
      exit 1
    fi
  elif [[ -z "$CHECKPOINT_PATH" ]]; then
    # Auto-detect latest checkpoint from ./outputs
    LATEST_CKPT=$(find "$SCRIPT_DIR/outputs" -name "latest_checkpoint.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    if [[ -n "$LATEST_CKPT" ]]; then
      CHECKPOINT_PATH="$LATEST_CKPT"
      echo "[Auto-detected latest checkpoint]: $CHECKPOINT_PATH"
    else
      echo "Error: No checkpoint found in $SCRIPT_DIR/outputs. Please set --checkpoint path/to/checkpoint.pt or --run_dir path/to/run_dir" >&2
      exit 1
    fi
  fi
fi

echo "==========================================================="
echo "🤖 Hybrid Diffusion Evaluation"
echo "==========================================================="
echo "  • Task ID:             $TASK_ID"
echo "  • Policy Mode:         $POLICY_MODE"
echo "  • Residual Alpha:      $ALPHA"
echo "  • Episodes:            $EPISODES"
echo "  • Diffusion Steps:     $DIFFUSION_STEPS"
echo "  • Replan Each Step:    $REPLAN_EACH_STEP"
echo "  • Checkpoint:          ${CHECKPOINT_PATH:-[None - Base Policy Only]}"
echo "  • Output Directory:    $OUTPUT_DIR"
echo "  • Flip Mode:           $IMAGE_FLIP_MODE"
echo "==========================================================="

PY_CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/train_hybrid_diffusion.py"
  --base_policy_path "$BASE_POLICY_PATH"
  --eval_only
  --task_id "$TASK_ID"
  --eval_episodes "$EPISODES"
  --eval_max_steps "$MAX_STEPS"
  --eval_policy_mode "$POLICY_MODE"
  --eval_residual_alpha "$ALPHA"
  --eval_diffusion_steps "$DIFFUSION_STEPS"
  --eval_output_dir "$OUTPUT_DIR"
  --image_flip_mode "$IMAGE_FLIP_MODE"
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  PY_CMD+=(--eval_checkpoint_path "$CHECKPOINT_PATH")
fi

if [[ "$REPLAN_EACH_STEP" == "1" || "$REPLAN_EACH_STEP" == "true" ]]; then
  PY_CMD+=(--eval_replan_each_step)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${EXTRA_ARGS[@]}")
fi

# Execute with terminal output and tee to eval.log in OUTPUT_DIR
LOG_FILE="$OUTPUT_DIR/eval.log"
echo "Starting evaluation... Full log will be saved to: $LOG_FILE"
"${PY_CMD[@]}" 2>&1 | tee "$LOG_FILE"
