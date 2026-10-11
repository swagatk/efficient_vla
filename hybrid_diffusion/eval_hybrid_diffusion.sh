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
#       --alpha_sweep LIST    Sweep multiple alpha values (e.g. '0.05 0.1 0.15 0.2' or '[0.05, 0.1, 0.15, 0.2]')
#   -e, --episodes NUM        Number of evaluation episodes (default: 10)
#   -s, --diffusion_steps NUM Diffusion integration steps (default: 10)
#   -b, --batch_size NUM      Episodes per process to avoid headless EGL driver crash (default: 3, 0 to disable)
#       --seed_offset NUM     Offset added to episode seeds (default: 0)
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
export PYTHONFAULTHANDLER=1

# Add paths to PYTHONPATH (including REPO_ROOT for sitecustomize.py and robosuite_mujoco_patch)
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -d "$HOME/LIBERO" ]]; then
  export PYTHONPATH="$REPO_ROOT:$HOME/LIBERO:$SCRIPT_DIR:${PYTHONPATH:-}"
else
  export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR:${PYTHONPATH:-}"
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
ALPHA_SWEEP="${ALPHA_SWEEP:-}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
IMAGE_FLIP_MODE="${IMAGE_FLIP_MODE:-vertical_horizontal}"
MAX_STEPS="${MAX_STEPS:-520}"
REPLAN_EACH_STEP="${REPLAN_EACH_STEP:-1}"
SEED_OFFSET="${SEED_OFFSET:-0}"
BATCH_SIZE="${BATCH_SIZE:-3}"
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
    --alpha_sweep)
      ALPHA_SWEEP="$2"
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
    -b|--batch_size|--chunk_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --seed_offset|--eval_seed_offset)
      SEED_OFFSET="$2"
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
      echo "      --alpha_sweep LIST     Sweep multiple alpha values (e.g. '0.05 0.1 0.15 0.2' or '[0.05, 0.1, 0.15, 0.2]')"
      echo "  -e, --episodes NUM         Number of evaluation episodes (default: 10)"
      echo "  -s, --diffusion_steps NUM  Diffusion integration steps (default: 10)"
      echo "  -b, --batch_size NUM       Episodes per process to avoid EGL driver crash (default: 5, 0 to disable)"
      echo "      --max_steps NUM        Maximum rollout steps per episode (default: 520)"
      echo "      --replan_each_step 0|1 Replan residual chunk at each step (default: 1)"
      echo "      --seed_offset NUM      Offset added to evaluation episode seeds (default: 0)"
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

run_eval_job() {
  local TARGET_DIR="$1"
  local TARGET_ALPHA="$2"
  mkdir -p "$TARGET_DIR"
  local TARGET_LOG="$TARGET_DIR/eval.log"

  echo "Starting evaluation... Full log will be saved to: $TARGET_LOG"

  if [[ "$POLICY_MODE" == "base" ]]; then
    echo "Running canonical lerobot_eval for base policy on task $TASK_ID ($EPISODES episodes)..."
    local CMD=(
      "$PYTHON_BIN" -m lerobot.scripts.lerobot_eval
      --policy.path="$BASE_POLICY_PATH"
      --policy.device=cuda
      --env.type=libero
      --env.task=libero_10
      --env.render_mode=rgb_array
      --env.max_parallel_tasks=1
      --env.observation_height=256
      --env.observation_width=256
      --env.episode_length="$MAX_STEPS"
      --eval.batch_size=1
      --eval.use_async_envs=false
      --eval.n_episodes="$EPISODES"
      --output_dir="$TARGET_DIR"
      --env.task_ids="[$TASK_ID]"
      --seed="$SEED_OFFSET"
    )
    "${CMD[@]}" 2>&1 | tee -a "$TARGET_LOG"
    local STATUS="${PIPESTATUS[0]}"
    if [[ "$STATUS" -ne 0 ]]; then
      echo "Error: lerobot_eval failed with status $STATUS" >&2
      return "$STATUS"
    fi

    "$PYTHON_BIN" - "$TARGET_DIR" "$TASK_ID" << 'PY'
import json, sys, os
target_dir = sys.argv[1]
task_id = int(sys.argv[2])
info_file = os.path.join(target_dir, "eval_info.json")
if os.path.exists(info_file):
    with open(info_file) as f:
        data = json.load(f)
    sr = data.get("overall", {}).get("pc_success", 0.0) / 100.0
    rew = data.get("overall", {}).get("avg_sum_reward", 0.0)
    n_eps = data.get("overall", {}).get("n_episodes", 0)
    csv_file = os.path.join(target_dir, "eval_summary.csv")
    with open(csv_file, "w") as f:
        f.write("task_id,task_name,success_rate,avg_reward,num_episodes\n")
        f.write(f"{task_id},task_{task_id},{sr:.4f},{rew:.4f},{n_eps}\n")
    print(f"[Summary] Successfully written to {csv_file}: Success Rate = {sr*100:.1f}%")
PY
    return 0
  fi

  if (( BATCH_SIZE > 0 && EPISODES > BATCH_SIZE )); then
    echo "Evaluating $EPISODES total episodes in safe batches of up to $BATCH_SIZE to avoid headless EGL crashes..."
    local REMAINING="$EPISODES"
    local CURR_OFFSET="$SEED_OFFSET"
    local BATCH_IDX=1

    while (( REMAINING > 0 )); do
      local CURR_EPS=$(( REMAINING > BATCH_SIZE ? BATCH_SIZE : REMAINING ))
      echo ""
      echo "==========================================================="
      echo "🚀 Launching Batch $BATCH_IDX: $CURR_EPS episodes (seed offset $CURR_OFFSET) -> $TARGET_DIR"
      echo "==========================================================="

      local CMD=(
        "$PYTHON_BIN" "$SCRIPT_DIR/train_hybrid_diffusion.py"
        --base_policy_path "$BASE_POLICY_PATH"
        --eval_only
        --task_id "$TASK_ID"
        --eval_episodes "$CURR_EPS"
        --eval_seed_offset "$CURR_OFFSET"
        --eval_max_steps "$MAX_STEPS"
        --eval_policy_mode "$POLICY_MODE"
        --eval_residual_alpha "$TARGET_ALPHA"
        --eval_diffusion_steps "$DIFFUSION_STEPS"
        --eval_output_dir "$TARGET_DIR"
        --image_flip_mode "$IMAGE_FLIP_MODE"
      )
      if [[ -n "$CHECKPOINT_PATH" ]]; then
        CMD+=(--eval_checkpoint_path "$CHECKPOINT_PATH")
      fi
      if [[ "$REPLAN_EACH_STEP" == "1" || "$REPLAN_EACH_STEP" == "true" ]]; then
        CMD+=(--eval_replan_each_step)
      fi
      if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_ARGS[@]}")
      fi

      "${CMD[@]}" 2>&1 | tee -a "$TARGET_LOG"
      local STATUS="${PIPESTATUS[0]}"
      if [[ "$STATUS" -ne 0 ]]; then
        echo "Error: Batch $BATCH_IDX failed with status $STATUS" >&2
        return "$STATUS"
      fi

      CURR_OFFSET=$(( CURR_OFFSET + CURR_EPS ))
      REMAINING=$(( REMAINING - CURR_EPS ))
      BATCH_IDX=$(( BATCH_IDX + 1 ))
      if (( REMAINING > 0 )); then
        sleep 2
      fi
    done
    echo ""
    echo "==========================================================="
    echo "✅ Completed: $EPISODES episodes merged into: $TARGET_DIR"
    echo "==========================================================="
  else
    local PY_CMD=(
      "$PYTHON_BIN" "$SCRIPT_DIR/train_hybrid_diffusion.py"
      --base_policy_path "$BASE_POLICY_PATH"
      --eval_only
      --task_id "$TASK_ID"
      --eval_episodes "$EPISODES"
      --eval_seed_offset "$SEED_OFFSET"
      --eval_max_steps "$MAX_STEPS"
      --eval_policy_mode "$POLICY_MODE"
      --eval_residual_alpha "$TARGET_ALPHA"
      --eval_diffusion_steps "$DIFFUSION_STEPS"
      --eval_output_dir "$TARGET_DIR"
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
    "${PY_CMD[@]}" 2>&1 | tee -a "$TARGET_LOG"
    local STATUS="${PIPESTATUS[0]}"
    if [[ "$STATUS" -ne 0 ]]; then
      return "$STATUS"
    fi
  fi
}

if [[ -n "$ALPHA_SWEEP" ]]; then
  CLEAN_SWEEP=$(echo "$ALPHA_SWEEP" | tr -d '[],"' | tr "'" " ")
  read -r -a ALPHA_LIST <<< "$CLEAN_SWEEP"

  if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="./outputs/eval_alpha_sweep_${TIMESTAMP}"
  fi
  mkdir -p "$OUTPUT_DIR"
  TOP_SUMMARY_CSV="$OUTPUT_DIR/eval_summary.csv"
  echo "alpha,task_id,task_name,success_rate,avg_reward,num_episodes,output_dir" > "$TOP_SUMMARY_CSV"

  echo "==========================================================="
  echo "🔬 Hybrid Diffusion Alpha Sweep"
  echo "==========================================================="
  echo "  • Task ID:             $TASK_ID"
  echo "  • Policy Mode:         $POLICY_MODE"
  echo "  • Alpha Sweep Values:  ${ALPHA_LIST[*]}"
  echo "  • Episodes Per Alpha:  $EPISODES"
  echo "  • Batch Size:          $BATCH_SIZE"
  echo "  • Diffusion Steps:     $DIFFUSION_STEPS"
  echo "  • Replan Each Step:    $REPLAN_EACH_STEP"
  echo "  • Checkpoint:          ${CHECKPOINT_PATH:-[None - Base Policy Only]}"
  echo "  • Root Output Dir:     $OUTPUT_DIR"
  echo "  • Flip Mode:           $IMAGE_FLIP_MODE"
  echo "==========================================================="

  for CURR_ALPHA in "${ALPHA_LIST[@]}"; do
    ALPHA_DIR="$OUTPUT_DIR/alpha_${CURR_ALPHA}"
    echo ""
    echo "###########################################################"
    echo "▶️  Evaluating Alpha = $CURR_ALPHA -> $ALPHA_DIR"
    echo "###########################################################"
    run_eval_job "$ALPHA_DIR" "$CURR_ALPHA"

    SUB_CSV="$ALPHA_DIR/eval_summary.csv"
    if [[ -f "$SUB_CSV" ]]; then
      tail -n +2 "$SUB_CSV" | while IFS=, read -r tid tname sr rew neps; do
        if [[ -n "$tid" ]]; then
          echo "${CURR_ALPHA},${tid},${tname},${sr},${rew},${neps},alpha_${CURR_ALPHA}" >> "$TOP_SUMMARY_CSV"
        fi
      done
    fi
  done

  echo ""
  echo "==========================================================="
  echo "🏁 Alpha Sweep Complete! Consolidated summary:"
  echo "==========================================================="
  cat "$TOP_SUMMARY_CSV"
  echo "==========================================================="
  echo "Summary saved to: $TOP_SUMMARY_CSV"
else
  # Single Alpha Run
  if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="./outputs/eval_${POLICY_MODE}_${TIMESTAMP}"
  fi
  mkdir -p "$OUTPUT_DIR"

  echo "==========================================================="
  echo "🤖 Hybrid Diffusion Evaluation"
  echo "==========================================================="
  echo "  • Task ID:             $TASK_ID"
  echo "  • Policy Mode:         $POLICY_MODE"
  echo "  • Residual Alpha:      $ALPHA"
  echo "  • Episodes:            $EPISODES"
  echo "  • Batch Size:          $BATCH_SIZE"
  echo "  • Seed Offset:         $SEED_OFFSET"
  echo "  • Diffusion Steps:     $DIFFUSION_STEPS"
  echo "  • Replan Each Step:    $REPLAN_EACH_STEP"
  echo "  • Checkpoint:          ${CHECKPOINT_PATH:-[None - Base Policy Only]}"
  echo "  • Output Directory:    $OUTPUT_DIR"
  echo "  • Flip Mode:           $IMAGE_FLIP_MODE"
  echo "==========================================================="

  run_eval_job "$OUTPUT_DIR" "$ALPHA"
fi
