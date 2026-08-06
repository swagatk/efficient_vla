#!/usr/bin/env bash
set -euo pipefail

# hybrid_diffusion/run_hybrid_diffusion.sh
# Orchestrates train_hybrid_diffusion.py with power management, resumption, and config logging.

PYTHON_BIN="${PYTHON_BIN:-/home/swagat/anaconda3/envs/lerobot_v040/bin/python}"

BASE_POLICY_PATH="${BASE_POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero_10}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
ACTION_DIM="${ACTION_DIM:-7}"
COND_DIM="${COND_DIM:-960}"
DIFF_HIDDEN_DIM="${DIFF_HIDDEN_DIM:-256}"
DIFF_LAYERS="${DIFF_LAYERS:-5}"
VIS_FREQ="${VIS_FREQ:-2}"
DEVICE="${DEVICE:-cuda}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0 1 2 4 6 7 8 9}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
IMAGE_FLIP_MODE="${IMAGE_FLIP_MODE:-vertical_horizontal}"
EVAL_POLICY_MODE="${EVAL_POLICY_MODE:-residual}"
RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.02}"
EVAL_DIFFUSION_STEPS="${EVAL_DIFFUSION_STEPS:-10}"
EVAL_ACTION_CLIP="${EVAL_ACTION_CLIP:-1.0}"
EVAL_LOG_ACTION_STATS_EVERY="${EVAL_LOG_ACTION_STATS_EVERY:-0}"
EVAL_REPLAN_EACH_STEP="${EVAL_REPLAN_EACH_STEP:-1}"
RESIDUAL_TARGET="${RESIDUAL_TARGET:-1}"
DELTA_L2_WEIGHT="${DELTA_L2_WEIGHT:-0.001}"
WANDB_PROJECT="${WANDB_PROJECT:-hybrid_diffusion_vla}"

RESUME="${RESUME:-0}"
USE_POWER_HARDENING=1
INTERRUPTED=0
TERMINATED=0
CLEANUP_DONE=0
CURRENT_CHILD_PID=""
ORIG_POWER_PROFILE=""
ORIG_SLEEP_MODE=""
OS_NAME="$(uname -s)"

# Power Hardening
apply_power_hardening() {
  [[ "$USE_POWER_HARDENING" == "1" ]] || return 0
  echo "[power] Enabling performance profile..."
  if command -v powerprofilesctl >/dev/null 2>&1; then
    ORIG_POWER_PROFILE="$(powerprofilesctl get 2>/dev/null || true)"
    powerprofilesctl set performance 2>/dev/null || true
  fi
  if command -v gsettings >/dev/null 2>&1; then
    ORIG_SLEEP_MODE="$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || true)"
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
  fi
}

restore_power_hardening() {
  [[ "$USE_POWER_HARDENING" == "1" ]] || return 0
  echo "[power] Restoring original power profile..."
  if [[ -n "$ORIG_POWER_PROFILE" ]] && command -v powerprofilesctl >/dev/null 2>&1; then
    powerprofilesctl set "$ORIG_POWER_PROFILE" 2>/dev/null || true
  fi
  if [[ -n "$ORIG_SLEEP_MODE" ]] && command -v gsettings >/dev/null 2>&1; then
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type "$ORIG_SLEEP_MODE" 2>/dev/null || true
  fi
}

# Interrupt handling
on_interrupt() {
  if [[ "$INTERRUPTED" -eq 0 ]]; then
    INTERRUPTED=1
    echo "[interrupt] Ctrl+C received. Will stop gracefully after current task."
  else
    echo "[interrupt] Second Ctrl+C received; exiting immediately."
    exit 130
  fi
}

on_term() {
  TERMINATED=1
  echo "[term] termination requested; shutting down now."
  if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    kill -TERM "$CURRENT_CHILD_PID" 2>/dev/null || true
  fi
  exit 143
}

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then return 0; fi
  CLEANUP_DONE=1
  restore_power_hardening || true
}

trap on_interrupt INT
trap on_term TERM
trap cleanup EXIT

# Set up output directory
if [[ -n "${RESUME_DIR:-}" && -d "${RESUME_DIR}" ]]; then
  OUTPUT_ROOT="${RESUME_DIR}"
  RESUME="1"
  echo "Resuming from directory: $OUTPUT_ROOT"
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="./outputs/run_hybrid_diffusion_${TIMESTAMP}"
  mkdir -p "$OUTPUT_ROOT"
  RESUME="0"
  echo "Created new run directory: $OUTPUT_ROOT"
fi

# Store config.json
CONFIG_PATH="$OUTPUT_ROOT/config.json"
if [[ "$RESUME" == "0" ]]; then
  "$PYTHON_BIN" -c "
import json
config = {
    'BASE_POLICY_PATH': '$BASE_POLICY_PATH',
    'DATASET_REPO_ID': '$DATASET_REPO_ID',
    'SEEDS': '$SEEDS',
    'EPOCHS': $EPOCHS,
    'BATCH_SIZE': $BATCH_SIZE,
    'LR': $LR,
    'CHUNK_SIZE': $CHUNK_SIZE,
    'ACTION_DIM': $ACTION_DIM,
    'COND_DIM': $COND_DIM,
    'DIFF_HIDDEN_DIM': $DIFF_HIDDEN_DIM,
    'DIFF_LAYERS': $DIFF_LAYERS,
    'VIS_FREQ': $VIS_FREQ,
    'DEVICE': '$DEVICE',
    'EVAL_TASK_IDS': '$EVAL_TASK_IDS',
    'EVAL_EPISODES': $EVAL_EPISODES,
    'IMAGE_FLIP_MODE': '$IMAGE_FLIP_MODE',
    'EVAL_POLICY_MODE': '$EVAL_POLICY_MODE',
    'RESIDUAL_ALPHA': $RESIDUAL_ALPHA,
    'EVAL_DIFFUSION_STEPS': $EVAL_DIFFUSION_STEPS,
    'EVAL_ACTION_CLIP': $EVAL_ACTION_CLIP,
    'EVAL_LOG_ACTION_STATS_EVERY': $EVAL_LOG_ACTION_STATS_EVERY,
    'EVAL_REPLAN_EACH_STEP': $EVAL_REPLAN_EACH_STEP,
    'RESIDUAL_TARGET': $RESIDUAL_TARGET,
    'DELTA_L2_WEIGHT': $DELTA_L2_WEIGHT,
    'WANDB_PROJECT': '$WANDB_PROJECT'
}
with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=4)
"
  echo "Saved parameters to $CONFIG_PATH"
else
  echo "Using existing parameters from $CONFIG_PATH"
fi

apply_power_hardening

SUMMARY_FILE="$OUTPUT_ROOT/hybrid_diff_results.csv"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  echo "seed,exit_code,wandb_run_id,checkpoint_dir" > "$SUMMARY_FILE"
fi

for SEED in $SEEDS; do
  [[ "$INTERRUPTED" -eq 0 ]] || break
  [[ "$TERMINATED" -eq 0 ]] || break

  UNIT_DIR="$OUTPUT_ROOT/seed_${SEED}"
  CHECKPOINT_DIR="$UNIT_DIR/checkpoints"
  DONE_MARKER="$UNIT_DIR/.completed"
  WANDB_RUN_ID_FILE="$UNIT_DIR/wandb_run_id.txt"

  mkdir -p "$UNIT_DIR" "$CHECKPOINT_DIR"

  if [[ "$RESUME" == "1" && -f "$DONE_MARKER" ]]; then
    echo "[seed=$SEED] already completed. Skipping."
    continue
  fi

  if [[ -f "$WANDB_RUN_ID_FILE" ]]; then
    WANDB_RUN_ID="$(head -n 1 "$WANDB_RUN_ID_FILE" | tr -d '[:space:]')"
  else
    WANDB_RUN_ID="$($PYTHON_BIN - <<'PY'
try:
    import wandb
    print(wandb.util.generate_id())
except Exception:
    import uuid
    print(uuid.uuid4().hex[:8])
PY
)"
    echo "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
  fi

  echo "----------------------------------------"
  echo "Starting Training for Seed: $SEED"
  echo "WandB Run ID: $WANDB_RUN_ID"
  echo "----------------------------------------"

  CMD=(
    env "PYTHONUNBUFFERED=1" "PYTHONHASHSEED=$SEED" "$PYTHON_BIN" -u "train_hybrid_diffusion.py"
    "--base_policy_path" "$BASE_POLICY_PATH"
    "--dataset_repo_id" "$DATASET_REPO_ID"
    "--batch_size" "$BATCH_SIZE"
    "--epochs" "$EPOCHS"
    "--lr" "$LR"
    "--chunk_size" "$CHUNK_SIZE"
    "--action_dim" "$ACTION_DIM"
    "--cond_dim" "$COND_DIM"
    "--diff_hidden_dim" "$DIFF_HIDDEN_DIM"
    "--diff_layers" "$DIFF_LAYERS"
    "--device" "$DEVICE"
    "--out_dir" "$CHECKPOINT_DIR"
    "--wandb_project" "$WANDB_PROJECT"
    "--wandb_run_id" "$WANDB_RUN_ID"
    "--wandb_resume" "allow"
    "--vis_freq" "$VIS_FREQ"
    "--eval_policy_mode" "$EVAL_POLICY_MODE"
    "--eval_residual_alpha" "$RESIDUAL_ALPHA"
    "--eval_diffusion_steps" "$EVAL_DIFFUSION_STEPS"
    "--eval_action_clip" "$EVAL_ACTION_CLIP"
    "--eval_log_action_stats_every" "$EVAL_LOG_ACTION_STATS_EVERY"
    "--eval_episodes" "$EVAL_EPISODES"
    "--image_flip_mode" "$IMAGE_FLIP_MODE"
    "--delta_l2_weight" "$DELTA_L2_WEIGHT"
    "--eval_task_ids" $EVAL_TASK_IDS
  )

  if [[ "$RESIDUAL_TARGET" == "1" ]]; then
    CMD+=("--residual_target")
  fi

  if [[ "$EVAL_REPLAN_EACH_STEP" == "1" ]]; then
    CMD+=("--eval_replan_each_step")
  fi

  set +e
  "${CMD[@]}" &
  CURRENT_CHILD_PID=$!
  wait "$CURRENT_CHILD_PID"
  EXIT_CODE=$?
  CURRENT_CHILD_PID=""
  set -e

  if [[ "$EXIT_CODE" -eq 130 ]]; then
    INTERRUPTED=1
  fi

  echo "$SEED,$EXIT_CODE,$WANDB_RUN_ID,$CHECKPOINT_DIR" >> "$SUMMARY_FILE"

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    touch "$DONE_MARKER"
    echo "Seed $SEED finished successfully!"
  else
    echo "Seed $SEED failed with exit code $EXIT_CODE"
  fi
done

echo "Hybrid Diffusion Training Script Complete."
echo "Results summary stored in: $SUMMARY_FILE"
