#!/usr/bin/env bash
# run_lora_training.sh
# Ordinarily trains supervised LoRA adapters for SmolVLA across tasks.
#
# To run normally:
#   bash run_lora_training.sh
# To resume latest run:
#   RESUME=1 bash run_lora_training.sh
# To resume a specific run:
#   RESUME_DIR=outputs/run_lora_training_20260801_223600 bash run_lora_training.sh
# To run a quick dry run:
#   DRY_RUN=1 bash run_lora_training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/swagat/anaconda3/envs/lerobot_v040/bin/python}"
export DATA_DIR="${DATA_DIR:-/home/swagat/libero_dataset/libero_10}"

# Target tasks to train (defaults to the 8 evaluated tasks)
TASKS=( ${TASKS:-0 1 2 4 6 7 8 9} )

# Hyperparameters & Memory Optimization Settings
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

DRY_RUN="${DRY_RUN:-0}"

# --- Setup Output Directory & Resuming ---
if [[ -n "${RESUME_DIR:-}" ]]; then
  if [[ ! -d "$RESUME_DIR" ]]; then
    echo "Error: Specified RESUME_DIR '$RESUME_DIR' does not exist."
    exit 1
  fi
  export OUTPUT_DIR="$RESUME_DIR"
  echo "--- Resuming training from specific run: $OUTPUT_DIR ---"
elif [[ "${RESUME:-0}" == "1" ]]; then
  # Find latest run in outputs/
  if [[ -d "outputs" ]]; then
    LATEST_RUN=$(find outputs -maxdepth 1 -name "run_lora_training_*" -type d | sort | tail -n 1)
    if [[ -n "$LATEST_RUN" ]]; then
      export OUTPUT_DIR="$LATEST_RUN"
      echo "--- Found latest run to resume: $OUTPUT_DIR ---"
    else
      echo "No existing run found to resume. Starting new run instead."
      export OUTPUT_DIR="outputs/run_lora_training_$(date +%Y%m%d_%H%M%S)"
      echo "--- Starting new training run: $OUTPUT_DIR ---"
    fi
  else
    echo "No outputs directory found. Starting new run instead."
    export OUTPUT_DIR="outputs/run_lora_training_$(date +%Y%m%d_%H%M%S)"
    echo "--- Starting new training run: $OUTPUT_DIR ---"
  fi
else
  # Start a completely new run
  export OUTPUT_DIR="outputs/run_lora_training_$(date +%Y%m%d_%H%M%S)"
  echo "--- Starting new training run: $OUTPUT_DIR ---"
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# If dry run is active, override parameters for quick step validation
if [[ "$DRY_RUN" == "1" ]]; then
  echo "=== Running in DRY RUN mode ==="
  EPOCHS=1
  BATCH_SIZE=8
  GRAD_ACCUM_STEPS=1
  GRADIENT_CHECKPOINTING=0
fi

# Generate global config.json in the output directory
JSON_DRY_RUN="false"
if [[ "$DRY_RUN" == "1" ]]; then
  JSON_DRY_RUN="true"
fi

JSON_GRAD_CHECK="false"
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  JSON_GRAD_CHECK="true"
fi

TASKS_JSON=$(echo "${TASKS[@]}" | sed 's/ /, /g')

cat <<EOF > "$OUTPUT_DIR/config.json"
{
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "grad_accum_steps": $GRAD_ACCUM_STEPS,
  "gradient_checkpointing": $JSON_GRAD_CHECK,
  "lr": $LR,
  "r": $LORA_RANK,
  "alpha": $LORA_ALPHA,
  "data_dir": "$DATA_DIR",
  "output_dir": "$OUTPUT_DIR",
  "dry_run": $JSON_DRY_RUN,
  "tasks": [$TASKS_JSON]
}
EOF


echo "============================================="
echo "Training tasks: ${TASKS[*]}"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "Learning rate: $LR"
echo "LoRA Rank: $LORA_RANK, Alpha: $LORA_ALPHA"
# Evict any existing Ollama models to ensure 100% free VRAM
if command -v ollama >/dev/null 2>&1; then
  ollama stop mdq100/qwen3.5-coder:35b >/dev/null 2>&1 || true
fi

# Background VRAM watchdog: continuously monitors and evicts competing GPU models
(
  while true; do
    if command -v ollama >/dev/null 2>&1; then
      if ollama ps 2>/dev/null | grep -q "qwen"; then
        ollama stop mdq100/qwen3.5-coder:35b >/dev/null 2>&1 || true
      fi
    fi
    sleep 3
  done
) &
WATCHDOG_PID=$!
trap "kill -9 $WATCHDOG_PID >/dev/null 2>&1 || true" EXIT INT TERM

for task_id in "${TASKS[@]}"; do
  TASK_DIR="$OUTPUT_DIR/task_$task_id"
  if [[ -f "$TASK_DIR/.completed" ]]; then
    echo "--- Task $task_id already completed. Skipping. ---"
    continue
  fi

  echo ""
  echo "---------------------------------------------"
  echo "Starting Supervised LoRA training for Task $task_id"
  echo "---------------------------------------------"
  
  CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/train_supervised_lora.py"
    --task_id "$task_id"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM_STEPS"
    --lr "$LR"
    --r "$LORA_RANK"
    --alpha "$LORA_ALPHA"
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
  )
  
  if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    CMD+=(--gradient_checkpointing)
  fi
  
  if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=(--dry_run)
  fi
  
  MAX_RETRIES=5
  RETRY_COUNT=0
  EXIT_CODE=0
  until [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; do
    set +e
    if command -v systemd-inhibit >/dev/null 2>&1; then
      systemd-inhibit --what=idle:sleep:shutdown --why="SmolVLA LoRA Training" "${CMD[@]}"
    elif command -v gnome-session-inhibit >/dev/null 2>&1; then
      gnome-session-inhibit --inhibit suspend:idle --reason "SmolVLA LoRA Training" "${CMD[@]}"
    else
      "${CMD[@]}"
    fi
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
      break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "⚠️ Warning: Task $task_id training process exited with code $EXIT_CODE. Automatically resuming from saved epoch checkpoint (Attempt $RETRY_COUNT/$MAX_RETRIES)..."
    sleep 3
  done

  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "Error: Task $task_id failed after $MAX_RETRIES attempts."
    exit $EXIT_CODE
  fi
  
  # Remove auto-generated PEFT README.md if it exists inside the task folder
  if [[ -f "$TASK_DIR/README.md" ]]; then
    rm "$TASK_DIR/README.md"
  fi
  
  if [[ "$DRY_RUN" != "1" ]]; then
    touch "$TASK_DIR/.completed"
  fi
  
  echo "Completed training for Task $task_id"
done

echo "============================================="
echo "ALL LORA TRAINING COMPLETED SUCCESSFULLY"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "============================================="
