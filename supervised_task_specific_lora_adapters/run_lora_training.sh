#!/usr/bin/env bash
# run_lora_training.sh
# Ordinarily trains supervised LoRA adapters for SmolVLA across tasks.
# Supports both native Linux and Windows WSL (with automated Windows 11 power profile & sleep inhibition).
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

# --- Environment & Interpreter Detection ---
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

# Add local LIBERO clone to PYTHONPATH if present
if [[ -d "$HOME/LIBERO" ]]; then
  export PYTHONPATH="$HOME/LIBERO:${PYTHONPATH:-}"
fi

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "/home/swagat/lerobot_datasets/libero_10" ]]; then
    DATA_DIR="/home/swagat/lerobot_datasets/libero_10"
  elif [[ -d "/home/swagat/libero_dataset/libero_10" ]]; then
    DATA_DIR="/home/swagat/libero_dataset/libero_10"
  elif [[ -d "/home/swagat/lerobot_datasets" ]]; then
    DATA_DIR="/home/swagat/lerobot_datasets"
  else
    DATA_DIR="/home/swagat/lerobot_datasets/libero_10"
  fi
fi
export DATA_DIR

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
PATIENCE="${PATIENCE:-4}"

DRY_RUN="${DRY_RUN:-0}"

# --- Windows WSL Power Profile & Sleep Management ---
IS_WSL=0
if grep -qi "microsoft" /proc/version 2>/dev/null || [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  IS_WSL=1
fi

POWERCFG_BIN=""
ORIG_POWER_SCHEME=""
ORIG_STANDBY_AC=""

if [[ "$IS_WSL" == "1" ]]; then
  if command -v powercfg.exe >/dev/null 2>&1; then
    POWERCFG_BIN="powercfg.exe"
  elif [[ -x "/mnt/c/Windows/System32/powercfg.exe" ]]; then
    POWERCFG_BIN="/mnt/c/Windows/System32/powercfg.exe"
  fi
fi

setup_wsl_power() {
  if [[ -n "$POWERCFG_BIN" ]]; then
    echo "==========================================================="
    echo "⚡ [WSL] Configuring Windows 11 Host Power Profile"
    echo "==========================================================="
    # Query current active scheme GUID
    ORIG_POWER_SCHEME=$($POWERCFG_BIN /getactivescheme 2>/dev/null | awk '{print $4}' | tr -d '\r')
    echo "  • Original Active Scheme: ${ORIG_POWER_SCHEME:-Unknown}"

    # Query current AC standby setting index
    ORIG_STANDBY_AC=$($POWERCFG_BIN /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 2>/dev/null | grep -i "Current AC Power Setting Index" | awk '{print $NF}' | tr -d '\r' || true)
    echo "  • Original Standby AC Setting: ${ORIG_STANDBY_AC:-Default}"

    # Find or duplicate High Performance scheme
    local high_perf_guid=""
    high_perf_guid=$($POWERCFG_BIN /list 2>/dev/null | grep -i "High performance" | awk '{print $4}' | head -n 1 | tr -d '\r' || true)
    if [[ -z "$high_perf_guid" ]]; then
      high_perf_guid=$($POWERCFG_BIN -duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>/dev/null | awk '{print $4}' | head -n 1 | tr -d '\r' || true)
    fi

    if [[ -n "$high_perf_guid" ]]; then
      $POWERCFG_BIN /setactive "$high_perf_guid" 2>/dev/null || true
      echo "  • Activated Windows High Performance scheme: $high_perf_guid"
    fi

    # Disable standby sleep timeouts while training (0 = never sleep)
    $POWERCFG_BIN /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0 2>/dev/null || true
    $POWERCFG_BIN /setdcvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0 2>/dev/null || true
    $POWERCFG_BIN /setactive SCHEME_CURRENT 2>/dev/null || true
    $POWERCFG_BIN /change standby-timeout-ac 0 2>/dev/null || true
    echo "  • Disabled Windows Standby Sleep (timeout set to 0)"
    echo "==========================================================="
  fi
}

restore_wsl_power() {
  if [[ -n "$POWERCFG_BIN" ]]; then
    echo ""
    echo "==========================================================="
    echo "⚡ [WSL] Restoring Windows 11 Host Power Profile"
    echo "==========================================================="
    if [[ -n "$ORIG_POWER_SCHEME" ]]; then
      $POWERCFG_BIN /setactive "$ORIG_POWER_SCHEME" 2>/dev/null || true
      echo "  • Restored original Windows Power Scheme ($ORIG_POWER_SCHEME)"
    fi
    if [[ -n "$ORIG_STANDBY_AC" ]]; then
      $POWERCFG_BIN /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE "$ORIG_STANDBY_AC" 2>/dev/null || true
      $POWERCFG_BIN /setactive SCHEME_CURRENT 2>/dev/null || true
      echo "  • Restored original Standby AC setting ($ORIG_STANDBY_AC)"
    fi
    echo "==========================================================="
  fi
}

# --- Pre-flight Checks ---
if [[ ! -d "$DATA_DIR" ]]; then
  echo "================================================================================"
  echo "⚠️ ERROR: Demonstration dataset directory not found at:"
  echo "  $DATA_DIR"
  echo ""
  echo "Please verify the LIBERO-10 demonstration dataset (.hdf5 / .h5 files)."
  echo "You can specify a custom dataset location via:"
  echo "  DATA_DIR=/path/to/demonstrations bash run_lora_training.sh"
  echo "================================================================================"
  exit 1
fi

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
echo "Python binary: $PYTHON_BIN"
echo "Training tasks: ${TASKS[*]}"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "Learning rate: $LR"
echo "LoRA Rank: $LORA_RANK, Alpha: $LORA_ALPHA"
echo "============================================="

# Evict any existing Ollama models to ensure 100% free VRAM
if command -v ollama >/dev/null 2>&1; then
  ollama stop mdq100/qwen3.5-coder:35b >/dev/null 2>&1 || true
fi

# Activate Windows WSL power configuration
if [[ "$IS_WSL" == "1" ]]; then
  setup_wsl_power
fi

# Background VRAM watchdog: continuously monitors and evicts competing GPU models
WATCHDOG_PID=""
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

cleanup() {
  if [[ -n "${WATCHDOG_PID:-}" ]]; then
    kill -9 "$WATCHDOG_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$IS_WSL" == "1" ]]; then
    restore_wsl_power
  fi
}
trap cleanup EXIT INT TERM

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
    --patience "$PATIENCE"
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
    if [[ "$IS_WSL" == "1" ]]; then
      # On WSL, host power management and sleep prevention is handled by powercfg.exe directly
      "${CMD[@]}"
    elif command -v systemd-inhibit >/dev/null 2>&1; then
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

