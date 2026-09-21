#!/usr/bin/env bash
# run_hybrid_diffusion.sh
# Hybrid Frozen Brain Diffusion Hands training and evaluation script.
# Configured for Ubuntu WSL on Windows 11 and native Linux environments.
#
# Usage:
#   bash run_hybrid_diffusion.sh
# Dry run:
#   DRY_RUN=1 bash run_hybrid_diffusion.sh
# Resume:
#   RESUME=1 bash run_hybrid_diffusion.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

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

# Headless rendering backend for MuJoCo and PyOpenGL on WSL
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Add paths to PYTHONPATH
if [[ -d "$HOME/LIBERO" ]]; then
  export PYTHONPATH="$HOME/LIBERO:$SCRIPT_DIR:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
fi

# Ensure LIBERO config exists and points to valid directories
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
mkdir -p "$LIBERO_CONFIG_PATH"
"$PYTHON_BIN" - <<'PY'
import os
import importlib.util
from pathlib import Path
import yaml
try:
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    cfg_path = cfg_dir / "config.yaml"
    dataset_root = os.environ.get("DATASET_ROOT", os.path.expanduser("~/lerobot_datasets/libero_10"))

    candidates = [
        Path(os.path.expanduser("~/LIBERO/libero/libero")),
        Path(os.path.expanduser("~/LIBERO/libero")),
    ]
    for mod in ["libero.libero", "libero"]:
        try:
            spec = importlib.util.find_spec(mod)
            if spec and spec.origin:
                p = Path(spec.origin).resolve().parent
                candidates.extend([p, p / "libero"])
        except Exception:
            pass

    benchmark_root = None
    for c in candidates:
        if (c / "init_files").is_dir() and (c / "bddl_files").is_dir():
            benchmark_root = c
            break

    if benchmark_root is None:
        for c in candidates:
            if (c / "init_files").is_dir():
                benchmark_root = c
                break

    if benchmark_root is None:
        benchmark_root = Path(os.path.expanduser("~/LIBERO/libero/libero"))

    needs_write = True
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                curr = yaml.safe_load(f)
            if isinstance(curr, dict) and Path(curr.get("init_states", "")).is_dir():
                needs_write = False
        except Exception:
            pass

    if needs_write:
        config = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(Path(dataset_root).resolve()),
            "assets": str(benchmark_root / "assets"),
        }
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"[WSL Setup] Initialized valid LIBERO config at: {cfg_path}")
except Exception as e:
    print(f"[WSL Setup] Notice: LIBERO config validation: {e}")
PY

# --- Dataset Path Detection ---
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "/home/swagat/lerobot_datasets/libero_10" ]]; then
    DATA_DIR="/home/swagat/lerobot_datasets/libero_10"
  elif [[ -d "/home/swagat/libero_dataset/libero_10" ]]; then
    DATA_DIR="/home/swagat/libero_dataset/libero_10"
  elif [[ -d "/home/swagat/lerobot_datasets" ]]; then
    DATA_DIR="/home/swagat/lerobot_datasets"
  else
    DATA_DIR="lerobot/libero_10"
  fi
fi
export DATA_DIR
DATASET_REPO_ID="${DATASET_REPO_ID:-$DATA_DIR}"

# --- Training & Architecture Parameters ---
BASE_POLICY_PATH="${BASE_POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
ACTION_DIM="${ACTION_DIM:-7}"
COND_DIM="${COND_DIM:-960}"
DIFF_HIDDEN_DIM="${DIFF_HIDDEN_DIM:-256}"
DIFF_LAYERS="${DIFF_LAYERS:-5}"
VIS_FREQ="${VIS_FREQ:-5}"
DEVICE="${DEVICE:-cuda}"
TASK_ID="${TASK_ID:-}"
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-}"
if [[ -n "$TASK_ID" ]]; then
  TRAIN_TASK_IDS="$TASK_ID"
  EVAL_TASK_IDS="$TASK_ID"
fi
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0 1 2 4 6 7 8 9}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
IMAGE_FLIP_MODE="${IMAGE_FLIP_MODE:-vertical_horizontal}"
RESIDUAL_TARGET="${RESIDUAL_TARGET:-0}"
if [[ "$RESIDUAL_TARGET" == "1" ]]; then
  EVAL_POLICY_MODE="${EVAL_POLICY_MODE:-residual}"
  RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-1.0}"
else
  EVAL_POLICY_MODE="${EVAL_POLICY_MODE:-hybrid}"
  RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.0}"
fi
EVAL_DIFFUSION_STEPS="${EVAL_DIFFUSION_STEPS:-10}"
EVAL_ACTION_CLIP="${EVAL_ACTION_CLIP:-1.0}"
EVAL_LOG_ACTION_STATS_EVERY="${EVAL_LOG_ACTION_STATS_EVERY:-0}"
EVAL_REPLAN_EACH_STEP="${EVAL_REPLAN_EACH_STEP:-0}"
DELTA_L2_WEIGHT="${DELTA_L2_WEIGHT:-0.001}"
WANDB_PROJECT="${WANDB_PROJECT:-hybrid_diffusion_vla}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DECODER_CACHE_CLEAR_FREQ="${DECODER_CACHE_CLEAR_FREQ:-500}"

RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == "1" ]]; then
  EVAL_EPISODES=1
  EVAL_TASK_IDS="0"
  EPOCHS=1
fi

# --- Windows WSL Host Power Profile & Sleep Management ---
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
    ORIG_POWER_SCHEME=$($POWERCFG_BIN /getactivescheme 2>/dev/null | awk '{print $4}' | tr -d '\r')
    echo "  • Original Active Scheme: ${ORIG_POWER_SCHEME:-Unknown}"

    ORIG_STANDBY_AC=$($POWERCFG_BIN /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 2>/dev/null | grep -i "Current AC Power Setting Index" | awk '{print $NF}' | tr -d '\r' || true)
    echo "  • Original Standby AC Setting: ${ORIG_STANDBY_AC:-Default}"

    local high_perf_guid=""
    high_perf_guid=$($POWERCFG_BIN /list 2>/dev/null | grep -i "High performance" | awk '{print $4}' | head -n 1 | tr -d '\r' || true)
    if [[ -z "$high_perf_guid" ]]; then
      high_perf_guid=$($POWERCFG_BIN -duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>/dev/null | awk '{print $4}' | head -n 1 | tr -d '\r' || true)
    fi

    if [[ -n "$high_perf_guid" ]]; then
      $POWERCFG_BIN /setactive "$high_perf_guid" 2>/dev/null || true
      echo "  • Activated Windows High Performance scheme: $high_perf_guid"
    fi

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

# --- Background VRAM Watchdog ---
WATCHDOG_PID=""
if command -v ollama >/dev/null 2>&1; then
  ollama stop mdq100/qwen3.5-coder:35b >/dev/null 2>&1 || true
fi

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

INTERRUPTED=0
TERMINATED=0
CLEANUP_DONE=0
CURRENT_CHILD_PID=""

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then return 0; fi
  CLEANUP_DONE=1
  if [[ -n "${WATCHDOG_PID:-}" ]]; then
    kill -9 "$WATCHDOG_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$IS_WSL" == "1" ]]; then
    restore_wsl_power || true
  fi
}

on_interrupt() {
  if [[ "$INTERRUPTED" -eq 0 ]]; then
    INTERRUPTED=1
    echo "[interrupt] Ctrl+C received. Gracefully terminating current step..."
    if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
      kill -INT "$CURRENT_CHILD_PID" 2>/dev/null || true
    fi
  else
    echo "[interrupt] Immediate exit."
    exit 130
  fi
}

on_term() {
  TERMINATED=1
  if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    kill -TERM "$CURRENT_CHILD_PID" 2>/dev/null || true
  fi
  exit 143
}

trap on_interrupt INT
trap on_term TERM
trap cleanup EXIT

# Apply WSL Host power profile
if [[ "$IS_WSL" == "1" ]]; then
  setup_wsl_power
fi

# --- Output Directory & Resuming ---
if [[ -n "${RESUME_DIR:-}" && -d "${RESUME_DIR}" ]]; then
  OUTPUT_ROOT="${RESUME_DIR}"
  RESUME="1"
  echo "--- Resuming from directory: $OUTPUT_ROOT ---"
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="./outputs/run_hybrid_diffusion_${TIMESTAMP}"
  mkdir -p "$OUTPUT_ROOT"
  echo "--- Created new run directory: $OUTPUT_ROOT ---"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "=== Running in DRY RUN mode ==="
  EPOCHS=1
  BATCH_SIZE=4
  VIS_FREQ=1
  EVAL_EPISODES=1
  EVAL_TASK_IDS="0"
fi

CONFIG_PATH="$OUTPUT_ROOT/config.json"
cat <<EOF > "$CONFIG_PATH"
{
  "base_policy_path": "$BASE_POLICY_PATH",
  "dataset_repo_id": "$DATASET_REPO_ID",
  "data_dir": "$DATA_DIR",
  "seeds": "$SEEDS",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "lr": $LR,
  "chunk_size": $CHUNK_SIZE,
  "action_dim": $ACTION_DIM,
  "cond_dim": $COND_DIM,
  "diff_hidden_dim": $DIFF_HIDDEN_DIM,
  "diff_layers": $DIFF_LAYERS,
  "vis_freq": $VIS_FREQ,
  "device": "$DEVICE",
  "eval_task_ids": "$EVAL_TASK_IDS",
  "eval_episodes": $EVAL_EPISODES,
  "image_flip_mode": "$IMAGE_FLIP_MODE",
  "eval_policy_mode": "$EVAL_POLICY_MODE",
  "residual_alpha": $RESIDUAL_ALPHA,
  "eval_diffusion_steps": $EVAL_DIFFUSION_STEPS,
  "eval_action_clip": $EVAL_ACTION_CLIP,
  "eval_log_action_stats_every": $EVAL_LOG_ACTION_STATS_EVERY,
  "eval_replan_each_step": $EVAL_REPLAN_EACH_STEP,
  "residual_target": $RESIDUAL_TARGET,
  "delta_l2_weight": $DELTA_L2_WEIGHT,
  "wandb_project": "$WANDB_PROJECT",
  "dry_run": $( [[ "$DRY_RUN" == "1" ]] && echo "true" || echo "false" )
}
EOF

echo "==========================================================="
echo "Python binary: $PYTHON_BIN"
echo "Dataset path: $DATASET_REPO_ID"
echo "Base policy: $BASE_POLICY_PATH"
echo "Seeds: $SEEDS | Epochs: $EPOCHS | Batch size: $BATCH_SIZE"
echo "Action dim: $ACTION_DIM | Chunk size: $CHUNK_SIZE"
echo "Flip mode: $IMAGE_FLIP_MODE | Eval policy mode: $EVAL_POLICY_MODE"
echo "Residual target training: $RESIDUAL_TARGET"
echo "==========================================================="

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
    env "PYTHONUNBUFFERED=1" "PYTHONFAULTHANDLER=1" "TORCHDYNAMO_DISABLE=1" "TORCH_COMPILE_DISABLE=1" "PYTHONHASHSEED=$SEED" "$PYTHON_BIN" -u "$SCRIPT_DIR/train_hybrid_diffusion.py"
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
    "--video_backend" "$VIDEO_BACKEND"
    "--decoder_cache_clear_freq" "$DECODER_CACHE_CLEAR_FREQ"
    "--eval_task_ids" $EVAL_TASK_IDS
  )

  if [[ "$RESIDUAL_TARGET" == "1" ]]; then
    CMD+=("--residual_target")
  fi

  if [[ "$EVAL_REPLAN_EACH_STEP" == "1" ]]; then
    CMD+=("--eval_replan_each_step")
  fi

  if [[ -n "$TASK_ID" ]]; then
    CMD+=("--task_id" "$TASK_ID")
  elif [[ -n "$TRAIN_TASK_IDS" ]]; then
    CMD+=("--train_task_ids" $TRAIN_TASK_IDS)
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=("--dry_run")
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
    if [[ "$DRY_RUN" != "1" ]]; then
      touch "$DONE_MARKER"
    fi
    echo "Seed $SEED finished successfully!"
  else
    echo "Seed $SEED failed with exit code $EXIT_CODE"
  fi
done

echo "==========================================================="
echo "Hybrid Diffusion Training Complete."
echo "Results summary stored in: $SUMMARY_FILE"
echo "==========================================================="
