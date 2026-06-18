#!/usr/bin/env bash
set -euo pipefail

# Week 8: Residual-target diffusion training runner.
#
# Features:
# - Progress heartbeat with ETA and progress.log
# - Interrupt-safe execution with resume markers
# - Per-seed WandB run-id persistence for resume continuity
# - Optional power hardening and guaranteed restore
# - Aggregated summary.csv built from per-seed result rows

BASE_POLICY_PATH="${BASE_POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero_10}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
ACTION_DIM="${ACTION_DIM:-7}"
COND_DIM="${COND_DIM:-960}"
DIFF_HIDDEN_DIM="${DIFF_HIDDEN_DIM:-256}"
DIFF_LAYERS="${DIFF_LAYERS:-5}"
VIS_FREQ="${VIS_FREQ:-2}"
DEVICE="${DEVICE:-cuda}"

EVAL_TASK_IDS="${EVAL_TASK_IDS:-0 1 3}"
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
PREFLIGHT_BASELINE_CHECK="${PREFLIGHT_BASELINE_CHECK:-0}"
PREFLIGHT_EPISODES="${PREFLIGHT_EPISODES:-3}"
MIN_BASELINE_SUCCESS="${MIN_BASELINE_SUCCESS:-0.25}"
PREFLIGHT_TASK_IDS="${PREFLIGHT_TASK_IDS:-$EVAL_TASK_IDS}"

WANDB_PROJECT="${WANDB_PROJECT:-hybrid_diffusion_vla}"
WANDB_GROUP="${WANDB_GROUP:-week8_residual_target_train}"
WANDB_RESUME_POLICY="${WANDB_RESUME_POLICY:-allow}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-0}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
USE_POWER_HARDENING="${USE_POWER_HARDENING:-1}"

INTERRUPTED=0
TERMINATED=0
CLEANUP_DONE=0
CURRENT_CHILD_PID=""
PROCESSED_JOBS=0
TOTAL_JOBS=0
START_TS=0
PROGRESS_LOG=""

ORIG_POWER_PROFILE=""
ORIG_SLEEP_MODE=""
OS_NAME="$(uname -s)"

format_duration() {
  local total_s="$1"
  local h=$((total_s / 3600))
  local m=$(((total_s % 3600) / 60))
  local s=$((total_s % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

print_progress() {
  local seed="$1"
  local status="$2"
  local now elapsed remaining avg eta pct line

  now=$(date +%s)
  elapsed=$((now - START_TS))
  remaining=$((TOTAL_JOBS - PROCESSED_JOBS))

  if (( PROCESSED_JOBS > 0 )); then
    avg=$((elapsed / PROCESSED_JOBS))
  else
    avg=0
  fi
  eta=$((avg * remaining))
  pct="$(awk -v p="$PROCESSED_JOBS" -v t="$TOTAL_JOBS" 'BEGIN{if(t>0) printf "%.1f", 100.0*p/t; else printf "0.0"}')"

  line="[progress] ${PROCESSED_JOBS}/${TOTAL_JOBS} (${pct}%) | seed=${seed} | ${status} | elapsed=$(format_duration "$elapsed") | eta=$(format_duration "$eta")"
  echo "$line"
  if [[ -n "$PROGRESS_LOG" ]]; then
    echo "$line" >> "$PROGRESS_LOG"
  fi
}

print_heartbeat() {
  local seed="$1"
  local epoch_lines="$2"
  local line

  line="[heartbeat] seed=${seed} | epoch_lines=${epoch_lines} | wall=$(date -Iseconds)"
  echo "$line"
  if [[ -n "$PROGRESS_LOG" ]]; then
    echo "$line" >> "$PROGRESS_LOG"
  fi
}

apply_power_hardening() {
  [[ "$USE_POWER_HARDENING" == "1" ]] || return 0

  if [[ "$OS_NAME" == "Darwin" ]]; then
    if command -v pmset >/dev/null 2>&1; then
      ORIG_SLEEP_MODE="$(pmset -g custom | awk '/ sleep / {print $2; exit}' 2>/dev/null || true)"
      pmset -a sleep 0 2>/dev/null || true
    fi
    return 0
  fi

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

  if [[ "$OS_NAME" == "Darwin" ]]; then
    if [[ -n "$ORIG_SLEEP_MODE" ]] && command -v pmset >/dev/null 2>&1; then
      pmset -a sleep "$ORIG_SLEEP_MODE" 2>/dev/null || true
    fi
    return 0
  fi

  if [[ -n "$ORIG_POWER_PROFILE" ]] && command -v powerprofilesctl >/dev/null 2>&1; then
    powerprofilesctl set "$ORIG_POWER_PROFILE" 2>/dev/null || true
  fi

  if [[ -n "$ORIG_SLEEP_MODE" ]] && command -v gsettings >/dev/null 2>&1; then
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type "$ORIG_SLEEP_MODE" 2>/dev/null || true
  fi
}

aggregate_results() {
  local summary_csv="$OUTPUT_ROOT/summary.csv"

  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$summary_csv" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = Path(sys.argv[2])

rows = []
for row_file in sorted(root.rglob("result_row.csv")):
    try:
        with row_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        continue

dedup = {}
for row in rows:
    unit = row.get("unit", "")
    dedup[unit] = row

fieldnames = [
    "unit",
    "seed",
    "overall_success",
    "exit_code",
    "log_file",
    "run_dir",
    "wandb_run_id",
]

with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for key in sorted(dedup.keys()):
        writer.writerow({k: dedup[key].get(k, "") for k in fieldnames})

print(f"Wrote {len(dedup)} deduplicated rows to {summary}")
PY
}

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return 0
  fi
  CLEANUP_DONE=1

  aggregate_results || true
  restore_power_hardening || true
}

on_interrupt() {
  if [[ "$INTERRUPTED" -eq 0 ]]; then
    INTERRUPTED=1
    echo "[interrupt] Ctrl+C received. Will stop after current seed finishes." | tee -a "$PROGRESS_LOG"
  else
    echo "[interrupt] second Ctrl+C received; exiting immediately." | tee -a "$PROGRESS_LOG"
    exit 130
  fi
}

on_term() {
  TERMINATED=1
  echo "[term] termination requested; shutting down now." | tee -a "$PROGRESS_LOG"
  if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    kill -TERM "$CURRENT_CHILD_PID" 2>/dev/null || true
  fi
  exit 143
}

trap on_interrupt INT
trap on_term TERM
trap cleanup EXIT

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="week8_diffusion_residual_train_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/week8_diffusion_residual_train/${RUN_NAME}}"
mkdir -p "$OUTPUT_ROOT"
PROGRESS_LOG="$OUTPUT_ROOT/progress.log"

if [[ "$RESUME" != "1" ]]; then
  find "$OUTPUT_ROOT" -name .completed -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name result_row.csv -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name wandb_run_id.txt -delete 2>/dev/null || true
fi

{
  echo "[progress] run_start=$(date -Iseconds)"
  echo "[progress] output_root=$OUTPUT_ROOT"
  echo "[progress] resume=$RESUME"
  echo "[progress] seeds=$SEEDS"
} > "$PROGRESS_LOG"

set -- $SEEDS
TOTAL_JOBS=$#
START_TS=$(date +%s)

echo "Starting Week 8 residual-target diffusion training..."
echo "Output root: $OUTPUT_ROOT"
echo "Seeds: $SEEDS"
echo "Resume mode: $RESUME"
echo "Progress log: $PROGRESS_LOG"
echo "Heartbeat interval (sec): $HEARTBEAT_SEC"
echo "Total planned runs: $TOTAL_JOBS"

if [[ "$TOTAL_JOBS" -lt 1 ]]; then
  echo "ERROR: no seeds provided." >&2
  exit 1
fi

apply_power_hardening

for SEED in $SEEDS; do
  [[ "$INTERRUPTED" -eq 0 ]] || break
  [[ "$TERMINATED" -eq 0 ]] || break

  UNIT="seed_${SEED}"
  UNIT_DIR="$OUTPUT_ROOT/$UNIT"
  CHECKPOINT_DIR="$UNIT_DIR/checkpoints"
  LOG_FILE="$UNIT_DIR/train.log"
  DONE_MARKER="$UNIT_DIR/.completed"
  RESULT_ROW_FILE="$UNIT_DIR/result_row.csv"
  WANDB_RUN_ID_FILE="$UNIT_DIR/wandb_run_id.txt"

  mkdir -p "$UNIT_DIR" "$CHECKPOINT_DIR"

  if [[ "$RESUME" != "1" ]]; then
    rm -f "$DONE_MARKER" "$RESULT_ROW_FILE" "$WANDB_RUN_ID_FILE"
    rm -f "$CHECKPOINT_DIR/latest_checkpoint.pt" "$CHECKPOINT_DIR"/hybrid_diff_epoch_*.pt 2>/dev/null || true
  fi

  if [[ "$RESUME" == "1" && -f "$DONE_MARKER" && -f "$RESULT_ROW_FILE" ]]; then
    echo "[seed=$SEED] already completed. Skipping."
    PROCESSED_JOBS=$((PROCESSED_JOBS + 1))
    print_progress "$SEED" "skipped (already completed)"
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

  print_progress "$SEED" "starting"

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
    "--wandb_resume" "$WANDB_RESUME_POLICY"
    "--wandb_run_id_file" "$WANDB_RUN_ID_FILE"
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

  if [[ "$PREFLIGHT_BASELINE_CHECK" == "1" ]]; then
    CMD+=(
      "--preflight_baseline_check"
      "--preflight_episodes" "$PREFLIGHT_EPISODES"
      "--min_baseline_success" "$MIN_BASELINE_SUCCESS"
      "--preflight_task_ids" $PREFLIGHT_TASK_IDS
    )
  fi

  set +e
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE" &
  CURRENT_CHILD_PID=$!

  LAST_EPOCH_LINES=-1
  while kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; do
    sleep "$HEARTBEAT_SEC"
    if [[ -f "$LOG_FILE" ]]; then
      EPOCH_LINES=$(grep -c "Epoch " "$LOG_FILE" 2>/dev/null || true)
    else
      EPOCH_LINES=0
    fi
    if [[ "$EPOCH_LINES" != "$LAST_EPOCH_LINES" ]]; then
      print_heartbeat "$SEED" "$EPOCH_LINES"
      LAST_EPOCH_LINES="$EPOCH_LINES"
    else
      print_heartbeat "$SEED" "$EPOCH_LINES"
    fi
  done

  wait "$CURRENT_CHILD_PID"
  EXIT_CODE=$?
  CURRENT_CHILD_PID=""
  set -e

  if [[ "$EXIT_CODE" -eq 130 ]]; then
    INTERRUPTED=1
  fi

  OVERALL_SUCCESS="$($PYTHON_BIN - "$LOG_FILE" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
matches = re.findall(r"Eval Complete \| Overall Success Rate:\s*([0-9]*\.?[0-9]+)", text)
print(matches[-1] if matches else "NA")
PY
)"

  {
    echo "unit,seed,overall_success,exit_code,log_file,run_dir,wandb_run_id"
    echo "$UNIT,$SEED,$OVERALL_SUCCESS,$EXIT_CODE,$LOG_FILE,$UNIT_DIR,$WANDB_RUN_ID"
  } > "$RESULT_ROW_FILE"

  if [[ "$EXIT_CODE" -eq 0 && "$OVERALL_SUCCESS" != "NA" ]]; then
    touch "$DONE_MARKER"
  fi

  PROCESSED_JOBS=$((PROCESSED_JOBS + 1))
  if [[ "$EXIT_CODE" -eq 0 ]]; then
    print_progress "$SEED" "done (overall_success=$OVERALL_SUCCESS)"
  else
    print_progress "$SEED" "done (exit_code=$EXIT_CODE)"
  fi

done

if [[ "$INTERRUPTED" -eq 1 ]]; then
  echo "[interrupt] loop stopped gracefully. Re-run with RESUME=1 to continue."
fi

aggregate_results

echo "Week 8 residual-target diffusion training script complete."
echo "Summary CSV: $OUTPUT_ROOT/summary.csv"
