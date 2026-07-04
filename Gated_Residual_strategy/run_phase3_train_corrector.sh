#!/usr/bin/env bash
set -euo pipefail

# Phase 3: Gated Residual Corrector Training Orchestrator
# Adheres to the create_new_script skill for robust, resumable jobs.

RESUME="${RESUME:-1}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
WANDB_PROJECT="${WANDB_PROJECT:-gated_residual_phase3}"
WANDB_RESUME_POLICY="${WANDB_RESUME_POLICY:-allow}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_POWER_HARDENING="${USE_POWER_HARDENING:-1}"
TRAIN_MODE="${TRAIN_MODE:-absolute}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/phase3_train_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_ROOT"
PROGRESS_LOG="$OUTPUT_ROOT/progress.log"

INTERRUPTED=0
START_TS=$(date +%s)
ORIG_POWER_PROFILE=""
ORIG_SLEEP_MODE=""
OS_NAME="$(uname -s)"
PID=""

if [[ "$RESUME" != "1" && -d "$OUTPUT_ROOT" ]]; then
  echo "[warn] OUTPUT_ROOT exists and RESUME=0; clearing stale markers" | tee -a "$PROGRESS_LOG"
  find "$OUTPUT_ROOT" -name .completed -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name result_row.csv -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name wandb_run_id.txt -delete 2>/dev/null || true
fi

on_interrupt() {
  if [[ "$INTERRUPTED" -eq 0 ]]; then
    INTERRUPTED=1
    echo "[interrupt] graceful stop requested" | tee -a "$PROGRESS_LOG"
  else
    echo "[interrupt] second interrupt, force exit" | tee -a "$PROGRESS_LOG"
    exit 130
  fi
}

on_term() {
  echo "[term] termination requested, exiting after cleanup" | tee -a "$PROGRESS_LOG"
  exit 143
}

apply_power_hardening() {
  [[ "$USE_POWER_HARDENING" == "1" ]] || return 0
  if [[ "$OS_NAME" == "Darwin" ]]; then
    if command -v pmset >/dev/null 2>&1; then
      ORIG_SLEEP_MODE="$(pmset -g custom | awk '/ sleep / {print $2; exit}' 2>/dev/null || true)"
      pmset -a sleep 0 2>/dev/null || true
    fi
  else
    if command -v powerprofilesctl >/dev/null 2>&1; then
      ORIG_POWER_PROFILE="$(powerprofilesctl get 2>/dev/null || true)"
      powerprofilesctl set performance 2>/dev/null || true
    fi
    if command -v gsettings >/dev/null 2>&1; then
      ORIG_SLEEP_MODE="$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || true)"
      gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
    fi
  fi
}

restore_power_hardening() {
  # Terminate background child process first if running
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi

  [[ "$USE_POWER_HARDENING" == "1" ]] || return 0
  if [[ "$OS_NAME" == "Darwin" ]]; then
    if [[ -n "$ORIG_SLEEP_MODE" ]] && command -v pmset >/dev/null 2>&1; then
      pmset -a sleep "$ORIG_SLEEP_MODE" 2>/dev/null || true
    fi
  else
    if [[ -n "$ORIG_POWER_PROFILE" ]] && command -v powerprofilesctl >/dev/null 2>&1; then
      powerprofilesctl set "$ORIG_POWER_PROFILE" 2>/dev/null || true
    fi
    if [[ -n "$ORIG_SLEEP_MODE" ]] && command -v gsettings >/dev/null 2>&1; then
      gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type "$ORIG_SLEEP_MODE" 2>/dev/null || true
    fi
  fi
}

trap on_interrupt INT
trap on_term TERM
trap restore_power_hardening EXIT
apply_power_hardening

print_progress() {
  local done="$1" total="$2" tag="$3"
  local now elapsed
  now=$(date +%s)
  elapsed=$((now - START_TS))
  echo "[progress] ${done}/${total} | ${tag} | elapsed=${elapsed}s" | tee -a "$PROGRESS_LOG"
}

aggregate_results() {
  local summary_csv="$OUTPUT_ROOT/summary.csv"
  $PYTHON_BIN - "$OUTPUT_ROOT" "$summary_csv" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = Path(sys.argv[2])
rows = []
for p in sorted(root.rglob("result_row.csv")):
    try:
        with p.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except Exception:
        pass

dedup = {}
for r in rows:
    unit = r.get("unit", "")
    dedup[unit] = r

with summary.open("w", newline="") as f:
    if dedup:
        fieldnames = set()
        for r in dedup.values():
            fieldnames.update(r.keys())
        ordered_fields = ["unit", "exit_code"] + sorted(list(fieldnames - {"unit", "exit_code"}))
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        for key in sorted(dedup.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9):
            writer.writerow(dedup[key])
PY
}

# We will train across 3 random seeds to verify model stability
SEEDS=(42 123 999)
TOTAL_UNITS=${#SEEDS[@]}
DONE_UNITS=0

for i in "${!SEEDS[@]}"; do
  [[ "$INTERRUPTED" -eq 0 ]] || break

  SEED="${SEEDS[$i]}"
  UNIT="train_seed_${SEED}"
  UNIT_DIR="$OUTPUT_ROOT/unit_${UNIT}"
  
  mkdir -p "$UNIT_DIR"
  DONE_MARKER="$UNIT_DIR/.completed"
  RESULT_FILE="$UNIT_DIR/result_row.csv"
  RUN_ID_FILE="$UNIT_DIR/wandb_run_id.txt"

  if [[ "$RESUME" != "1" ]]; then
    rm -f "$DONE_MARKER" "$RESULT_FILE" "$RUN_ID_FILE"
  fi

  if [[ "$RESUME" == "1" && -f "$DONE_MARKER" && -f "$RESULT_FILE" ]]; then
    DONE_UNITS=$((DONE_UNITS + 1))
    print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} skipped"
    continue
  fi

  RUN_ID=""
  if [[ -f "$RUN_ID_FILE" ]]; then
    RUN_ID="$(head -n1 "$RUN_ID_FILE" | tr -d '[:space:]')"
  fi

  CMD=(
    env PYTHONUNBUFFERED=1 WANDB_RUN_ID_FILE="$RUN_ID_FILE" "$PYTHON_BIN" "-u" "$SCRIPT_DIR/train_residual_corrector.py"
    --data_dir "$DATA_DIR"
    --output_dir "$UNIT_DIR"
    --seed "$SEED"
    --epochs 15
    --train_mode "$TRAIN_MODE"
    --wandb_project "$WANDB_PROJECT"
    --wandb_resume "$WANDB_RESUME_POLICY"
  )
  if [[ -n "$RUN_ID" ]]; then
    CMD+=(--wandb_run_id "$RUN_ID")
  fi

  print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} starting"
  print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} heartbeat-start"

  "${CMD[@]}" > >( tee "$UNIT_DIR/run.log" ) 2>&1 &
  PID=$!

  while kill -0 "$PID" 2>/dev/null; do
    sleep "$HEARTBEAT_SEC"
    print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} running"
  done

  if wait "$PID"; then
    RC=0
  else
    RC=$?
  fi

  if [[ ! -f "$RUN_ID_FILE" ]]; then
    echo "child process must write wandb id to \$WANDB_RUN_ID_FILE" | tee -a "$PROGRESS_LOG"
  fi

  METRICS_FILE="$UNIT_DIR/training_metrics.csv"
  if [[ -f "$METRICS_FILE" ]]; then
    HEADER=$(head -n 1 "$METRICS_FILE")
    VALUES=$(tail -n 1 "$METRICS_FILE")
    echo "unit,exit_code,${HEADER}" > "$RESULT_FILE"
    echo "$UNIT,$RC,${VALUES}" >> "$RESULT_FILE"
  else
    echo "unit,exit_code" > "$RESULT_FILE"
    echo "$UNIT,$RC" >> "$RESULT_FILE"
  fi

  if [[ "$RC" -eq 0 ]]; then
    touch "$DONE_MARKER"
  fi

  DONE_UNITS=$((DONE_UNITS + 1))
  print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} done rc=${RC}"
done

aggregate_results

[[ "$INTERRUPTED" -eq 0 ]] || exit 130
