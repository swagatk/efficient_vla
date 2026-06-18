#!/usr/bin/env bash
set -euo pipefail

# Phase 1: Data Collection & Eval Harness runner
# Adheres to the create_new_script skill for robust, resumable jobs.

RESUME="${RESUME:-1}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_POWER_HARDENING="${USE_POWER_HARDENING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT"
PROGRESS_LOG="$OUTPUT_ROOT/progress.log"

ORIG_POWER_PROFILE=""
ORIG_SLEEP_MODE=""
OS_NAME="$(uname -s)"

INTERRUPTED=0
START_TS=$(date +%s)

if [[ "$RESUME" != "1" && -d "$OUTPUT_ROOT" ]]; then
  echo "[warn] OUTPUT_ROOT exists and RESUME=0; clearing stale markers" | tee -a "$PROGRESS_LOG"
  find "$OUTPUT_ROOT" -name .completed -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name result_row.csv -delete 2>/dev/null || true
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
        writer = csv.DictWriter(f, fieldnames=["unit", "exit_code"])
        writer.writeheader()
        for key in sorted(dedup.keys()):
            writer.writerow(dedup[key])
PY
}

analyze_results() {
  echo "[analyze] Analyzing Phase 1 evaluation results..." | tee -a "$PROGRESS_LOG"
  $PYTHON_BIN - "$OUTPUT_ROOT" <<'PY'
import json
import sys
import math
from pathlib import Path

root = Path(sys.argv[1])
all_successes = []

# Read all individual evaluation result files to aggregate success booleans
for p in root.rglob("task_*_results_*.json"):
    try:
        with p.open() as f:
            data = json.load(f)
            if 'seed_results' in data:
                for sr in data['seed_results']:
                    if 'per_episode_success' in sr:
                        all_successes.extend(sr['per_episode_success'])
    except Exception:
        pass

if all_successes:
    n = len(all_successes)
    p = sum(all_successes) / n
    
    # Calculate 95% Wald Confidence Interval
    z = 1.96
    se = math.sqrt((p * (1 - p)) / n) if n > 0 else 0
    ci_lower = max(0.0, p - z * se)
    ci_upper = min(1.0, p + z * se)

    results = {
        "overall_mean_success_rate": p,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "total_episodes": n,
        "total_successes": sum(all_successes)
    }

    out_file = root / "global_aggregate_results.json"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*40)
    print(" PHASE 1: BASELINE BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Episodes Analyzed: {n}")
    print(f"Overall Success Rate: {results['overall_mean_success_rate']:.2%}")
    print(f"95% CI: [{results['ci_95_lower']:.2%}, {results['ci_95_upper']:.2%}]")
    print("="*40 + "\n")
else:
    print("\n[analyze] No evaluation results found to analyze.\n")
PY
}

# Generate list of units (Task 0-9, Seed 0-2 for both collection and evaluation)
UNITS=()
for TASK_ID in {0..9}; do
  for SEED in 0 1 2; do
    UNITS+=("collect_t${TASK_ID}_s${SEED}" "eval_t${TASK_ID}_s${SEED}")
  done
done

TOTAL_UNITS=${#UNITS[@]}
DONE_UNITS=0

for UNIT in "${UNITS[@]}"; do
  [[ "$INTERRUPTED" -eq 0 ]] || break

  UNIT_DIR="$OUTPUT_ROOT/unit_${UNIT}"
  mkdir -p "$UNIT_DIR"
  DONE_MARKER="$UNIT_DIR/.completed"
  RESULT_FILE="$UNIT_DIR/result_row.csv"

  if [[ "$RESUME" != "1" ]]; then
    rm -f "$DONE_MARKER" "$RESULT_FILE"
  fi

  if [[ "$RESUME" == "1" && -f "$DONE_MARKER" && -f "$RESULT_FILE" ]]; then
    DONE_UNITS=$((DONE_UNITS + 1))
    print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} skipped"
    continue
  fi

  # Determine what script to run based on the unit name
  if [[ "$UNIT" == collect_* ]]; then
    TASK=$(echo "$UNIT" | sed -n 's/.*_t\([0-9]*\)_s.*/\1/p')
    SEED=$(echo "$UNIT" | sed -n 's/.*_s\([0-9]*\)/\1/p')
    CMD=( env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "-u" "$SCRIPT_DIR/collect_failure_data.py" --task_id "$TASK" --seed "$SEED" --num_episodes 10 --output_dir "$UNIT_DIR" )
  else
    TASK=$(echo "$UNIT" | sed -n 's/.*_t\([0-9]*\)_s.*/\1/p')
    SEED=$(echo "$UNIT" | sed -n 's/.*_s\([0-9]*\)/\1/p')
    CMD=( env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "-u" "$SCRIPT_DIR/eval_gated_baseline.py" --task_id "$TASK" --seed "$SEED" --output_dir "$UNIT_DIR" )
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

  echo "unit,exit_code" > "$RESULT_FILE"
  echo "$UNIT,$RC" >> "$RESULT_FILE"

  if [[ "$RC" -eq 0 ]]; then
    touch "$DONE_MARKER"
  fi

  DONE_UNITS=$((DONE_UNITS + 1))
  print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} done rc=${RC}"
done

aggregate_results
analyze_results

[[ "$INTERRUPTED" -eq 0 ]] || exit 130