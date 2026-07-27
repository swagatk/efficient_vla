#!/usr/bin/env bash
set -euo pipefail

# Phase 1: Data Collection & Eval Harness runner
# Adheres to the create_new_script skill for robust, resumable jobs.

RESUME="${RESUME:-1}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_POWER_HARDENING="${USE_POWER_HARDENING:-1}"
FAILURE_WINDOW="${FAILURE_WINDOW:-30}"
MAX_STEPS="${MAX_STEPS:-520}"
NUM_EPISODES="${NUM_EPISODES:-10}"
BENCHMARK="${BENCHMARK:-libero_10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/phase1_run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT"
PROGRESS_LOG="$OUTPUT_ROOT/progress.log"

# Save execution configuration (including all flags and defaults) to config.json
"$PYTHON_BIN" - <<PY
import json, os

config = {
    "resume": int("${RESUME}"),
    "heartbeat_sec": int("${HEARTBEAT_SEC}"),
    "python_bin": "${PYTHON_BIN}",
    "use_power_hardening": int("${USE_POWER_HARDENING}"),
    "failure_window": int("${FAILURE_WINDOW}"),
    "max_steps": int("${MAX_STEPS}"),
    "num_episodes": int("${NUM_EPISODES}"),
    "benchmark": "${BENCHMARK}",
    "mujoco_gl": os.environ.get("MUJOCO_GL", "egl"),
    "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM", "egl"),
    "output_root": "${OUTPUT_ROOT}",
    "script_dir": "${SCRIPT_DIR}"
}

with open(os.path.join("${OUTPUT_ROOT}", "config.json"), "w") as f:
    json.dump(config, f, indent=4)
PY

ORIG_POWER_PROFILE=""
ORIG_SLEEP_MODE=""
POWER_RESTORED=0
OS_NAME="$(uname -s)"
PID=""

INTERRUPTED=0
START_TS=$(date +%s)

if [[ "$RESUME" != "1" && -d "$OUTPUT_ROOT" ]]; then
  echo "[warn] OUTPUT_ROOT exists and RESUME=0; clearing stale markers" | tee -a "$PROGRESS_LOG"
  find "$OUTPUT_ROOT" -name .completed -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name result_row.csv -delete 2>/dev/null || true
fi

on_interrupt() {
  echo "[interrupt] Ctrl+C pressed, cleaning up and exiting..." | tee -a "$PROGRESS_LOG"
  restore_power_hardening
  exit 130
}

on_term() {
  echo "[term] termination requested, cleaning up and exiting..." | tee -a "$PROGRESS_LOG"
  restore_power_hardening
  exit 143
}

apply_power_hardening() {
  [[ "$USE_POWER_HARDENING" == "1" ]] || return 0
  local state_file="$OUTPUT_ROOT/.orig_power_state"
  
  if [[ "$RESUME" == "1" && -f "$state_file" ]]; then
    # Load original values from the state file
    # shellcheck disable=SC1090
    source "$state_file"
  fi

  if [[ "$OS_NAME" == "Darwin" ]]; then
    if command -v pmset >/dev/null 2>&1; then
      if [[ -z "${ORIG_SLEEP_MODE:-}" ]]; then
        ORIG_SLEEP_MODE="$(pmset -g custom | awk '/ sleep / {print $2; exit}' 2>/dev/null || true)"
        echo "ORIG_SLEEP_MODE=\"$ORIG_SLEEP_MODE\"" > "$state_file"
      fi
      pmset -a sleep 0 2>/dev/null || true
    fi
  else
    # Initialize state file
    [[ -f "$state_file" ]] || touch "$state_file"
    
    if command -v powerprofilesctl >/dev/null 2>&1; then
      if [[ -z "${ORIG_POWER_PROFILE:-}" ]]; then
        ORIG_POWER_PROFILE="$(powerprofilesctl get 2>/dev/null || true)"
        echo "ORIG_POWER_PROFILE=\"$ORIG_POWER_PROFILE\"" >> "$state_file"
      fi
      powerprofilesctl set performance 2>/dev/null || true
    fi
    if command -v gsettings >/dev/null 2>&1; then
      if [[ -z "${ORIG_SLEEP_MODE:-}" ]]; then
        ORIG_SLEEP_MODE="$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || true)"
        echo "ORIG_SLEEP_MODE=\"$ORIG_SLEEP_MODE\"" >> "$state_file"
      fi
      gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
    fi
  fi
}

restore_power_hardening() {
  if [[ "${POWER_RESTORED:-0}" -eq 1 ]]; then
    return 0
  fi
  POWER_RESTORED=1

  # Ignore signals during cleanup to prevent interrupting the restoration process
  trap "" INT TERM

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
from collections import defaultdict

root = Path(sys.argv[1])
all_successes = []
task_successes = defaultdict(list)
task_names = {}

# Read all individual evaluation result files to aggregate success booleans
for p in root.rglob("evaluation_report_*.json"):
    try:
        with p.open() as f:
            data = json.load(f)
            if 'tasks' in data:
                for task_id, task_data in data['tasks'].items():
                    if 'task_name' in task_data and task_id not in task_names:
                        task_names[task_id] = task_data['task_name']
                    if 'runs' in task_data:
                        for run in task_data['runs']:
                            if 'episode_details' in run:
                                for ep in run['episode_details']:
                                    if 'success' in ep:
                                        all_successes.append(ep['success'])
                                        task_successes[task_id].append(ep['success'])
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

    task_results = {}
    for tid in sorted(task_successes.keys(), key=lambda x: int(x) if x.isdigit() else x):
        t_list = task_successes[tid]
        t_n = len(t_list)
        t_p = sum(t_list) / t_n if t_n > 0 else 0.0
        t_se = math.sqrt((t_p * (1 - t_p)) / t_n) if t_n > 0 else 0
        task_results[str(tid)] = {
            "task_name": task_names.get(tid, f"Task_{tid}"),
            "success_rate": t_p,
            "ci_95_lower": max(0.0, t_p - z * t_se),
            "ci_95_upper": min(1.0, t_p + z * t_se),
            "total_episodes": t_n,
            "total_successes": sum(t_list)
        }

    results = {
        "overall_mean_success_rate": p,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "total_episodes": n,
        "total_successes": sum(all_successes),
        "task_results": task_results
    }

    out_file = root / "global_aggregate_results.json"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(" PHASE 1: BASELINE BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Task ID':<10} {'Task Name':<35} {'Success Rate':<15} {'Episodes':<10}")
    print("-" * 70)
    for tid, t_data in task_results.items():
        name_trunc = (t_data['task_name'][:32] + '...') if len(t_data['task_name']) > 35 else t_data['task_name']
        print(f"{tid:<10} {name_trunc:<35} {t_data['success_rate']:<15.2%} {t_data['total_episodes']:<10}")
    print("-" * 70)
    print(f"Total Episodes Analyzed: {n}")
    print(f"Overall Success Rate: {results['overall_mean_success_rate']:.2%}")
    print(f"95% CI: [{results['ci_95_lower']:.2%}, {results['ci_95_upper']:.2%}]")
    print("="*70 + "\n")
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
    CMD=( env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "-u" "$SCRIPT_DIR/collect_failure_data.py" --task_id "$TASK" --seed "$SEED" --num_episodes "$NUM_EPISODES" --max_steps "$MAX_STEPS" --output_dir "$UNIT_DIR" --failure_window "$FAILURE_WINDOW" )
  else
    TASK=$(echo "$UNIT" | sed -n 's/.*_t\([0-9]*\)_s.*/\1/p')
    SEED=$(echo "$UNIT" | sed -n 's/.*_s\([0-9]*\)/\1/p')
    CMD=( env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "-u" "$SCRIPT_DIR/eval_gated_baseline.py" --task_id "$TASK" --seed "$SEED" --num_episodes "$NUM_EPISODES" --max_steps "$MAX_STEPS" --benchmark "$BENCHMARK" --output_dir "$UNIT_DIR" )
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