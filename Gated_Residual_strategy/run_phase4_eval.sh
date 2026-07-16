#!/usr/bin/env bash
set -euo pipefail

# Phase 4: Gated Residual Strategy Evaluation Orchestrator
# Coordinates evaluation of the gated policy task-by-task to avoid memory leaks.

RESUME="${RESUME:-1}"
GATE_DIR="${GATE_DIR:-}"
CORRECTOR_DIR="${CORRECTOR_DIR:-}"
THRESHOLD="${THRESHOLD:-0.5}"
ALPHA="${ALPHA:-0.5}"
NUM_EPISODES="${NUM_EPISODES:-10}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_POWER_HARDENING="${USE_POWER_HARDENING:-1}"
INFERENCE_MODE="${INFERENCE_MODE:-absolute}"
BENCHMARK="${BENCHMARK:-libero_10}"
ADAPTIVE_GATING="${ADAPTIVE_GATING:-0}"
ACTIVE_GATING_TASKS="${ACTIVE_GATING_TASKS:-2 7 9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/phase4_eval_results_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"
PROGRESS_LOG="$OUTPUT_DIR/progress.log"

INTERRUPTED=0
START_TS=$(date +%s)
ORIG_POWER_PROFILE=""
ORIG_SLEEP_MODE=""
POWER_RESTORED=0
OS_NAME="$(uname -s)"
PID=""

if [[ "$RESUME" != "1" && -d "$OUTPUT_DIR" ]]; then
  echo "[warn] OUTPUT_DIR exists and RESUME=0; clearing stale markers" | tee -a "$PROGRESS_LOG"
  find "$OUTPUT_DIR" -name .completed -delete 2>/dev/null || true
fi

on_interrupt() {
  echo "[interrupt] Ctrl+C pressed, cleaning up and exiting..." | tee -a "$PROGRESS_LOG"
  echo "" | tee -a "$PROGRESS_LOG"
  echo "====================================================================" | tee -a "$PROGRESS_LOG"
  echo "EVALUATION INTERRUPTED" | tee -a "$PROGRESS_LOG"
  echo "====================================================================" | tee -a "$PROGRESS_LOG"
  echo "To resume this specific evaluation run later, execute:" | tee -a "$PROGRESS_LOG"
  echo "  OUTPUT_DIR=$OUTPUT_DIR bash run_phase4_eval.sh" | tee -a "$PROGRESS_LOG"
  echo "====================================================================" | tee -a "$PROGRESS_LOG"
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
  local output_dir="$1"
  echo "Aggregating individual evaluation reports..." | tee -a "$PROGRESS_LOG"
  
  "$PYTHON_BIN" - "$output_dir" <<'PY'
import sys
import json
import numpy as np
from pathlib import Path

output_dir = Path(sys.argv[1])
reports = []
for p in output_dir.rglob("evaluation_report_*.json"):
    if p.name == "evaluation_report_aggregate.json":
        continue
    try:
        with p.open() as f:
            reports.append(json.load(f))
    except Exception:
        pass

if not reports:
    print("No evaluation reports found to aggregate.")
    sys.exit(0)

all_runs = []
config = None
for r in reports:
    if not config and "config" in r:
        config = r["config"]
    for task_id, task_data in r.get("tasks", {}).items():
        all_runs.extend(task_data.get("runs", []))

task_success_rates = []
task_summaries = {}
unique_task_ids = sorted(list(set(run["task_id"] for run in all_runs)))

def calculate_confidence_interval(success_rates):
    if len(success_rates) < 2:
        return np.mean(success_rates), 0.0, 0.0
    mean = np.mean(success_rates)
    std = np.std(success_rates, ddof=1)
    margin = 1.96 * (std / np.sqrt(len(success_rates)))
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)

for task_id in unique_task_ids:
    task_runs = [r for r in all_runs if r["task_id"] == task_id]
    rates = [r["success_rate"] for r in task_runs]
    mean, ci_lower, ci_upper = calculate_confidence_interval(rates)
    task_success_rates.append(mean)
    
    task_summaries[str(task_id)] = {
        "task_name": task_runs[0]["task_name"],
        "success_rate_mean": mean,
        "success_rate_ci_95": [ci_lower, ci_upper],
        "runs": task_runs
    }

overall_mean, overall_ci_lower, overall_ci_upper = calculate_confidence_interval(task_success_rates)

aggregate_report = {
    "overall_mean_success_rate": overall_mean,
    "ci_95_lower": overall_ci_lower,
    "ci_95_upper": overall_ci_upper,
    "config": config,
    "tasks": task_summaries
}

aggregate_path = output_dir / "evaluation_report_aggregate.json"
with aggregate_path.open("w") as f:
    json.dump(aggregate_report, f, indent=2)

print("\n" + "="*60)
print("EVALUATION COMPLETED AND AGGREGATED")
print("="*60)
print(f"Overall Success Rate: {overall_mean:.2%}")
print(f"95% Confidence Interval: [{overall_ci_lower:.2%}, {overall_ci_upper:.2%}]")
print(f"Aggregated report saved to: {aggregate_path}")
print("="*60)
PY
}

# We iterate over tasks and seeds
TASKS=( ${TASKS:-0 1 2 3 4 5 6 7 8 9} )
SEEDS=( ${SEEDS:-0 1 2} )
TOTAL_RUNS=$(( ${#TASKS[@]} * ${#SEEDS[@]} ))
DONE_RUNS=0

echo "Starting Gated Residual Strategy Evaluation..." | tee -a "$PROGRESS_LOG"
echo "Outputs will be saved in: $OUTPUT_DIR" | tee -a "$PROGRESS_LOG"
echo "To resume this run if interrupted, execute with:" | tee -a "$PROGRESS_LOG"
echo "  OUTPUT_DIR=$OUTPUT_DIR bash run_phase4_eval.sh" | tee -a "$PROGRESS_LOG"
echo "----------------------------------------------------" | tee -a "$PROGRESS_LOG"

for seed in "${SEEDS[@]}"; do
  for task_id in "${TASKS[@]}"; do
    [[ "$INTERRUPTED" -eq 0 ]] || break

    UNIT="task${task_id}_seed${seed}"
    UNIT_DIR="$OUTPUT_DIR/unit_${UNIT}"
    mkdir -p "$UNIT_DIR"
    DONE_MARKER="$UNIT_DIR/.completed"
    
    if [[ "$RESUME" == "1" && -f "$DONE_MARKER" ]]; then
      DONE_RUNS=$((DONE_RUNS + 1))
      print_progress "$DONE_RUNS" "$TOTAL_RUNS" "unit=${UNIT} skipped"
      continue
    fi

    print_progress "$DONE_RUNS" "$TOTAL_RUNS" "unit=${UNIT} starting"
    
    CMD=(
      "$PYTHON_BIN" "$SCRIPT_DIR/eval_gated_baseline.py"
      --task_id "$task_id"
      --seed "$seed"
      --threshold "$THRESHOLD"
      --alpha "$ALPHA"
      --num_episodes "$NUM_EPISODES"
      --inference_mode "$INFERENCE_MODE"
      --output_dir "$UNIT_DIR"
      --benchmark "$BENCHMARK"
    )
    if [[ -n "$GATE_DIR" ]]; then
      CMD+=(--gate_dir "$GATE_DIR")
    fi
    if [[ -n "$CORRECTOR_DIR" ]]; then
      CMD+=(--corrector_dir "$CORRECTOR_DIR")
    fi
    if [[ "$ADAPTIVE_GATING" == "1" ]]; then
      CMD+=(--adaptive_gating --active_gating_tasks $ACTIVE_GATING_TASKS)
    fi

    # Run in background to track PID and handle interrupts properly
    "${CMD[@]}" > >( tee "$UNIT_DIR/run.log" ) 2>&1 &
    PID=$!

    while kill -0 "$PID" 2>/dev/null; do
      sleep "$HEARTBEAT_SEC"
      print_progress "$DONE_RUNS" "$TOTAL_RUNS" "unit=${UNIT} running"
    done

    if wait "$PID"; then
      touch "$DONE_MARKER"
      RC=0
    else
      RC=$?
    fi

    DONE_RUNS=$((DONE_RUNS + 1))
    print_progress "$DONE_RUNS" "$TOTAL_RUNS" "unit=${UNIT} done rc=${RC}"
  done
done

# Run aggregation across all unit runs if not interrupted
if [[ "$INTERRUPTED" -eq 0 ]]; then
  aggregate_results "$OUTPUT_DIR"
fi

[[ "$INTERRUPTED" -eq 0 ]] || exit 130
