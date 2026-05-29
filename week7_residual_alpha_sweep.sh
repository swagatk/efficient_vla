#!/usr/bin/env bash
set -euo pipefail

# Week 6: Tight residual-alpha sweep for hybrid diffusion eval.
#
# What this script does:
# 1) Runs eval-only residual mode across a small alpha grid.
# 2) Stores per-alpha logs under a timestamped output directory.
# 3) Extracts overall/task success metrics into a consolidated CSV.
# 4) Writes a short report with the best alpha found.
#
# Usage:
#   bash week7_residual_alpha_sweep.sh
#
# Optional overrides:
#   ALPHAS="0.02 0.05 0.1 0.2 0.3" EVAL_EPISODES=30 EVAL_TASK_IDS="0 1 3" \
#   REPEATS=3 \
#   CHECKPOINT_PATH=checkpoints_hybrid_diffusion/hybrid_diff_epoch_020.pt \
#   bash week7_residual_alpha_sweep.sh

BASE_POLICY_PATH="${BASE_POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints_hybrid_diffusion/hybrid_diff_epoch_020.pt}"
ALPHAS="${ALPHAS:-0.02 0.05 0.1 0.2 0.3}"
REPEATS="${REPEATS:-1}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0 1 3}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
IMAGE_FLIP_MODE="${IMAGE_FLIP_MODE:-vertical_horizontal}"
EVAL_DIFFUSION_STEPS="${EVAL_DIFFUSION_STEPS:-10}"
EVAL_ACTION_CLIP="${EVAL_ACTION_CLIP:-1.0}"
EVAL_LOG_ACTION_STATS_EVERY="${EVAL_LOG_ACTION_STATS_EVERY:-0}"
EVAL_REPLAN_EACH_STEP="${EVAL_REPLAN_EACH_STEP:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-hybrid_diffusion_vla}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"

INTERRUPTED=0
PROCESSED_JOBS=0
START_TS=0
TOTAL_JOBS=0
PROGRESS_LOG=""

format_duration() {
  local total_s="$1"
  local h=$((total_s / 3600))
  local m=$(((total_s % 3600) / 60))
  local s=$((total_s % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

print_progress() {
  local alpha="$1"
  local repeat_idx="$2"
  local status="$3"
  local now elapsed remaining avg eta
  local line

  now=$(date +%s)
  elapsed=$((now - START_TS))
  remaining=$((TOTAL_JOBS - PROCESSED_JOBS))

  if (( PROCESSED_JOBS > 0 )); then
    avg=$((elapsed / PROCESSED_JOBS))
  else
    avg=0
  fi
  eta=$((avg * remaining))

  line="[progress] ${PROCESSED_JOBS}/${TOTAL_JOBS} ($(awk -v p="$PROCESSED_JOBS" -v t="$TOTAL_JOBS" 'BEGIN{if(t>0) printf "%.1f", 100.0*p/t; else printf "0.0"}')%) | alpha=${alpha} repeat=${repeat_idx} | ${status} | elapsed=$(format_duration "$elapsed") | eta=$(format_duration "$eta")"
  echo "$line"
  if [[ -n "$PROGRESS_LOG" ]]; then
    echo "$line" >> "$PROGRESS_LOG"
  fi
}

print_heartbeat() {
  local alpha="$1"
  local repeat_idx="$2"
  local done_eps="$3"
  local total_eps="$4"
  local line

  line="[heartbeat] alpha=${alpha} repeat=${repeat_idx} | eval_episodes=${done_eps}/${total_eps} | wall=$(date -Iseconds)"
  echo "$line"
  if [[ -n "$PROGRESS_LOG" ]]; then
    echo "$line" >> "$PROGRESS_LOG"
  fi
}

on_interrupt() {
  if [[ "$INTERRUPTED" -eq 0 ]]; then
    INTERRUPTED=1
    echo ""
    echo "[interrupt] Ctrl+C received. Finishing current step and saving progress for resume..."
  else
    echo "[interrupt] second Ctrl+C received; exiting immediately."
    exit 130
  fi
}

trap on_interrupt INT TERM

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="week7_residual_alpha_sweep_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/week7_residual_alpha_sweep/${RUN_NAME}}"
mkdir -p "$OUTPUT_ROOT"
PROGRESS_LOG="${OUTPUT_ROOT}/progress.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSOLIDATED_DIR="${CONSOLIDATED_DIR:-$SCRIPT_DIR}"
mkdir -p "$CONSOLIDATED_DIR"

SUMMARY_CSV="${CONSOLIDATED_DIR}/week7_residual_alpha_sweep_summary.csv"
AGG_SUMMARY_CSV="${CONSOLIDATED_DIR}/week7_residual_alpha_sweep_aggregate.csv"
REPORT_TXT="${CONSOLIDATED_DIR}/week7_residual_alpha_sweep_report.txt"

echo "Starting Week 7 tight residual-alpha sweep..."
echo "Output root: $OUTPUT_ROOT"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Alphas: $ALPHAS"
echo "Repeats per alpha: $REPEATS"
echo "Eval task IDs: $EVAL_TASK_IDS"
echo "Resume mode: $RESUME"
echo "Progress log: $PROGRESS_LOG"
echo "Heartbeat interval (sec): $HEARTBEAT_SEC"

{
  echo "[progress] run_start=$(date -Iseconds)"
  echo "[progress] output_root=$OUTPUT_ROOT"
  echo "[progress] checkpoint_path=$CHECKPOINT_PATH"
} > "$PROGRESS_LOG"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "ERROR: REPEATS must be a positive integer, got: $REPEATS" >&2
  exit 1
fi

set -- $ALPHAS
ALPHA_COUNT=$#
TOTAL_JOBS=$((ALPHA_COUNT * REPEATS))
START_TS=$(date +%s)
set -- $EVAL_TASK_IDS
TASK_COUNT=$#
TOTAL_EVAL_EPISODES_PER_REPEAT=$((TASK_COUNT * EVAL_EPISODES))
echo "Total planned runs: $TOTAL_JOBS (${ALPHA_COUNT} alphas x ${REPEATS} repeats)"
echo "Episodes per repeat: $TOTAL_EVAL_EPISODES_PER_REPEAT (${TASK_COUNT} tasks x ${EVAL_EPISODES} episodes)"

for ALPHA in $ALPHAS; do
  if [[ "$INTERRUPTED" -eq 1 ]]; then
    break
  fi

  RUN_DIR="$OUTPUT_ROOT/alpha_${ALPHA}"
  mkdir -p "$RUN_DIR"

  echo ""
  echo "=========================================================="
  echo "Residual Alpha Sweep | alpha=$ALPHA | repeats=$REPEATS"
  echo "=========================================================="

  for REPEAT_IDX in $(seq 1 "$REPEATS"); do
    if [[ "$INTERRUPTED" -eq 1 ]]; then
      break
    fi

    REPEAT_DIR="$RUN_DIR/repeat_${REPEAT_IDX}"
    mkdir -p "$REPEAT_DIR"
    LOG_FILE="$REPEAT_DIR/eval.log"
    RESULT_ROW_FILE="$REPEAT_DIR/result_row.csv"
    COMPLETED_MARKER="$REPEAT_DIR/.completed"
    WANDB_RUN_ID_FILE="$REPEAT_DIR/wandb_run_id.txt"

    if [[ "$RESUME" == "1" && -f "$COMPLETED_MARKER" && -f "$RESULT_ROW_FILE" ]]; then
      echo "[alpha=$ALPHA] repeat ${REPEAT_IDX}/${REPEATS} already completed. Skipping."
      PROCESSED_JOBS=$((PROCESSED_JOBS + 1))
      print_progress "$ALPHA" "$REPEAT_IDX" "skipped (already completed)"
      continue
    fi

    echo ""
    echo "[alpha=$ALPHA] repeat ${REPEAT_IDX}/${REPEATS}"
    print_progress "$ALPHA" "$REPEAT_IDX" "starting"

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

    CMD=(
      env "PYTHONUNBUFFERED=1" "$PYTHON_BIN" -u "train_hybrid_diffusion.py"
      "--base_policy_path" "$BASE_POLICY_PATH"
      "--eval_only"
      "--eval_checkpoint_path" "$CHECKPOINT_PATH"
      "--eval_policy_mode" "residual"
      "--eval_residual_alpha" "$ALPHA"
      "--eval_diffusion_steps" "$EVAL_DIFFUSION_STEPS"
      "--eval_action_clip" "$EVAL_ACTION_CLIP"
      "--eval_log_action_stats_every" "$EVAL_LOG_ACTION_STATS_EVERY"
      "--eval_task_ids" $EVAL_TASK_IDS
      "--eval_episodes" "$EVAL_EPISODES"
      "--image_flip_mode" "$IMAGE_FLIP_MODE"
      "--wandb_project" "$WANDB_PROJECT"
      "--eval_wandb_run_id" "$WANDB_RUN_ID"
      "--eval_wandb_resume" "allow"
    )

    if [[ "$EVAL_REPLAN_EACH_STEP" == "1" ]]; then
      CMD+=("--eval_replan_each_step")
    fi

    set +e
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE" &
    PIPE_PID=$!

    LAST_DONE_EPS=-1
    while kill -0 "$PIPE_PID" 2>/dev/null; do
      sleep "$HEARTBEAT_SEC"
      if [[ -f "$LOG_FILE" ]]; then
        DONE_EPS=$(grep -c "Eval Ep " "$LOG_FILE" 2>/dev/null || true)
      else
        DONE_EPS=0
      fi
      if [[ "$DONE_EPS" != "$LAST_DONE_EPS" ]]; then
        print_heartbeat "$ALPHA" "$REPEAT_IDX" "$DONE_EPS" "$TOTAL_EVAL_EPISODES_PER_REPEAT"
        LAST_DONE_EPS="$DONE_EPS"
      else
        print_heartbeat "$ALPHA" "$REPEAT_IDX" "$DONE_EPS" "$TOTAL_EVAL_EPISODES_PER_REPEAT"
      fi
    done

    wait "$PIPE_PID"
    EXIT_CODE=$?
    set -e

    if [[ "$EXIT_CODE" -eq 130 ]]; then
      INTERRUPTED=1
    fi

    PARSED="$($PYTHON_BIN - "$LOG_FILE" "$EVAL_TASK_IDS" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
task_ids = [t for t in sys.argv[2].split() if t]
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

overall_matches = re.findall(r"Eval Complete \| Overall Success Rate:\s*([0-9]*\.?[0-9]+)", text)
overall = overall_matches[-1] if overall_matches else "NA"

# Pull the most recent task success for each requested task id.
task_rates = {}
for task_id in task_ids:
    matches = re.findall(rf"eval_task/task_{re.escape(task_id)}_success_rate\s+([0-9]*\.?[0-9]+)", text)
    task_rates[task_id] = matches[-1] if matches else "NA"

summary = ";".join([f"task_{tid}={task_rates[tid]}" for tid in task_ids]) if task_ids else "NA"
print(f"{overall}|{summary}")
PY
)"

    OVERALL_SUCCESS="NA"
    TASK_SUCCESS_SUMMARY="NA"
    if [[ "$PARSED" == *"|"* ]]; then
      OVERALL_SUCCESS="${PARSED%%|*}"
      TASK_SUCCESS_SUMMARY="${PARSED#*|}"
    fi

    {
      echo "alpha,repeat,overall_success,task_success_rates,exit_code,log_file,run_dir"
      echo "$ALPHA,$REPEAT_IDX,$OVERALL_SUCCESS,\"$TASK_SUCCESS_SUMMARY\",$EXIT_CODE,$LOG_FILE,$REPEAT_DIR"
    } > "$RESULT_ROW_FILE"

    if [[ "$EXIT_CODE" -eq 0 && "$OVERALL_SUCCESS" != "NA" ]]; then
      touch "$COMPLETED_MARKER"
    fi

    PROCESSED_JOBS=$((PROCESSED_JOBS + 1))
    if [[ "$EXIT_CODE" -eq 0 ]]; then
      print_progress "$ALPHA" "$REPEAT_IDX" "done (success=${OVERALL_SUCCESS})"
    else
      print_progress "$ALPHA" "$REPEAT_IDX" "done (exit_code=${EXIT_CODE})"
    fi

    if [[ "$INTERRUPTED" -eq 1 ]]; then
      echo "[interrupt] stopping sweep loops; rerun with same OUTPUT_ROOT to resume."
      break
    fi
  done
done

  $PYTHON_BIN - "$OUTPUT_ROOT" "$SUMMARY_CSV" <<'PY'
  import csv
  import sys
  from pathlib import Path

  output_root = Path(sys.argv[1])
  summary_csv = Path(sys.argv[2])

  rows = []
  for row_file in sorted(output_root.rglob("result_row.csv")):
    try:
      with row_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
          rows.append(row)
    except Exception:
      continue

  def _key(r):
    alpha = r.get("alpha", "")
    repeat = r.get("repeat", "")
    try:
      alpha_f = float(alpha)
    except Exception:
      alpha_f = 1e9
    try:
      repeat_i = int(repeat)
    except Exception:
      repeat_i = 1_000_000
    return (alpha_f, repeat_i)

  rows.sort(key=_key)

  with summary_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["alpha", "repeat", "overall_success", "task_success_rates", "exit_code", "log_file", "run_dir"])
    for r in rows:
      writer.writerow([
        r.get("alpha", "NA"),
        r.get("repeat", "NA"),
        r.get("overall_success", "NA"),
        r.get("task_success_rates", "NA"),
        r.get("exit_code", "NA"),
        r.get("log_file", "NA"),
        r.get("run_dir", "NA"),
      ])

  print(f"Wrote summary rows: {len(rows)} to {summary_csv}")
  PY

$PYTHON_BIN - "$SUMMARY_CSV" "$AGG_SUMMARY_CSV" "$REPORT_TXT" <<'PY'
import csv
import math
import statistics
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
agg_csv = Path(sys.argv[2])
report_txt = Path(sys.argv[3])

rows = []
with summary_csv.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def parse_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

valid = []
for r in rows:
  if str(r.get("exit_code", "")).strip() not in {"0", "0.0"}:
    continue
  score = parse_float(r.get("overall_success", "NA"))
  if math.isnan(score):
    continue
  r["_score"] = score
  valid.append(r)

alpha_to_scores = {}
for r in valid:
  alpha = r.get("alpha", "NA")
  alpha_to_scores.setdefault(alpha, []).append(r["_score"])

aggregates = []
for alpha, scores in alpha_to_scores.items():
  n = len(scores)
  mean = sum(scores) / n
  median = statistics.median(scores)
  if n > 1:
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    std = math.sqrt(var)
    ci95_half_width = 1.96 * std / math.sqrt(n)
  else:
    std = 0.0
    ci95_half_width = 0.0
  aggregates.append({
    "alpha": alpha,
    "n": n,
    "mean": mean,
    "std": std,
    "median": median,
    "ci95_half_width": ci95_half_width,
  })

# Robust ranking: prioritize median, then mean.
aggregates.sort(key=lambda x: (x["median"], x["mean"]), reverse=True)

with agg_csv.open("w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow([
    "alpha",
    "num_valid_runs",
    "mean_overall_success",
    "std_overall_success",
    "median_overall_success",
    "ci95_half_width",
    "ci95_lower",
    "ci95_upper",
  ])
  for a in aggregates:
    writer.writerow([
      a["alpha"],
      a["n"],
      f"{a['mean']:.6f}",
      f"{a['std']:.6f}",
      f"{a['median']:.6f}",
      f"{a['ci95_half_width']:.6f}",
      f"{(a['mean'] - a['ci95_half_width']):.6f}",
      f"{(a['mean'] + a['ci95_half_width']):.6f}",
    ])

lines = []
lines.append("Week 7 Residual Alpha Sweep Report")
lines.append("=================================")
lines.append(f"Rows evaluated: {len(rows)}")
lines.append(f"Valid rows (exit=0 and parseable score): {len(valid)}")
lines.append("")

if aggregates:
  best = aggregates[0]
  lines.append(f"Best alpha (by median, then mean): {best['alpha']}")
  lines.append(f"Best mean overall success: {best['mean']:.6f}")
  lines.append(f"Best std overall success: {best['std']:.6f}")
  lines.append(f"Best median overall success: {best['median']:.6f}")
  lines.append(f"Best 95% CI half-width (mean): {best['ci95_half_width']:.6f}")
  lines.append(
    f"Best 95% CI (mean): [{(best['mean'] - best['ci95_half_width']):.6f}, {(best['mean'] + best['ci95_half_width']):.6f}]"
  )
  lines.append(f"Valid repeats for best alpha: {best['n']}")
  lines.append("")
  lines.append("Ranking (desc median, then mean):")
  for a in aggregates:
    lines.append(
      f"  alpha={a['alpha']} median={a['median']:.6f} mean={a['mean']:.6f} std={a['std']:.6f} ci95=[{(a['mean'] - a['ci95_half_width']):.6f}, {(a['mean'] + a['ci95_half_width']):.6f}] n={a['n']}"
    )
else:
  lines.append("No valid overall_success values were parsed from logs.")

report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo ""
echo "Sweep complete."
echo "Summary CSV: $SUMMARY_CSV"
echo "Aggregate CSV: $AGG_SUMMARY_CSV"
echo "Report: $REPORT_TXT"
echo "Output root: $OUTPUT_ROOT"
print_progress "-" "-" "final"

if [[ "$INTERRUPTED" -eq 1 ]]; then
  exit 130
fi
