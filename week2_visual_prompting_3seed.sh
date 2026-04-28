#!/usr/bin/env bash
set -euo pipefail

# Week 2: Visual Prompting (GroundingDINO) 3-seed evaluation
#
# What this script does:
# 1) Runs 3 seeded evaluations through eval_week2_visual_prompting.py.
# 2) Logs per-seed overall success and profiling hooks (peak VRAM, latency proxy, loop rate).
# 3) Logs per-task success rates for each seed.
# 4) Produces aggregate Week-2 summary reports comparable to Week-1 outputs.
#
# Usage:
#   bash week2_visual_prompting_3seed.sh
#
# Optional overrides:
#   SEEDS="0 1 2" N_EPISODES=10 EPISODE_HORIZON=520 \
#   TASK_SUITE=libero_10 OUTPUT_ROOT=~/lerobot/outputs/week2_visual_prompting/week2_visual_prompting_YYYYmmdd_HHMMSS \
#   bash week2_visual_prompting_3seed.sh

DATASET_ROOT="${DATASET_ROOT:-$HOME/libero_dataset}"
MODEL_ID="${MODEL_ID:-HuggingFaceVLA/smolvla_libero}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
N_EPISODES="${N_EPISODES:-10}"
EPISODE_HORIZON="${EPISODE_HORIZON:-520}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OBSERVATION_HEIGHT="${OBSERVATION_HEIGHT:-256}"
OBSERVATION_WIDTH="${OBSERVATION_WIDTH:-256}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-10}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
GROUNDING_DEVICE="${GROUNDING_DEVICE:-$POLICY_DEVICE}"
SEEDS="${SEEDS:-0 1 2}"
TARGET_SUCCESS="${TARGET_SUCCESS:-46.0}"

# If 1, pass a seed flag through to lerobot-eval via eval_week2_visual_prompting.py.
USE_EVAL_SEED_FLAG="${USE_EVAL_SEED_FLAG:-1}"
SEED_FLAG_MODE="${SEED_FLAG_MODE:-auto}" # auto|eval.seed|seed|none

# Visual prompting toggles (GroundingDINO wrapper uses these)
BOX_OVERLAY="${BOX_OVERLAY:-1}"
TEXT_HINT="${TEXT_HINT:-1}"

# Runtime mode for week2 eval entrypoint.
INPROCESS="${INPROCESS:-1}"

EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="week2_visual_prompting_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/lerobot/outputs/week2_visual_prompting/${RUN_NAME}}"
mkdir -p "$OUTPUT_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSOLIDATED_DIR="${CONSOLIDATED_DIR:-$SCRIPT_DIR}"
mkdir -p "$CONSOLIDATED_DIR"

WEEK2_ENTRYPOINT="$SCRIPT_DIR/eval_week2_visual_prompting.py"
if [[ ! -f "$WEEK2_ENTRYPOINT" ]]; then
  echo "ERROR: missing $WEEK2_ENTRYPOINT" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"

SUMMARY_CSV="$CONSOLIDATED_DIR/week2_runs_summary.csv"
TASK_SUMMARY_CSV="$CONSOLIDATED_DIR/week2_task_success_summary.csv"
AGGREGATE_REPORT="$CONSOLIDATED_DIR/week2_aggregate_report.txt"

# Comparable to Week-1 columns, plus task_success_rates and eval timing from eval_info.
echo "seed,success_rate,peak_vram_gb,avg_step_latency_ms,e2e_loop_rate_hz,wall_time_sec,exit_code,run_dir,eval_s,eval_ep_s,task_success_rates" > "$SUMMARY_CSV"
echo "seed,task_group,task_id,task_success_rate" > "$TASK_SUMMARY_CSV"

read -r -a EXTRA_ARGS_ARRAY <<< "$EXTRA_EVAL_ARGS"

RESOLVED_SEED_FLAG_MODE="none"
if [[ "$USE_EVAL_SEED_FLAG" == "1" ]]; then
  case "$SEED_FLAG_MODE" in
    auto)
      if lerobot-eval -h 2>&1 | grep -q -- '--eval.seed'; then
        RESOLVED_SEED_FLAG_MODE="eval.seed"
      elif lerobot-eval -h 2>&1 | grep -q -- '--seed'; then
        RESOLVED_SEED_FLAG_MODE="seed"
      else
        RESOLVED_SEED_FLAG_MODE="none"
      fi
      ;;
    eval.seed|seed|none)
      RESOLVED_SEED_FLAG_MODE="$SEED_FLAG_MODE"
      ;;
    *)
      echo "ERROR: invalid SEED_FLAG_MODE=$SEED_FLAG_MODE (use auto|eval.seed|seed|none)" >&2
      exit 1
      ;;
  esac
fi

echo "Starting Week-2 visual prompting runs..."
echo "Output root: $OUTPUT_ROOT"
echo "Seed flag mode: $RESOLVED_SEED_FLAG_MODE"

default_python="python3"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  default_python="$PYTHON_BIN"
fi

for SEED in $SEEDS; do
  RUN_DIR="$OUTPUT_ROOT/seed_${SEED}"
  EVAL_OUT_DIR="$RUN_DIR/eval_output"
  mkdir -p "$RUN_DIR" "$EVAL_OUT_DIR"

  LOG_FILE="$RUN_DIR/eval.log"
  GPU_LOG_CSV="$RUN_DIR/gpu_usage.csv"

  echo ""
  echo "===== Seed $SEED ====="

  GPU_MON_PID=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
      --format=csv -l 1 > "$GPU_LOG_CSV" 2>/dev/null &
    GPU_MON_PID=$!
  fi

  START_NS="$(date +%s%N)"

  CMD=(
    "$default_python" "$WEEK2_ENTRYPOINT"
    "--model_id=$MODEL_ID"
    "--task_suite=$TASK_SUITE"
    "--n_episodes=$N_EPISODES"
    "--batch_size=$BATCH_SIZE"
    "--output_dir=$EVAL_OUT_DIR"
    "--dataset_root=$DATASET_ROOT"
    "--libero_config_path=$LIBERO_CONFIG_PATH"
    "--max_episodes_rendered=$MAX_EPISODES_RENDERED"
    "--observation_height=$OBSERVATION_HEIGHT"
    "--observation_width=$OBSERVATION_WIDTH"
    "--policy_device=$POLICY_DEVICE"
    "--grounding_device=$GROUNDING_DEVICE"
    "--episode_horizon=$EPISODE_HORIZON"
  )

  if [[ "$BOX_OVERLAY" == "1" ]]; then
    CMD+=("--box_overlay")
  fi

  if [[ "$TEXT_HINT" == "1" ]]; then
    CMD+=("--text_hint")
  fi

  if [[ "$INPROCESS" == "0" ]]; then
    CMD+=("--no-inprocess")
  fi

  if [[ "$RESOLVED_SEED_FLAG_MODE" == "eval.seed" ]]; then
    CMD+=("--extra_eval_arg=--eval.seed=$SEED")
  elif [[ "$RESOLVED_SEED_FLAG_MODE" == "seed" ]]; then
    CMD+=("--extra_eval_arg=--seed=$SEED")
  fi

  for arg in "${EXTRA_ARGS_ARRAY[@]}"; do
    [[ -n "$arg" ]] && CMD+=("--extra_eval_arg=$arg")
  done

  set +e
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  END_NS="$(date +%s%N)"
  WALL_SEC="$(awk -v s="$START_NS" -v e="$END_NS" 'BEGIN{printf "%.6f", (e-s)/1000000000.0}')"

  if [[ -n "$GPU_MON_PID" ]]; then
    kill "$GPU_MON_PID" >/dev/null 2>&1 || true
    wait "$GPU_MON_PID" 2>/dev/null || true
  fi

  PEAK_VRAM_GB="NA"
  if [[ -f "$GPU_LOG_CSV" ]]; then
    PEAK_VRAM_GB="$(awk -F, 'NR>1 {gsub(/ MiB/, "", $6); if ($6+0>max) max=$6+0} END {if (max>0) printf "%.3f", max/1024.0; else print "NA"}' "$GPU_LOG_CSV")"
  fi

  TOTAL_STEPS=$((N_EPISODES * EPISODE_HORIZON))
  E2E_LOOP_RATE_HZ="$(awk -v steps="$TOTAL_STEPS" -v t="$WALL_SEC" 'BEGIN{if (t>0) printf "%.6f", steps/t; else print "NA"}')"
  AVG_STEP_LATENCY_MS="$(awk -v hz="$E2E_LOOP_RATE_HZ" 'BEGIN{if (hz+0>0) printf "%.6f", 1000.0/hz; else print "NA"}')"

  if [[ "$EXIT_CODE" -ne 0 ]]; then
    AVG_STEP_LATENCY_MS="NA"
    E2E_LOOP_RATE_HZ="NA"
  fi

  EVAL_INFO_JSON="$EVAL_OUT_DIR/eval_info.json"
  PARSED_FIELDS="$($default_python - "$EVAL_INFO_JSON" "$SEED" "$TASK_SUMMARY_CSV" <<'PY'
import json
import pathlib
import sys

info_path = pathlib.Path(sys.argv[1])
seed = sys.argv[2]
task_csv = pathlib.Path(sys.argv[3])

if not info_path.exists():
    print("NA|NA|NA|NA|NA")
    sys.exit(0)

try:
    data = json.loads(info_path.read_text())
except Exception:
    print("NA|NA|NA|NA|NA")
    sys.exit(0)

overall = data.get("overall", {})
pc_success = overall.get("pc_success", "NA")
eval_s = overall.get("eval_s", "NA")
eval_ep_s = overall.get("eval_ep_s", "NA")

per_task = data.get("per_task", [])
triples = []
for item in per_task:
    task_group = item.get("task_group", "unknown")
    task_id = item.get("task_id", "NA")
    successes = item.get("metrics", {}).get("successes", [])
    if successes:
        rate = 100.0 * sum(1 for x in successes if x) / len(successes)
        rate_text = f"{rate:.4f}"
    else:
        rate_text = "NA"
    triples.append((task_group, task_id, rate_text))

# Stable ordering by group then task id (numeric when possible).
def task_sort_key(t):
    group, task_id, _ = t
    try:
        tid = int(task_id)
    except Exception:
        tid = 10**9
    return (str(group), tid, str(task_id))

triples = sorted(triples, key=task_sort_key)

with task_csv.open("a") as f:
    for group, task_id, rate_text in triples:
        f.write(f"{seed},{group},{task_id},{rate_text}\n")

# Compact single-cell summary for seed table.
summary_parts = [f"{group}:{task_id}={rate_text}" for group, task_id, rate_text in triples]
summary = ";".join(summary_parts) if summary_parts else "NA"

if isinstance(pc_success, (int, float)):
    pc_text = f"{pc_success:.4f}"
else:
    pc_text = "NA"

if isinstance(eval_s, (int, float)):
    eval_s_text = f"{eval_s:.6f}"
else:
    eval_s_text = "NA"

if isinstance(eval_ep_s, (int, float)):
    eval_ep_s_text = f"{eval_ep_s:.6f}"
else:
    eval_ep_s_text = "NA"

print(f"{pc_text}|{eval_s_text}|{eval_ep_s_text}|{summary}|{len(triples)}")
PY
)"

  SUCCESS_RATE="NA"
  EVAL_S="NA"
  EVAL_EP_S="NA"
  TASK_SUCCESS_RATES="NA"
  TASK_COUNT="0"

  IFS='|' read -r SUCCESS_RATE EVAL_S EVAL_EP_S TASK_SUCCESS_RATES TASK_COUNT <<< "$PARSED_FIELDS"

  echo "$SEED,$SUCCESS_RATE,$PEAK_VRAM_GB,$AVG_STEP_LATENCY_MS,$E2E_LOOP_RATE_HZ,$WALL_SEC,$EXIT_CODE,$RUN_DIR,$EVAL_S,$EVAL_EP_S,$TASK_SUCCESS_RATES" >> "$SUMMARY_CSV"

  echo "Seed $SEED done: success=$SUCCESS_RATE, tasks=$TASK_COUNT, peak_vram_gb=$PEAK_VRAM_GB, latency_ms=$AVG_STEP_LATENCY_MS, loop_hz=$E2E_LOOP_RATE_HZ, exit=$EXIT_CODE"
done

$default_python - "$SUMMARY_CSV" "$TASK_SUMMARY_CSV" "$TARGET_SUCCESS" <<'PY' > "$AGGREGATE_REPORT"
import csv
import math
import statistics
import sys

summary_csv = sys.argv[1]
task_csv = sys.argv[2]
target = float(sys.argv[3])

def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None

rows = []
success_vals = []
with open(summary_csv, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
        v = parse_float(r.get("success_rate", ""))
        if v is not None:
            success_vals.append(v)

print("Week 2 Visual Prompting: Aggregate Report")
print("=" * 44)
print(f"Runs logged: {len(rows)}")
print(f"Target success (%): {target:.2f}")

if success_vals:
    mean = statistics.fmean(success_vals)
    std = statistics.stdev(success_vals) if len(success_vals) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(success_vals)) if len(success_vals) > 1 else 0.0
    delta = mean - target
    print(f"Mean success (%): {mean:.4f}")
    print(f"Std success (%): {std:.4f}")
    print(f"95% CI half-width: {ci95:.4f}")
    print(f"Delta vs target (%): {delta:+.4f}")
else:
    print("Mean success (%): NA")
    print("Std success (%): NA")
    print("95% CI half-width: NA")
    print("Delta vs target (%): NA")

print("\nPer-seed runs:")
for r in rows:
    print(
        f"  seed={r.get('seed','NA')}, success={r.get('success_rate','NA')}, "
        f"peak_vram_gb={r.get('peak_vram_gb','NA')}, latency_ms={r.get('avg_step_latency_ms','NA')}, "
        f"loop_hz={r.get('e2e_loop_rate_hz','NA')}, wall_sec={r.get('wall_time_sec','NA')}, "
        f"eval_s={r.get('eval_s','NA')}, eval_ep_s={r.get('eval_ep_s','NA')}, exit={r.get('exit_code','NA')}"
    )

# Aggregate per-task success across seeds.
per_task = {}
with open(task_csv, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = (r.get("task_group", "unknown"), r.get("task_id", "NA"))
        v = parse_float(r.get("task_success_rate", ""))
        if v is None:
            continue
        per_task.setdefault(key, []).append(v)

print("\nPer-task success across seeds (%):")
if not per_task:
    print("  NA")
else:
    def sort_key(item):
        (group, task_id), _vals = item
        try:
            tid = int(task_id)
        except Exception:
            tid = 10**9
        return (str(group), tid, str(task_id))

    for (group, task_id), vals in sorted(per_task.items(), key=sort_key):
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(
            f"  {group}:{task_id} -> mean={mean:.4f}, std={std:.4f}, ci95_half_width={ci95:.4f}, seeds={len(vals)}"
        )
PY

echo ""
echo "Week-2 visual prompting 3-seed run complete."
echo "Artifacts root:        $OUTPUT_ROOT"
echo "Runs summary CSV:      $SUMMARY_CSV"
echo "Task summary CSV:      $TASK_SUMMARY_CSV"
echo "Aggregate report text: $AGGREGATE_REPORT"
