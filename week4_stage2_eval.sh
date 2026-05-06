#!/usr/bin/env bash
set -euo pipefail

# Week 4: Stage-2 Evaluation
#
# What this script does:
# 1) Runs 3 seeded evaluations using the optimal configuration from Week 3
#    (grounding_freq=1, box_style="mask").
# 2) Logs per-seed overall success and profiling hooks.
# 3) Logs per-task success rates for each seed.
# 4) Produces aggregate Week-4 summary reports.
#
# Usage:
#   bash week4_stage2_eval.sh

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
TARGET_SUCCESS="${TARGET_SUCCESS:-48.67}" # Set to Week 3 best mean success rate

# Optimal parameters found in Week 3
GROUNDING_FREQ="${GROUNDING_FREQ:-1}"
BOX_STYLE="${BOX_STYLE:-mask}"

# If 1, pass a seed flag through to lerobot-eval via eval_week2_visual_prompting.py.
USE_EVAL_SEED_FLAG="${USE_EVAL_SEED_FLAG:-1}"
SEED_FLAG_MODE="${SEED_FLAG_MODE:-auto}" # auto|eval.seed|seed|none

# Visual prompting toggles (GroundingDINO wrapper uses these)
BOX_OVERLAY="${BOX_OVERLAY:-1}"
TEXT_HINT="${TEXT_HINT:-1}"

# Runtime mode for eval entrypoint.
INPROCESS="${INPROCESS:-1}"

EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="week4_stage2_eval_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/lerobot/outputs/week4_stage2_eval/${RUN_NAME}}"
mkdir -p "$OUTPUT_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSOLIDATED_DIR="${CONSOLIDATED_DIR:-$SCRIPT_DIR}"
mkdir -p "$CONSOLIDATED_DIR"

# Leveraging the Week 2 python eval script as the core logic is identical
WEEK4_ENTRYPOINT="$SCRIPT_DIR/eval_week2_visual_prompting.py"
if [[ ! -f "$WEEK4_ENTRYPOINT" ]]; then
  echo "ERROR: missing $WEEK4_ENTRYPOINT" >&2
  exit 1
fi

restore_power_settings() {
  set +e
  powerprofilesctl set balanced 2>/dev/null || true
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'suspend' 2>/dev/null || true
}

trap restore_power_settings EXIT

powerprofilesctl set performance 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"

SUMMARY_CSV="$CONSOLIDATED_DIR/week4_runs_summary.csv"
TASK_SUMMARY_CSV="$CONSOLIDATED_DIR/week4_task_success_summary.csv"
AGGREGATE_REPORT="$CONSOLIDATED_DIR/week4_aggregate_report.txt"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "seed,grounding_freq,box_style,success_rate,peak_vram_gb,avg_step_latency_ms,e2e_loop_rate_hz,wall_time_sec,exit_code,run_dir,eval_s,eval_ep_s,task_success_rates" > "$SUMMARY_CSV"
fi
if [[ ! -f "$TASK_SUMMARY_CSV" ]]; then
  echo "seed,grounding_freq,box_style,task_group,task_id,task_success_rate" > "$TASK_SUMMARY_CSV"
fi

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

echo "Starting Week 4 Stage 2 Evaluation..."
echo "Output root: $OUTPUT_ROOT"
echo "Seed flag mode: $RESOLVED_SEED_FLAG_MODE"
echo "Config: Freq=$GROUNDING_FREQ | Style=$BOX_STYLE"

default_python="python3"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  default_python="$PYTHON_BIN"
fi

for SEED in $SEEDS; do
  RUN_DIR="$OUTPUT_ROOT/freq_${GROUNDING_FREQ}_style_${BOX_STYLE}/seed_${SEED}"
  EVAL_OUT_DIR="$RUN_DIR/eval_output"

  if [[ -f "$RUN_DIR/.completed" ]]; then
    echo ""
    echo "===== Seed $SEED already completed successfully. Skipping. ====="
    continue
  fi

  if [[ -d "$EVAL_OUT_DIR" ]]; then
    echo "Cleaning up partial outputs from previous interrupted run for Seed $SEED..."
    rm -rf "$EVAL_OUT_DIR"
  fi

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
    "$default_python" "$WEEK4_ENTRYPOINT"
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
    "--grounding_frequency=$GROUNDING_FREQ"
    "--box_style=$BOX_STYLE"
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
  PARSED_FIELDS="$($default_python - "$EVAL_INFO_JSON" "$SEED" "$TASK_SUMMARY_CSV" "$GROUNDING_FREQ" "$BOX_STYLE" <<'PY'
import json
import pathlib
import sys

info_path = pathlib.Path(sys.argv[1])
seed = sys.argv[2]
task_csv = pathlib.Path(sys.argv[3])
freq = sys.argv[4]
style = sys.argv[5]

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
        f.write(f"{seed},{freq},{style},{group},{task_id},{rate_text}\n")

summary_parts = [f"{group}:{task_id}={rate_text}" for group, task_id, rate_text in triples]
summary = ";".join(summary_parts) if summary_parts else "NA"

pc_text = f"{pc_success:.4f}" if isinstance(pc_success, (int, float)) else "NA"
eval_s_text = f"{eval_s:.6f}" if isinstance(eval_s, (int, float)) else "NA"
eval_ep_s_text = f"{eval_ep_s:.6f}" if isinstance(eval_ep_s, (int, float)) else "NA"

print(f"{pc_text}|{eval_s_text}|{eval_ep_s_text}|{summary}|{len(triples)}")
PY
)"

  SUCCESS_RATE="NA"
  EVAL_S="NA"
  EVAL_EP_S="NA"
  TASK_SUCCESS_RATES="NA"
  TASK_COUNT="0"

  IFS='|' read -r SUCCESS_RATE EVAL_S EVAL_EP_S TASK_SUCCESS_RATES TASK_COUNT <<< "$PARSED_FIELDS"

  echo "$SEED,$GROUNDING_FREQ,$BOX_STYLE,$SUCCESS_RATE,$PEAK_VRAM_GB,$AVG_STEP_LATENCY_MS,$E2E_LOOP_RATE_HZ,$WALL_SEC,$EXIT_CODE,$RUN_DIR,$EVAL_S,$EVAL_EP_S,$TASK_SUCCESS_RATES" >> "$SUMMARY_CSV"

  echo "Seed $SEED done: success=$SUCCESS_RATE, tasks=$TASK_COUNT, peak_vram_gb=$PEAK_VRAM_GB, latency_ms=$AVG_STEP_LATENCY_MS, loop_hz=$E2E_LOOP_RATE_HZ, exit=$EXIT_CODE"

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    touch "$RUN_DIR/.completed"
  fi
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

print("Week 4 Stage-2 Evaluation: Aggregate Report")
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
        f"  seed={r.get('seed','NA')}, freq={r.get('grounding_freq','NA')}, style={r.get('box_style','NA')}, "
        f"success={r.get('success_rate','NA')}, peak_vram_gb={r.get('peak_vram_gb','NA')}, "
        f"latency_ms={r.get('avg_step_latency_ms','NA')}, loop_hz={r.get('e2e_loop_rate_hz','NA')}, "
        f"exit={r.get('exit_code','NA')}"
    )
PY

echo ""
echo "Week 4 Stage 2 evaluation complete."
echo "Artifacts root:        $OUTPUT_ROOT"
echo "Runs summary CSV:      $SUMMARY_CSV"
echo "Task summary CSV:      $TASK_SUMMARY_CSV"
echo "Aggregate report text: $AGGREGATE_REPORT"