#!/usr/bin/env bash
set -euo pipefail

# Week 1: Reproducible Baseline Lock (SmolVLA on LIBERO-10)
# What this script does:
# 1) Freezes and logs a fixed evaluation protocol.
# 2) Runs 3 seeded evaluations.
# 3) Captures hardware + profiling hooks (peak VRAM, latency/step, loop rate).
# 4) Produces per-seed logs and an aggregate summary.
#
# Usage:
#   bash week1_repro_baseline_lock.sh
#
# Optional overrides:
#   SEEDS="0 1 2" N_EPISODES=50 EPISODE_HORIZON=300 \
#   DATASET_ROOT=~/libero_dataset MODEL_ID=HuggingFaceVLA/smolvla_libero \
#   bash week1_repro_baseline_lock.sh

DATASET_ROOT="${DATASET_ROOT:-$HOME/libero_dataset}"
MODEL_ID="${MODEL_ID:-HuggingFaceVLA/smolvla_libero}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_SPLIT="${TASK_SPLIT:-LIBERO-10 official split}"
N_EPISODES="${N_EPISODES:-10}"
EPISODE_HORIZON="${EPISODE_HORIZON:-300}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OBSERVATION_HEIGHT="${OBSERVATION_HEIGHT:-256}"
OBSERVATION_WIDTH="${OBSERVATION_WIDTH:-256}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-10}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
TARGET_SUCCESS="${TARGET_SUCCESS:-46.0}"
USE_EVAL_SEED_FLAG="${USE_EVAL_SEED_FLAG:-1}"
SEED_FLAG_MODE="${SEED_FLAG_MODE:-auto}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="week1_baseline_lock_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/week1_baseline_lock/${RUN_NAME}}"
mkdir -p "$OUTPUT_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSOLIDATED_DIR="${CONSOLIDATED_DIR:-$SCRIPT_DIR}"
mkdir -p "$CONSOLIDATED_DIR"

# Headless rendering backend for MuJoCo.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export DATASET_ROOT

# LIBERO reads paths from $LIBERO_CONFIG_PATH/config.yaml.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
mkdir -p "$LIBERO_CONFIG_PATH"

command -v lerobot-eval >/dev/null 2>&1 || {
  echo "ERROR: lerobot-eval not found in PATH." >&2
  exit 1
}

python - <<'PY'
import os
from pathlib import Path
import yaml
import libero.libero as libero_pkg

cfg_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
cfg_path = cfg_dir / "config.yaml"

benchmark_root = Path(libero_pkg.__file__).resolve().parent
config = {
    "benchmark_root": str(benchmark_root),
    "bddl_files": str(benchmark_root / "bddl_files"),
    "init_states": str(benchmark_root / "init_files"),
    "datasets": str(Path(os.path.expanduser(os.environ.get("DATASET_ROOT", "~/libero_dataset"))).resolve()),
    "assets": str(benchmark_root / "assets"),
}

cfg_dir.mkdir(parents=True, exist_ok=True)
with cfg_path.open("w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(f"Wrote LIBERO config: {cfg_path}")
for k, v in config.items():
    print(f"  {k}: {v}")
PY

PROTOCOL_DIR="$OUTPUT_ROOT/protocol"
mkdir -p "$PROTOCOL_DIR"

# Snapshot fixed protocol + environment metadata.
cat > "$PROTOCOL_DIR/fixed_protocol.yaml" <<EOF
run_name: "$RUN_NAME"
timestamp: "$(date -Iseconds)"
project_stage: "Week 1 Reproducible Baseline Lock"
model_id: "$MODEL_ID"
task_suite: "$TASK_SUITE"
task_split: "$TASK_SPLIT"
episodes_per_seed: $N_EPISODES
episode_horizon: $EPISODE_HORIZON
camera_settings:
  observation_height: $OBSERVATION_HEIGHT
  observation_width: $OBSERVATION_WIDTH
  render_mode: "rgb_array"
evaluation_seeds: [$(echo "$SEEDS" | xargs | sed 's/ /, /g')]
policy_device: "$POLICY_DEVICE"
batch_size: $BATCH_SIZE
dataset_root: "$DATASET_ROOT"
libero_config_path: "$LIBERO_CONFIG_PATH"
mujoco_gl: "$MUJOCO_GL"
pyopengl_platform: "$PYOPENGL_PLATFORM"
EOF

{
  echo "===== uname -a ====="
  uname -a
  echo
  echo "===== lscpu ====="
  lscpu || true
  echo
  echo "===== memory ====="
  free -h || true
  echo
  echo "===== nvidia-smi ====="
  nvidia-smi || true
} > "$PROTOCOL_DIR/hardware_stats.txt" 2>&1

SUMMARY_CSV="$CONSOLIDATED_DIR/week1_run_summary.csv"
TASK_SUMMARY_CSV="$CONSOLIDATED_DIR/week1_task_success_summary.csv"
echo "seed,success_rate,peak_vram_gb,avg_step_latency_ms,e2e_loop_rate_hz,wall_time_sec,exit_code,run_dir,task_success_rates" > "$SUMMARY_CSV"
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

echo "Starting Week 1 baseline lock runs..."
echo "Output root: $OUTPUT_ROOT"
echo "Seed flag mode: $RESOLVED_SEED_FLAG_MODE"

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
    lerobot-eval
    "--policy.path=$MODEL_ID"
    "--policy.device=$POLICY_DEVICE"
    --env.type=libero
    "--env.task=$TASK_SUITE"
    --env.render_mode=rgb_array
    --env.max_parallel_tasks=1
    "--env.observation_height=$OBSERVATION_HEIGHT"
    "--env.observation_width=$OBSERVATION_WIDTH"
    "--env.episode_length=$EPISODE_HORIZON"
    "--eval.batch_size=$BATCH_SIZE"
    --eval.use_async_envs=false
    "--eval.n_episodes=$N_EPISODES"
    "--eval.max_episodes_rendered=$MAX_EPISODES_RENDERED"
    "--output_dir=$EVAL_OUT_DIR"
  )

  if [[ "$RESOLVED_SEED_FLAG_MODE" == "eval.seed" ]]; then
    CMD+=("--eval.seed=$SEED")
  elif [[ "$RESOLVED_SEED_FLAG_MODE" == "seed" ]]; then
    CMD+=("--seed=$SEED")
  fi

  set +e
  "${CMD[@]}" "${EXTRA_ARGS_ARRAY[@]}" 2>&1 | tee "$LOG_FILE"
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

  # Try to read success metric from JSON artifacts first; fallback to log regex.
  SUCCESS_RATE="$(python - "$EVAL_OUT_DIR" "$LOG_FILE" <<'PY'
import json
import pathlib
import re
import sys

eval_dir = pathlib.Path(sys.argv[1])
log_file = pathlib.Path(sys.argv[2])

keys = {
    "success_rate",
    "eval_success_rate",
    "eval/success_rate",
    "avg_success",
}

def find_in_obj(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, (int, float)):
                return float(v)
            out = find_in_obj(v)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for item in obj:
            out = find_in_obj(item)
            if out is not None:
                return out
    return None

for p in sorted(eval_dir.rglob("*.json")):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    val = find_in_obj(data)
    if val is not None:
        if 0.0 <= val <= 1.0:
            val *= 100.0
        print(f"{val:.4f}")
        sys.exit(0)

text = ""
try:
    text = log_file.read_text(errors="ignore")
except Exception:
    pass

# Prefer explicit final metrics; avoid transient progress metrics like running_success_rate.
patterns = [
    r"eval/success_rate\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    r"(?<!running_)success(?:\s*rate)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    r"(?<!running_)success(?:\s*rate)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
]
for pat in patterns:
    matches = re.findall(pat, text, flags=re.IGNORECASE)
    if matches:
        val = float(matches[-1])
        if 0.0 <= val <= 1.0:
            val *= 100.0
        print(f"{val:.4f}")
        sys.exit(0)

# Fallback: if only running_success_rate exists in logs, take the last seen value.
running_matches = re.findall(
    r"running_success_rate\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    text,
    flags=re.IGNORECASE,
)
if running_matches:
    val = float(running_matches[-1])
    if 0.0 <= val <= 1.0:
        val *= 100.0
    print(f"{val:.4f}")
    sys.exit(0)

print("NA")
PY
  )"

  EVAL_INFO_JSON="$EVAL_OUT_DIR/eval_info.json"
  TASK_PARSE_RESULT="$(python - "$EVAL_INFO_JSON" "$SEED" "$TASK_SUMMARY_CSV" <<'PY'
import json
import pathlib
import sys

info_path = pathlib.Path(sys.argv[1])
seed = sys.argv[2]
task_csv = pathlib.Path(sys.argv[3])

if not info_path.exists():
  print("NA|NA|0")
  sys.exit(0)

try:
  data = json.loads(info_path.read_text())
except Exception:
  print("NA|NA|0")
  sys.exit(0)

overall = data.get("overall", {})
pc_success = overall.get("pc_success", "NA")

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
    f.write(f"{seed},{group},{task_id},{rate_text}\n")

summary_parts = [f"{group}:{task_id}={rate_text}" for group, task_id, rate_text in triples]
summary = ";".join(summary_parts) if summary_parts else "NA"

if isinstance(pc_success, (int, float)):
  pc_text = f"{pc_success:.4f}"
else:
  pc_text = "NA"

print(f"{pc_text}|{summary}|{len(triples)}")
PY
  )"

SUCCESS_RATE_FROM_EVAL_INFO="NA"
TASK_SUCCESS_RATES="NA"
TASK_COUNT="0"
IFS='|' read -r SUCCESS_RATE_FROM_EVAL_INFO TASK_SUCCESS_RATES TASK_COUNT <<< "$TASK_PARSE_RESULT"

if [[ "$SUCCESS_RATE" == "NA" && "$SUCCESS_RATE_FROM_EVAL_INFO" != "NA" ]]; then
  SUCCESS_RATE="$SUCCESS_RATE_FROM_EVAL_INFO"
fi

  TOTAL_STEPS=$((N_EPISODES * EPISODE_HORIZON))
  E2E_LOOP_RATE_HZ="$(awk -v steps="$TOTAL_STEPS" -v t="$WALL_SEC" 'BEGIN{if (t>0) printf "%.6f", steps/t; else print "NA"}')"
  AVG_STEP_LATENCY_MS="$(awk -v hz="$E2E_LOOP_RATE_HZ" 'BEGIN{if (hz+0>0) printf "%.6f", 1000.0/hz; else print "NA"}')"

  # For failed/interrupted runs, avoid reporting derived throughput metrics from partial progress.
  if [[ "$EXIT_CODE" -ne 0 ]]; then
    AVG_STEP_LATENCY_MS="NA"
    E2E_LOOP_RATE_HZ="NA"
  fi

  # If evaluator logs explicit latency/loop metrics, prefer those.
  PARSED_LATENCY="$(grep -Eio 'avg[^\n]*latency[^0-9]*[0-9]+(\.[0-9]+)?\s*ms' "$LOG_FILE" | tail -n1 | grep -Eo '[0-9]+(\.[0-9]+)?' || true)"
  PARSED_LOOP_HZ="$(grep -Eio '(control loop|loop rate|throughput)[^0-9]*[0-9]+(\.[0-9]+)?\s*hz' "$LOG_FILE" | tail -n1 | grep -Eo '[0-9]+(\.[0-9]+)?' || true)"
  if [[ -n "$PARSED_LATENCY" ]]; then
    AVG_STEP_LATENCY_MS="$PARSED_LATENCY"
  fi
  if [[ -n "$PARSED_LOOP_HZ" ]]; then
    E2E_LOOP_RATE_HZ="$PARSED_LOOP_HZ"
  fi

  cat > "$RUN_DIR/seed_report.yaml" <<EOF
seed: $SEED
success_rate_percent: "$SUCCESS_RATE"
peak_vram_gb: "$PEAK_VRAM_GB"
avg_step_latency_ms: "$AVG_STEP_LATENCY_MS"
e2e_loop_rate_hz: "$E2E_LOOP_RATE_HZ"
wall_time_sec: "$WALL_SEC"
episodes: $N_EPISODES
episode_horizon: $EPISODE_HORIZON
total_nominal_steps: $TOTAL_STEPS
exit_code: $EXIT_CODE
run_dir: "$RUN_DIR"
EOF

  echo "$SEED,$SUCCESS_RATE,$PEAK_VRAM_GB,$AVG_STEP_LATENCY_MS,$E2E_LOOP_RATE_HZ,$WALL_SEC,$EXIT_CODE,$RUN_DIR,$TASK_SUCCESS_RATES" >> "$SUMMARY_CSV"

  echo "Seed $SEED done: success=$SUCCESS_RATE, tasks=$TASK_COUNT, peak_vram_gb=$PEAK_VRAM_GB, latency_ms=$AVG_STEP_LATENCY_MS, loop_hz=$E2E_LOOP_RATE_HZ, exit=$EXIT_CODE"
done

AGGREGATE_REPORT="$CONSOLIDATED_DIR/week1_agregate_report.txt"
python - "$SUMMARY_CSV" "$TASK_SUMMARY_CSV" "$TARGET_SUCCESS" <<'PY' > "$AGGREGATE_REPORT"
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

success_vals = []
rows = []
with open(summary_csv, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
    v = parse_float(r.get("success_rate", ""))
    if v is not None:
      success_vals.append(v)

print("Week 1 Baseline Lock: Aggregate Report")
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
        f"  seed={r['seed']}, success={r['success_rate']}, "
        f"peak_vram_gb={r['peak_vram_gb']}, latency_ms={r['avg_step_latency_ms']}, "
        f"loop_hz={r['e2e_loop_rate_hz']}, wall_sec={r['wall_time_sec']}, exit={r['exit_code']}"
    )

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
echo "Week 1 baseline lock complete."
echo "Artifacts: $OUTPUT_ROOT"
echo "Protocol:  $PROTOCOL_DIR/fixed_protocol.yaml"
echo "Summary:   $SUMMARY_CSV"
echo "Report:    $AGGREGATE_REPORT"
