#!/usr/bin/env bash
set -euo pipefail

# Ignore terminal job-control stop signals so long runs don't get suspended by
# transient TTY state changes in some terminal environments.
trap '' TTOU TTIN TSTP

# Train + eval SAC on one LIBERO task across 3 seeds, with resumable execution.
#
# Usage:
#   bash run_sac_3seed_task.sh <task_id> <train_episodes> <eval_episodes>
#
# Example:
#   bash run_sac_3seed_task.sh 0 285 20
#
# Resume behavior:
# - Re-running this script resumes unfinished seeds from their latest checkpoints.
# - Ctrl+C during training triggers trainer-side interrupt checkpointing.
# - Completed phases are tracked via marker files under outputs/sac_3seed/task_<id>/seed_<seed>/.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <task_id> <train_episodes> <eval_episodes>" >&2
  exit 1
fi

TASK_ID="$1"
TRAIN_EPISODES="$2"
EVAL_EPISODES="$3"

if ! [[ "$TASK_ID" =~ ^[0-9]+$ && "$TRAIN_EPISODES" =~ ^[0-9]+$ && "$EVAL_EPISODES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: all arguments must be integers." >&2
  exit 1
fi

if (( TRAIN_EPISODES <= 0 || EVAL_EPISODES <= 0 )); then
  echo "ERROR: train_episodes and eval_episodes must be > 0." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-0 1 2}"

# Optional passthrough args for training and eval commands.
# Example:
#   EXTRA_TRAIN_ARGS="--hard_task_preset --use_visual_prompting"
#   EXTRA_EVAL_ARGS="--hard_task_preset --use_visual_prompting"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/sac_3seed/task_${TASK_ID}}"
mkdir -p "$OUTPUT_ROOT"

acquire_runner_lock() {
  local lock_path="$OUTPUT_ROOT/.runner.lock"
  if ! command -v flock >/dev/null 2>&1; then
    echo "[runner] warning: flock not found; single-instance lock disabled."
    return
  fi

  # Keep lock file descriptor open for the lifetime of this process.
  exec {RUNNER_LOCK_FD}>"$lock_path"
  if ! flock -n "$RUNNER_LOCK_FD"; then
    echo "[runner] another runner instance is active for this task/output: $OUTPUT_ROOT" >&2
    echo "[runner] stop the existing run (or wait for it to finish) before starting a new one." >&2
    exit 1
  fi
}

# By default, temporarily disable GNOME battery idle suspend while this runner is active.
# Set GNOME_BATTERY_SLEEP_DISABLE=0 to opt out.
GNOME_BATTERY_SLEEP_DISABLE="${GNOME_BATTERY_SLEEP_DISABLE:-1}"
GNOME_BATTERY_SLEEP_SETTINGS_APPLIED=0
ORIG_GNOME_BATTERY_SLEEP_TYPE=""
ORIG_GNOME_BATTERY_SLEEP_TIMEOUT=""
ORIG_POWER_PROFILE=""
POWER_PROFILE_SETTINGS_APPLIED=0

apply_system_power_overrides() {
  if [[ "$GNOME_BATTERY_SLEEP_DISABLE" == "1" ]]; then
    if command -v gsettings >/dev/null 2>&1; then
      ORIG_GNOME_BATTERY_SLEEP_TYPE="$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 2>/dev/null || true)"
      ORIG_GNOME_BATTERY_SLEEP_TIMEOUT="$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 2>/dev/null || true)"

      if [[ -n "$ORIG_GNOME_BATTERY_SLEEP_TYPE" && -n "$ORIG_GNOME_BATTERY_SLEEP_TIMEOUT" ]]; then
        if gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null && \
           gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0 2>/dev/null; then
          GNOME_BATTERY_SLEEP_SETTINGS_APPLIED=1
          echo "[runner] GNOME battery idle suspend disabled for this run."
        else
          echo "[runner] failed to set GNOME battery suspend override; continuing without it."
        fi
      fi
    else
      echo "[runner] gsettings not found; skipping GNOME battery suspend override."
    fi
  fi

  if command -v powerprofilesctl >/dev/null 2>&1; then
    ORIG_POWER_PROFILE="$(powerprofilesctl get 2>/dev/null || true)"
    if [[ -n "$ORIG_POWER_PROFILE" ]]; then
      if powerprofilesctl set performance 2>/dev/null; then
        POWER_PROFILE_SETTINGS_APPLIED=1
        echo "[runner] Power profile set to 'performance' for this run."
      else
        echo "[runner] failed to set power profile to 'performance'; continuing without it."
      fi
    fi
  else
    echo "[runner] powerprofilesctl not found; skipping power profile management."
  fi
}

restore_system_power_overrides() {
  if (( GNOME_BATTERY_SLEEP_SETTINGS_APPLIED == 1 )); then
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type "$ORIG_GNOME_BATTERY_SLEEP_TYPE" 2>/dev/null || true
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout "$ORIG_GNOME_BATTERY_SLEEP_TIMEOUT" 2>/dev/null || true
    echo "[runner] GNOME battery idle suspend settings restored."
  fi

  if (( POWER_PROFILE_SETTINGS_APPLIED == 1 )); then
    powerprofilesctl set "$ORIG_POWER_PROFILE" 2>/dev/null || true
    echo "[runner] Power profile restored to '$ORIG_POWER_PROFILE'."
  fi
}

CONSOLIDATED_CSV="$OUTPUT_ROOT/consolidated_sac_vs_week1_task_${TASK_ID}.csv"
WEEK1_REFERENCE_CSV="$SCRIPT_DIR/week1_baseline_failures.csv"
CONSOLIDATED_HEADER="seed,task_id,train_episodes,eval_episodes,train_mean_success_rate,eval_success_rate,eval_mean_reward,eval_mean_steps,checkpoint_path,seed_output_dir,week1_reference_csv"

if [[ ! -f "$CONSOLIDATED_CSV" ]]; then
  echo "$CONSOLIDATED_HEADER" > "$CONSOLIDATED_CSV"
fi

migrate_consolidated_schema_if_needed() {
  local current_header
  current_header="$(head -n 1 "$CONSOLIDATED_CSV" 2>/dev/null || true)"

  if [[ "$current_header" == "$CONSOLIDATED_HEADER" ]]; then
    return
  fi

  local tmp_csv
  tmp_csv="$(mktemp)"

  awk -F, -v OFS=, -v hdr="$CONSOLIDATED_HEADER" '
    NR==1 {
      print hdr
      next
    }
    NF==10 {
      # Old format: insert train_mean_success_rate=NA after eval_episodes.
      print $1,$2,$3,$4,"NA",$5,$6,$7,$8,$9,$10
      next
    }
    NF>=11 {
      print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11
      next
    }
  ' "$CONSOLIDATED_CSV" > "$tmp_csv"

  mv "$tmp_csv" "$CONSOLIDATED_CSV"
}

migrate_consolidated_schema_if_needed

read -r -a EXTRA_TRAIN_ARRAY <<< "$EXTRA_TRAIN_ARGS"
read -r -a EXTRA_EVAL_ARRAY <<< "$EXTRA_EVAL_ARGS"

INTERRUPTED=0
INTERRUPT_COUNT=0

on_interrupt() {
  INTERRUPT_COUNT=$((INTERRUPT_COUNT + 1))
  INTERRUPTED=1
  echo
  if (( INTERRUPT_COUNT == 1 )); then
    echo "[runner] interrupt received; stopping current subprocess cleanly..."
    echo "[runner] press Ctrl+C again to force immediate exit."
  else
    echo "[runner] second interrupt received; forcing exit now."
    exit 130
  fi
}

trap on_interrupt INT TERM
trap restore_system_power_overrides EXIT

apply_system_power_overrides
acquire_runner_lock

run_and_tee() {
  local log_file="$1"
  shift

  set +e
  "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  return $rc
}

run_phase_or_exit() {
  local phase_name="$1"
  local seed="$2"
  local log_file="$3"
  shift 3

  run_and_tee "$log_file" "$@"
  local rc=$?
  if (( rc != 0 )); then
    echo "[runner] ${phase_name} exited with code $rc for seed $seed"
    if (( INTERRUPTED == 1 )) || [[ "$rc" -eq 130 ]]; then
      echo "[runner] interrupted. Re-run the same command to resume."
      exit 130
    fi
    exit "$rc"
  fi
}

upsert_summary_row() {
  local seed="$1"
  local train_mean_success_rate="$2"
  local success_rate="$3"
  local mean_reward="$4"
  local mean_steps="$5"
  local checkpoint_path="$6"
  local seed_dir="$7"

  local tmp_csv
  tmp_csv="$(mktemp)"

  awk -F, -v s="$seed" 'NR==1 || $1 != s' "$CONSOLIDATED_CSV" > "$tmp_csv"
  echo "$seed,$TASK_ID,$TRAIN_EPISODES,$EVAL_EPISODES,$train_mean_success_rate,$success_rate,$mean_reward,$mean_steps,$checkpoint_path,$seed_dir,$WEEK1_REFERENCE_CSV" >> "$tmp_csv"
  mv "$tmp_csv" "$CONSOLIDATED_CSV"
}

extract_train_mean_success_rate() {
  local train_log="$1"
  "$PYTHON_BIN" - "$train_log" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''

pattern = re.compile(r"\[progress\].*?success_mean=([0-9]*\.?[0-9]+)")
match = None
for m in pattern.finditer(text):
    match = m

if match is None:
    print("NA")
else:
    print(match.group(1))
PY
}

extract_eval_metrics() {
  local eval_log="$1"
  "$PYTHON_BIN" - "$eval_log" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''

pattern = re.compile(r"\[eval\]\s+success_rate=([0-9]*\.?[0-9]+).*?mean_reward=([-+]?[0-9]*\.?[0-9]+).*?mean_steps=([-+]?[0-9]*\.?[0-9]+)")
match = None
for m in pattern.finditer(text):
  match = m

if match is None:
  print("NA|NA|NA")
else:
  print(f"{match.group(1)}|{match.group(2)}|{match.group(3)}")
PY
}

echo "[runner] task_id=$TASK_ID train_episodes=$TRAIN_EPISODES eval_episodes=$EVAL_EPISODES"
echo "[runner] output_root=$OUTPUT_ROOT"
echo "[runner] seeds=$SEEDS"

for SEED in $SEEDS; do
  if (( INTERRUPTED == 1 )); then
    break
  fi

  SEED_DIR="$OUTPUT_ROOT/seed_${SEED}"
  CHECKPOINT_DIR="$SEED_DIR/checkpoints_task_${TASK_ID}"
  CHECKPOINT_PATH="$CHECKPOINT_DIR/latest_checkpoint.pt"
  TRAIN_LOG="$SEED_DIR/train.log"
  EVAL_LOG="$SEED_DIR/eval.log"
  TRAIN_DONE_MARKER="$SEED_DIR/train.done"
  EVAL_DONE_MARKER="$SEED_DIR/eval.done"

  mkdir -p "$SEED_DIR" "$CHECKPOINT_DIR"

  echo
  echo "===== Seed $SEED ====="

  if [[ ! -f "$TRAIN_DONE_MARKER" ]]; then
    echo "[runner] training seed $SEED (resumable)..."
    TRAIN_CMD=(
      "$PYTHON_BIN" "-u" "$SCRIPT_DIR/train_online_sac.py"
      --task_id "$TASK_ID"
      --seed "$SEED"
      --num_episodes "$TRAIN_EPISODES"
      --resume_training
      --checkpoint_dir "$CHECKPOINT_DIR"
      --checkpoint_path "$CHECKPOINT_PATH"
      --wandb_name "sac_task${TASK_ID}_seed${SEED}"
      --wandb_group "sac_task_${TASK_ID}"
    )

    TRAIN_CMD+=("${EXTRA_TRAIN_ARRAY[@]}")

    run_phase_or_exit "training" "$SEED" "$TRAIN_LOG" "${TRAIN_CMD[@]}"

    touch "$TRAIN_DONE_MARKER"
    echo "[runner] training complete for seed $SEED"
  else
    echo "[runner] training already complete for seed $SEED (marker found)."
  fi

  if [[ ! -f "$EVAL_DONE_MARKER" ]]; then
    echo "[runner] evaluating seed $SEED..."
    EVAL_CMD=(
      "$PYTHON_BIN" "-u" "$SCRIPT_DIR/train_online_sac.py"
      --task_id "$TASK_ID"
      --seed "$SEED"
      --eval
      --eval_episodes "$EVAL_EPISODES"
      --checkpoint_dir "$CHECKPOINT_DIR"
      --checkpoint_path "$CHECKPOINT_PATH"
    )

    EVAL_CMD+=("${EXTRA_EVAL_ARRAY[@]}")

    run_phase_or_exit "eval" "$SEED" "$EVAL_LOG" "${EVAL_CMD[@]}"

    touch "$EVAL_DONE_MARKER"
    echo "[runner] eval complete for seed $SEED"
  else
    echo "[runner] eval already complete for seed $SEED (marker found)."
  fi

    METRICS="$(extract_eval_metrics "$EVAL_LOG")"

  IFS='|' read -r SUCCESS_RATE MEAN_REWARD MEAN_STEPS <<< "$METRICS"
  TRAIN_MEAN_SUCCESS_RATE="$(extract_train_mean_success_rate "$TRAIN_LOG")"

  upsert_summary_row "$SEED" "$TRAIN_MEAN_SUCCESS_RATE" "$SUCCESS_RATE" "$MEAN_REWARD" "$MEAN_STEPS" "$CHECKPOINT_PATH" "$SEED_DIR"
  echo "[runner] consolidated: seed=$SEED train_mean_success_rate=$TRAIN_MEAN_SUCCESS_RATE success_rate=$SUCCESS_RATE mean_reward=$MEAN_REWARD mean_steps=$MEAN_STEPS"
done

if (( INTERRUPTED == 1 )); then
  echo "[runner] interrupted; progress saved. Resume by re-running the same command."
  exit 130
fi

echo
echo "[runner] all requested seeds processed."
echo "[runner] consolidated output: $CONSOLIDATED_CSV"
echo "[runner] week1 reference: $WEEK1_REFERENCE_CSV"
