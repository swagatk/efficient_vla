# create_new_script

Build bash scripts for long-running training/evaluation jobs with operational safety and resume support.

## Goal

Whenever generating a new execution script (training/eval/sweep), always include:

1. Continuous progress output on CLI at fixed intervals.
2. Optional GPU/laptop power hardening during execution.
3. Guaranteed restoration of system power settings on interrupt or normal exit.
4. WandB logging support.
5. WandB run id persistence per seed/repeat so resumed runs continue in the same WandB run.
6. Ctrl+C-safe interruption and resume from last checkpoint/marker.

## When To Apply

Apply this skill when creating or rewriting scripts that run:

- Training loops.
- Evaluation loops.
- Hyperparameter sweeps.
- Multi-seed or multi-repeat experiments.

Do not skip these requirements even for single-run scripts.

## Required Script Features

### 1) Progress heartbeat

- Add `HEARTBEAT_SEC` (default `60`).
- Print progress lines to stdout and append to `progress.log` under output directory.
- Print an immediate progress line when a unit starts, then continue at `HEARTBEAT_SEC` intervals.
- Progress line should include:
  - Completed jobs / total jobs.
  - Percent complete.
  - Current seed/repeat/alpha/task as relevant.
  - Elapsed time and ETA.

### 2) Power profile hardening (for GPU workloads)

- Before starting work:
  - Set power profile to `performance` (if available).
  - Disable sleep/suspend using platform-aware logic.
- On exit (success, failure, or interrupt):
  - Restore previous power profile.
  - Restore previous sleep/suspend setting.

Platform support guidance:

- Linux/GNOME: use `powerprofilesctl` and `gsettings`.
- macOS: detect via `uname -s` and use `pmset -g` to read current values, `pmset -a sleep 0` to disable sleep, then restore prior value on exit.
- Other platforms: skip power hardening silently.

Implementation must be best-effort and non-fatal when tools are unavailable.

### 3) Interrupt-safe behavior

- Trap `INT` and `TERM` with different semantics.
- First Ctrl+C:
  - Mark interrupted.
  - Stop after the current child process (seed/repeat) completes.
  - Save progress markers.
- Second Ctrl+C:
  - Exit immediately with code `130`.
- On `TERM`:
  - Perform immediate cleanup and exit without two-step Ctrl+C behavior.
- Note: step-level interruption inside a running trainer requires application-side signal handling beyond this script.

### 4) Resume support

- Persist completion markers per unit of work (seed/repeat/alpha).
- Skip completed units on rerun when `RESUME=1`.
- When `RESUME=0`, clear existing `DONE_MARKER`, `RESULT_FILE`, and `RUN_ID_FILE` for each unit before starting that unit.
- Implement `aggregate_results()` and call it before final exit: concatenate per-unit `result_row.csv` files into `summary.csv`, deduplicating by unit id.

### 5) WandB logging and run continuity

- Always pass WandB project/name/group tags for traceability.
- Persist run id per unit of work (seed/repeat) in `wandb_run_id.txt`.
- The run id in `wandb_run_id.txt` must be the id returned by `wandb.init()` on first run.
- Child Python must write that id to the path in `WANDB_RUN_ID_FILE` after initialization.
- On resume, reuse that exact run id by passing it back to the child script.
- Launch child command with WandB resume policy (`allow` by default).

### 6) Checkpoint-aware resume

- For training scripts, resume from last checkpoint when present.
- For eval/sweep scripts, resume from per-unit completion markers and keep WandB run continuity.

## Standard Environment Variables

Scripts should expose these variables by default (when applicable):

- `OUTPUT_ROOT`
- `RESUME` (default `1`)
- `HEARTBEAT_SEC` (default `60`)
- `WANDB_PROJECT`
- `WANDB_RESUME_POLICY` (default `allow`)
- `PYTHON_BIN` (default `python`)
- `USE_POWER_HARDENING` (default `1`)

## Acceptance Checklist (Mandatory)

Before finalizing a generated script, verify all are true:

- [ ] `bash -n <script>` passes. *(see §3 Interrupt-safe behavior)*
- [ ] Progress heartbeat prints while job is running, including an immediate start line. *(see §1 Progress heartbeat)*
- [ ] `progress.log` is created under output directory. *(see §1 Progress heartbeat)*
- [ ] Ctrl+C once performs graceful stop and preserves markers. *(see §3 Interrupt-safe behavior)*
- [ ] Ctrl+C twice exits immediately. *(see §3 Interrupt-safe behavior)*
- [ ] Resume rerun skips completed units when `RESUME=1`, and `RESUME=0` forces clean unit rerun. *(see §4 Resume support)*
- [ ] WandB run id file exists per seed/repeat and is reused on resume from child-reported run id. *(see §5 WandB logging and run continuity)*
- [ ] Power profile/sleep settings are restored on all exits. *(see §2 Power profile hardening)*

## Recommended Bash Skeleton

Use this pattern and adapt to script-specific loops.

```bash
#!/usr/bin/env bash
set -euo pipefail

RESUME="${RESUME:-1}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
WANDB_PROJECT="${WANDB_PROJECT:-my_project}"
WANDB_RESUME_POLICY="${WANDB_RESUME_POLICY:-allow}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_POWER_HARDENING="${USE_POWER_HARDENING:-1}"

INTERRUPTED=0
START_TS=$(date +%s)
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT"
PROGRESS_LOG="$OUTPUT_ROOT/progress.log"

if [[ "$RESUME" != "1" && -d "$OUTPUT_ROOT" ]]; then
  echo "[warn] OUTPUT_ROOT exists and RESUME=0; clearing stale markers" | tee -a "$PROGRESS_LOG"
  find "$OUTPUT_ROOT" -name .completed -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name result_row.csv -delete 2>/dev/null || true
  find "$OUTPUT_ROOT" -name wandb_run_id.txt -delete 2>/dev/null || true
fi

ORIG_POWER_PROFILE=""
ORIG_SLEEP_MODE=""
OS_NAME="$(uname -s)"

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
    writer = csv.DictWriter(f, fieldnames=["unit", "exit_code"])
    writer.writeheader()
    for key in sorted(dedup.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9):
        writer.writerow(dedup[key])
PY
}

# Example unit loop: seed/repeat/alpha
TOTAL_UNITS=10
DONE_UNITS=0

for UNIT in $(seq 1 "$TOTAL_UNITS"); do
  [[ "$INTERRUPTED" -eq 0 ]] || break

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
    env PYTHONUNBUFFERED=1 WANDB_RUN_ID_FILE="$RUN_ID_FILE" "$PYTHON_BIN" -u train_or_eval.py
    --wandb_project "$WANDB_PROJECT"
    --wandb_resume "$WANDB_RESUME_POLICY"
  )
  if [[ -n "$RUN_ID" ]]; then
    CMD+=(--wandb_run_id "$RUN_ID")
  fi

  print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} starting"
  print_progress "$DONE_UNITS" "$TOTAL_UNITS" "unit=${UNIT} heartbeat-start"

  "${CMD[@]}" > >(
    tee "$UNIT_DIR/run.log"
  ) 2>&1 &
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

  if [[ -f "$RUN_ID_FILE" ]]; then
    :
  else
    echo "child process must write wandb id to \$WANDB_RUN_ID_FILE" | tee -a "$PROGRESS_LOG"
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

[[ "$INTERRUPTED" -eq 0 ]] || exit 130
