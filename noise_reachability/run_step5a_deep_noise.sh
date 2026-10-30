#!/usr/bin/env bash
# run_step5a_deep_noise.sh
# Step 5a of noise_reachability/README.md. The command lines below are the record of how reported numbers were
# produced. Finished episodes are skipped, so a stage can be re-run after an interruption.
#
#   STAGE=pilot STREAMS="2 4 6" OUT_ROOT=... bash noise_reachability/run_step5a_deep_noise.sh
#   STAGE=harness|control|search|confirm STREAMS=4 OUT_ROOT=... bash noise_reachability/run_step5a_deep_noise.sh
#   STAGE=control STREAMS=1 UNITS_LIMIT=2 CONTROL_IDS=0 ...          # smoke run
# POLICY_PATH: the snapshot at revision a24f1f3804 (the notebook passes it). harness and control end with their gate:
# a failure stops the run before more is spent.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
# The repo root must be importable at start-up so sitecustomize loads robosuite_mujoco_patch (FAIL-003).
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" PYTHONFAULTHANDLER=1
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
STAGE="${STAGE:?set STAGE}"
STREAMS="${STREAMS:-2}"
NOISE_IDS="${NOISE_IDS:-0-69}"
CONTROL_IDS="${CONTROL_IDS:-0-34}"
UNITS_LIMIT="${UNITS_LIMIT:-0}"
export POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_libero}"
cd "$REPO_ROOT"

unit() {  # output_dir log args...: call the entry point until it reports nothing left
  local out="$1" log="$2"; shift 2
  mkdir -p "$out"
  local code=3
  while [[ $code == 3 ]]; do
    code=0
    "$PYTHON_BIN" -u noise_reachability/deep_noise_search.py --output_dir "$out" --policy_path "$POLICY_PATH" "$@" >> "$log" 2>&1 || code=$?
    if [[ $code != 0 && $code != 3 ]]; then echo "unit failed ($code): $*" >&2; return "$code"; fi
  done
}
export -f unit
export PYTHON_BIN

run_units() {  # reads lines "out log args..." on stdin
  xargs -r -P "$1" -L 1 bash -c 'unit "$@"' _
}

case "$STAGE" in
  pilot)
    # Worst-case episodes (state 2 fails, so runs all 520 steps), ids apart from every result. Wall time counts
    # the model loads, one per process, as the run does.
    for n in $STREAMS; do
      d="$OUT_ROOT/pilot/streams_$n"
      mkdir -p "$d"
      t0=$(date +%s)
      for k in $(seq 1 "$n"); do
        ids="$((9000 + 100 * n + 10 * k))-$((9000 + 100 * n + 10 * k + ${PILOT_EPISODES:-5} - 1))"
        echo "$d/s$k $d/s$k.log --family sampled --init_state 2 --noise_ids $ids"
      done | run_units "$n"
      echo "$(( $(date +%s) - t0 ))" > "$d/wall_seconds.txt"
    done ;;
  harness)
    mkdir -p "$OUT_ROOT/harness"
    for s in 0 1 2; do
      echo "$s"
    done | xargs -P "$STREAMS" -I{} bash -c \
      'code=3; while [[ $code == 3 ]]; do code=0; "$PYTHON_BIN" -u noise_reachability/eval_noise_selection.py --task_id 0 --policies base --policy_path "$POLICY_PATH" --seeds {} --init_states 0-9 --output_dir "'"$OUT_ROOT"'/harness/seed{}" >> "'"$OUT_ROOT"'/harness/seed{}.log" 2>&1 || code=$?; done; exit $code'
    "$PYTHON_BIN" -I -S - "$OUT_ROOT/harness" <<'PY'
import csv, glob, sys
root = sys.argv[1]
rows = [r for f in sorted(glob.glob(f"{root}/seed*/per_episode_base.csv")) for r in csv.DictReader(open(f))]
with open(f"{root}/per_episode_base.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
PY
    "$PYTHON_BIN" -I -S noise_reachability/analyze_step5a.py --root "$OUT_ROOT" --gate harness ;;
  control)
    mkdir -p "$OUT_ROOT/logs"
    for f in sampled full repeat; do
      echo "$OUT_ROOT $OUT_ROOT/logs/${f}_state9.log --family $f --init_state 9 --noise_ids $CONTROL_IDS"
    done | { if [[ "$UNITS_LIMIT" != 0 ]]; then head -n "$UNITS_LIMIT"; else cat; fi; } | run_units "$STREAMS"
    if [[ "$UNITS_LIMIT" == 0 ]]; then
      "$PYTHON_BIN" -I -S noise_reachability/analyze_step5a.py --root "$OUT_ROOT" --gate control
    fi ;;
  search)
    for s in 2 5 6 7 8; do for f in sampled full repeat; do
      echo "$OUT_ROOT $OUT_ROOT/logs/${f}_state$s.log --family $f --init_state $s --noise_ids $NOISE_IDS"
    done; done | run_units "$STREAMS" ;;
  confirm)
    "$PYTHON_BIN" -I -S noise_reachability/analyze_step5a.py --root "$OUT_ROOT" --hits | while read -r f s k; do
      echo "$OUT_ROOT $OUT_ROOT/logs/confirm_${f}_n${k}_state$s.log --family $f --init_state $s --noise_ids $k --jitter_seeds 5000-5003"
    done | run_units "$STREAMS" ;;
  *) echo "STAGE must be pilot, harness, control, search or confirm" >&2; exit 2 ;;
esac
