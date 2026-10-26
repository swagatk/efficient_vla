#!/usr/bin/env bash
# run_step2_noise_selection.sh
# Step 2 of noise_reachability/README.md: the base policy and noise-selection policies on each task, paired
# episode by episode (same environment and torch seed per (seed, init state)). The command lines below are
# the record of how reported numbers were produced.
#
# Usage (from anywhere):
#   STAGE=select bash noise_reachability/run_step2_noise_selection.sh   # choose score and N: init states 10-19
#   STAGE=report bash noise_reachability/run_step2_noise_selection.sh   # report: init states 0-9 x seeds 0-2
#   SMOKE=1 bash noise_reachability/run_step2_noise_selection.sh        # task 0, seed 0, init state 0
# Override TASKS, SEEDS, INIT_STATES, POLICIES or OUT_ROOT to change a stage.
#
# Safe to re-run: finished episodes are skipped. Outputs: $OUT_ROOT/task_<t>/per_episode_<policy>.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
# The repo root must be importable at start-up so sitecustomize loads robosuite_mujoco_patch (FAIL-003).
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" PYTHONFAULTHANDLER=1

STAGE="${STAGE:-select}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  STAGE=smoke; TASKS="${TASKS:-0}"; SEEDS="${SEEDS:-0}"; INIT_STATES="${INIT_STATES:-0}"
  POLICIES="${POLICIES:-base,medoid_n8}"
elif [[ "$STAGE" == "select" ]]; then
  INIT_STATES="${INIT_STATES:-10-19}"; SEEDS="${SEEDS:-0,1}"
  POLICIES="${POLICIES:-base,first_n8,medoid_n8,accel_n8,medoid_n16,accel_n16}"
elif [[ "$STAGE" == "report" ]]; then
  INIT_STATES="${INIT_STATES:-0-9}"; SEEDS="${SEEDS:-0,1,2}"
  POLICIES="${POLICIES:?set POLICIES=base,<chosen policy> from the select stage}"
else
  echo "STAGE must be select or report" >&2; exit 2
fi
TASKS="${TASKS:-0 4 6 7 8}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/noise_selection/$STAGE}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
cd "$REPO_ROOT"

for task in $TASKS; do
  out="$OUT_ROOT/task_$task"
  mkdir -p "$out"
  # At most 10 episodes per process (FAIL-004); exit code 3 = call again.
  while true; do
    set +e
    "$PYTHON_BIN" -u "$SCRIPT_DIR/eval_noise_selection.py" --task_id "$task" --policies "$POLICIES" \
      --seeds "$SEEDS" --init_states "$INIT_STATES" --output_dir "$out" "${EXTRA[@]}" 2>&1 | tee -a "$out/eval.log"
    code=${PIPESTATUS[0]}
    set -e
    if [[ $code -eq 0 ]]; then break; fi
    if [[ $code -ne 3 ]]; then echo "[runner] task $task failed with exit code $code" >&2; exit "$code"; fi
    sleep 2  # stagger EGL process start-up (FAIL-006)
  done
done
