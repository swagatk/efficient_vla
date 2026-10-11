#!/usr/bin/env bash
# run_task3_residual.sh
# Residual hybrid-diffusion experiment on one LIBERO-10 task (default: task 3), end to end:
#   1. cache frozen-SmolVLA features + base action chunks for the task's demos (residual_cache.py build)
#   2. train the residual diffusion head on the cache                           (residual_cache.py train)
#   3. evaluate base and residual policies on the same init states and seeds    (train_hybrid_diffusion.py --eval_only)
#
# Usage:
#   bash run_task3_residual.sh                 # all stages
#   STAGES="eval_base" bash run_task3_residual.sh
#   STAGES="eval_residual" ALPHA=0.5 bash run_task3_residual.sh
#   SEED=1 bash run_task3_residual.sh          # independent repeat: head training seed and eval noise seed
#
# Every stage is resumable: finished caches, checkpoints and per-seed episode records are reused.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONFAULTHANDLER=1
# REPO_ROOT must be importable so robosuite_mujoco_patch loads (FAIL-003).
export PYTHONPATH="$REPO_ROOT:$HOME/LIBERO:$SCRIPT_DIR:${PYTHONPATH:-}"

TASK_ID="${TASK_ID:-3}"
BASE_POLICY_PATH="${BASE_POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
DATA_DIR="${DATA_DIR:-$HOME/lerobot_datasets/libero_10}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/outputs/task${TASK_ID}_residual}"
STAGES="${STAGES:-cache train eval_base eval_residual}"

NUM_BASE_SAMPLES="${NUM_BASE_SAMPLES:-4}"
EPOCHS="${EPOCHS:-300}"
# Head size and feature-noise level were picked on 5 held-out demos (val step-0 action MSE).
DIFF_HIDDEN_DIM="${DIFF_HIDDEN_DIM:-512}"
DIFF_LAYERS="${DIFF_LAYERS:-6}"
COND_NOISE_STD="${COND_NOISE_STD:-0.5}"
# SEED sets the head's training seed and the eval sampling-noise seed (--eval_noise_seed) for both
# policies; the init states are the same 50 for every seed. SEED=0 keeps the original directory names.
SEED="${SEED:-0}"
SEED_TAG=""
[[ "$SEED" != "0" ]] && SEED_TAG="_s${SEED}"
RUN_NAME="${RUN_NAME:-h${DIFF_HIDDEN_DIM}_l${DIFF_LAYERS}_n${COND_NOISE_STD}_e${EPOCHS}${SEED_TAG}}"
CHECKPOINT="${CHECKPOINT:-$OUT_ROOT/$RUN_NAME/latest_checkpoint.pt}"

EPISODES="${EPISODES:-50}"          # LIBERO ships 50 init states per task
EVAL_BATCH="${EVAL_BATCH:-3}"       # episodes per process (FAIL-004)
ALPHA="${ALPHA:-1.0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
# Rollout batches write eval_results.json; one W&B run per 3-episode batch is noise.
EVAL_WANDB_MODE="${EVAL_WANDB_MODE:-disabled}"

CACHE="$OUT_ROOT/feature_cache.pt"
mkdir -p "$OUT_ROOT"

run_eval() {
  local mode="$1" out_dir="$2"
  shift 2
  mkdir -p "$out_dir"
  local offset=0
  while (( offset < EPISODES )); do
    local n=$(( EPISODES - offset < EVAL_BATCH ? EPISODES - offset : EVAL_BATCH ))
    local attempt
    for attempt in 1 2 3; do
      # -S skips sitecustomize: this helper only reads JSON and must not import torch/MuJoCo.
      # A crash here must not abort the run (FAIL-008), hence the fallback to 0.
      local done_eps=0
      if [[ -f "$out_dir/eval_results.json" ]]; then
        done_eps=$("$PYTHON_BIN" -S -c "import json,sys; d=json.load(open(sys.argv[1])); s={e['seed'] for e in d['episodes']}; print(sum(1 for i in range($offset,$offset+$n) if i in s))" "$out_dir/eval_results.json" 2>/dev/null) || done_eps=0
      fi
      (( done_eps >= n )) && break
      echo "[eval:$mode] seeds $offset..$(( offset + n - 1 )) (attempt $attempt)"
      WANDB_MODE="$EVAL_WANDB_MODE" "$PYTHON_BIN" -u "$SCRIPT_DIR/train_hybrid_diffusion.py" \
        --base_policy_path "$BASE_POLICY_PATH" --eval_only --task_id "$TASK_ID" \
        --eval_policy_mode "$mode" --eval_episodes "$n" --eval_seed_offset "$offset" \
        --eval_diffusion_steps "$DIFFUSION_STEPS" --eval_replan_each_step \
        --image_flip_mode vertical_horizontal --eval_warmup_steps 10 --eval_noise_seed "$SEED" \
        --eval_output_dir "$out_dir" "$@" >> "$out_dir/eval.log" 2>&1 \
        || echo "[eval:$mode] batch at offset $offset exited non-zero; see $out_dir/eval.log"
      sleep 2
    done
    offset=$(( offset + n ))
  done
  cat "$out_dir/eval_summary.csv"
}

for stage in $STAGES; do
  case "$stage" in
    cache)
      if [[ ! -f "$CACHE" ]]; then
        "$PYTHON_BIN" -u "$SCRIPT_DIR/residual_cache.py" build --task_id "$TASK_ID" --data_dir "$DATA_DIR" \
          --base_policy_path "$BASE_POLICY_PATH" --num_base_samples "$NUM_BASE_SAMPLES" --out "$CACHE" \
          2>&1 | tee "$OUT_ROOT/build_cache.log"
      fi
      ;;
    train)
      # train_history.json is written last, so it marks a finished run. Delete the run dir to retrain.
      if [[ -f "$OUT_ROOT/$RUN_NAME/train_history.json" ]]; then
        echo "[train] $OUT_ROOT/$RUN_NAME already trained; skipping"
      else
        "$PYTHON_BIN" -u "$SCRIPT_DIR/residual_cache.py" train --cache "$CACHE" --out_dir "$OUT_ROOT/$RUN_NAME" \
          --epochs "$EPOCHS" --diff_hidden_dim "$DIFF_HIDDEN_DIM" --diff_layers "$DIFF_LAYERS" --cond_noise_std "$COND_NOISE_STD" \
          --seed "$SEED" \
          2>&1 | tee "$OUT_ROOT/$RUN_NAME.train.log"
      fi
      ;;
    eval_base)
      run_eval base "$OUT_ROOT/eval_base${SEED_TAG}"
      ;;
    eval_residual)
      run_eval residual "$OUT_ROOT/eval_residual_${RUN_NAME}_alpha${ALPHA}" \
        --eval_checkpoint_path "$CHECKPOINT" --eval_residual_alpha "$ALPHA"
      ;;
    *)
      echo "Unknown stage: $stage" >&2
      exit 1
      ;;
  esac
done
