#!/usr/bin/env bash
# run_continued_training.sh
# Continue training lerobot/smolvla_libero on the LIBERO-10 demonstrations of lerobot/libero with lerobot-train,
# the recipe the checkpoint was trained with except for the episodes, the step budget and the learning-rate
# schedule (README.md). The command line below is the record of how reported numbers were produced.
#
# Usage (from anywhere):
#   STEPS=20000 OUT_DIR=/content/ct_run bash continued_training/run_continued_training.sh
#   STEPS=20000 OUT_DIR=/content/ct_run SYNC_DIR=/content/drive/MyDrive/ct bash continued_training/run_continued_training.sh
#   RESUME=1 OUT_DIR=/content/ct_run bash continued_training/run_continued_training.sh   # continue from checkpoints/last
#   SMOKE=1 bash continued_training/run_continued_training.sh                             # 4 steps, batch 2, 2 episodes
#
# With SYNC_DIR set, every new checkpoint (weights, processors and optimizer state) is mirrored to
# $SYNC_DIR/last/ while training runs, so a lost runtime loses at most SAVE_FREQ steps.
# RESUME=1 with no local checkpoint (new runtime) first restores $SYNC_DIR/last into $OUT_DIR.
# Output: $OUT_DIR/checkpoints/<step>/pretrained_model (what lerobot-eval loads); log $OUT_DIR.train.log
# (outside $OUT_DIR: lerobot-train refuses to start a fresh run in an existing folder).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/swagat/miniconda3/envs/lerobot_env/bin/python}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-$(dirname "$PYTHON_BIN")/lerobot-train}"

# Starting checkpoint: lerobot/smolvla_libero at the revision behind the 57.0% baseline
# (week6/smolvla_libero_lerobot_eval_kaggle/).
POLICY_REPO="${POLICY_REPO:-lerobot/smolvla_libero}"
POLICY_REVISION="${POLICY_REVISION:-a24f1f380461b665158d52acd140152c79e8b36d}"
# The dataset the checkpoint was trained on; episodes 0-378 are the 10 LIBERO-10 tasks (task_index 0-9) at
# this revision, checked by the notebook before training.
DATASET_REPO="${DATASET_REPO:-lerobot/libero}"
DATASET_REVISION="${DATASET_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}"
FIRST_EPISODE="${FIRST_EPISODE:-0}"
LAST_EPISODE="${LAST_EPISODE:-378}"

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"            # as the checkpoint
LR="${LR:-5e-5}"                          # checkpoint: 1e-4 from scratch of the expert; lower to continue
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DECAY_LR="${DECAY_LR:-2.5e-6}"            # as the checkpoint
SAVE_FREQ="${SAVE_FREQ:-5000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LOG_FREQ="${LOG_FREQ:-100}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/continued_training/run_seed${SEED}}"
SYNC_DIR="${SYNC_DIR:-}"
RESUME="${RESUME:-0}"

if [[ "${SMOKE:-0}" == 1 ]]; then
  STEPS=4 BATCH_SIZE=2 SAVE_FREQ=2 NUM_WORKERS=2 LOG_FREQ=1 WARMUP_STEPS=1 LAST_EPISODE=1
  OUT_DIR="${OUT_DIR%/}_smoke"
fi

RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

sync_last() {
  # Mirror checkpoints/last (a symlink to the newest step) to $SYNC_DIR/last; returns quietly without one.
  local last="$OUT_DIR/checkpoints/last"
  [[ -n "$SYNC_DIR" && -e "$last" ]] || return 0
  local step; step="$(basename "$(readlink -f "$last")")"
  [[ "$(cat "$SYNC_DIR/last_step" 2>/dev/null)" == "$step" ]] && return 0
  mkdir -p "$SYNC_DIR/last"
  rsync -a --delete "$last/" "$SYNC_DIR/last/"
  echo "$step" > "$SYNC_DIR/last_step"
  cp "$TRAIN_LOG" "$SYNC_DIR/train.log"
  echo "[sync] checkpoint $step -> $SYNC_DIR/last"
}

TRAIN_LOG="${OUT_DIR%/}.train.log"

if [[ "$RESUME" == 1 ]]; then
  if [[ ! -e "$OUT_DIR/checkpoints/last" && -n "$SYNC_DIR" && -f "$SYNC_DIR/last_step" ]]; then
    step="$(cat "$SYNC_DIR/last_step")"
    mkdir -p "$OUT_DIR/checkpoints/$step"
    rsync -a "$SYNC_DIR/last/" "$OUT_DIR/checkpoints/$step/"
    ln -sfn "$step" "$OUT_DIR/checkpoints/last"
    [[ -f "$SYNC_DIR/train.log" ]] && cp "$SYNC_DIR/train.log" "$TRAIN_LOG"   # keep one log across runtimes
    echo "restored checkpoint $step from $SYNC_DIR/last"
  fi
  CFG="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  [[ -f "$CFG" ]] || { echo "RESUME=1 but $CFG does not exist"; exit 1; }
  CMD=("$LEROBOT_TRAIN" "--config_path=$CFG" --resume=true)
else
  [[ -e "$OUT_DIR" ]] && { echo "$OUT_DIR exists; use RESUME=1 or a new OUT_DIR"; exit 1; }
  POLICY_PATH="$("$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$POLICY_REPO', revision='$POLICY_REVISION'))" | tail -n 1)"
  EPISODES="[$(seq -s, "$FIRST_EPISODE" "$LAST_EPISODE")]"
  CMD=(
    "$LEROBOT_TRAIN"
    "--policy.path=$POLICY_PATH"
    --policy.push_to_hub=false
    --policy.device=cuda
    "--policy.optimizer_lr=$LR"
    "--policy.scheduler_warmup_steps=$WARMUP_STEPS"
    "--policy.scheduler_decay_steps=$STEPS"
    "--policy.scheduler_decay_lr=$DECAY_LR"
    "--dataset.repo_id=$DATASET_REPO"
    "--dataset.revision=$DATASET_REVISION"
    "--dataset.episodes=$EPISODES"
    "--rename_map=$RENAME_MAP"
    "--steps=$STEPS"
    "--batch_size=$BATCH_SIZE"
    "--save_freq=$SAVE_FREQ"
    "--log_freq=$LOG_FREQ"
    "--num_workers=$NUM_WORKERS"
    "--seed=$SEED"
    "--output_dir=$OUT_DIR"
    --job_name=smolvla_libero10_continued
    --wandb.enable=false
  )
fi

mkdir -p "$(dirname "$TRAIN_LOG")"
echo "$(date -Is) ${CMD[*]}" | tee -a "$TRAIN_LOG"
echo "git $(git -C "$REPO_ROOT" rev-parse HEAD) dirty=$([[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]] && echo yes || echo no)" | tee -a "$TRAIN_LOG"
"${CMD[@]}" >> "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
  sync_last
done
wait "$TRAIN_PID" && RC=0 || RC=$?
sync_last
if [[ -n "$SYNC_DIR" ]]; then cp "$TRAIN_LOG" "$SYNC_DIR/train.log"; fi
echo "lerobot-train exit code $RC"
exit "$RC"
