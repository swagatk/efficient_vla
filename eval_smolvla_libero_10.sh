#!/usr/bin/env bash
set -euo pipefail

# Evaluate SmolVLA on LIBERO-10 with LeRobot's built-in evaluator.
# Usage:
#   bash eval_smolvla_libero_10.sh
# Optional overrides:
#   DATASET_ROOT=~/libero_dataset MODEL_ID=HuggingFaceVLA/smolvla_libero N_EPISODES=10 BATCH_SIZE=1 bash eval_smolvla_libero_10.sh

DATASET_ROOT="${DATASET_ROOT:-$HOME/libero_dataset}"
MODEL_ID="${MODEL_ID:-HuggingFaceVLA/smolvla_libero}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
N_EPISODES="${N_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval/smolvla_libero_10}"
MAX_EPISODES_RENDERED="${MAX_EPISODES_RENDERED:-10}"
OBSERVATION_HEIGHT="${OBSERVATION_HEIGHT:-256}"
OBSERVATION_WIDTH="${OBSERVATION_WIDTH:-256}"
export DATASET_ROOT

# Headless rendering backend for MuJoCo.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# LIBERO reads paths from $LIBERO_CONFIG_PATH/config.yaml.
# We generate that config so bddl/init/assets resolve from the installed LIBERO package,
# while datasets resolve to your local DATASET_ROOT.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
mkdir -p "$LIBERO_CONFIG_PATH"

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

lerobot-eval \
  --policy.path="$MODEL_ID" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task="$TASK_SUITE" \
  --env.render_mode=rgb_array \
  --env.max_parallel_tasks=1 \
  --env.observation_height="$OBSERVATION_HEIGHT" \
  --env.observation_width="$OBSERVATION_WIDTH" \
  --eval.batch_size="$BATCH_SIZE" \
  --eval.use_async_envs=false \
  --eval.n_episodes="$N_EPISODES" \
  --eval.max_episodes_rendered="$MAX_EPISODES_RENDERED" \
  --output_dir="$OUTPUT_DIR"
