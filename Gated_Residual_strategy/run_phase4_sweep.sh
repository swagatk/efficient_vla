#!/usr/bin/env bash
# run_phase4_sweep.sh
# Runs a hyperparameter sweep over threshold and alpha for Phase 4 Gated Residual Evaluation.
#
# To run normally:
#   bash run_phase4_sweep.sh
# To run a dry run:
#   SCRIPT_DRY_RUN=1 bash run_phase4_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GATE_DIR="${GATE_DIR:-/home/swagat/GIT/efficient_vla/Gated_Residual_strategy/outputs/phase2_train_20260707_191607}"
export CORRECTOR_DIR="${CORRECTOR_DIR:-/home/swagat/GIT/efficient_vla/Gated_Residual_strategy/outputs/phase3_train_20260707_225150}"
export PYTHON_BIN="${PYTHON_BIN:-/home/swagat/anaconda3/envs/lerobot_v040/bin/python}"

# Hyperparameters to sweep
# Format: "threshold,alpha,name"
CONFIGS=(
  "0.5,0.1,config_A"
  "0.9,0.5,config_B"
  "0.7,0.3,config_C"
)

# Dry run settings vs normal settings
if [[ "${SCRIPT_DRY_RUN:-0}" == "1" ]]; then
  echo "=== Running in DRY RUN mode ==="
  export NUM_EPISODES=1
  export SEEDS="0"
  export TASKS="2"
else
  export NUM_EPISODES=10
  export SEEDS="0 1 2"
  export TASKS="2 7 9"
fi

export ACTIVE_GATING_TASKS="2 7 9"
export ADAPTIVE_GATING=1
export INFERENCE_MODE="delta"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="$SCRIPT_DIR/outputs/phase4_sweep_${TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

echo "Sweep results will be saved to: $SWEEP_DIR"
echo "Active Gating Tasks: $ACTIVE_GATING_TASKS"
echo "Seeds: $SEEDS"
echo "Tasks: $TASKS"
echo "Episodes: $NUM_EPISODES"
echo "============================================="

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r thresh alpha name <<< "$cfg"
  
  echo ""
  echo "---------------------------------------------"
  echo "Running Sweep Config: $name (threshold=$thresh, alpha=$alpha)"
  echo "---------------------------------------------"
  
  export THRESHOLD="$thresh"
  export ALPHA="$alpha"
  export OUTPUT_DIR="$SWEEP_DIR/$name"
  
  # Run the modified phase4 eval orchestrator
  bash "$SCRIPT_DIR/run_phase4_eval.sh"
  
  echo "Finished Config: $name"
done

echo ""
echo "============================================="
echo "SWEEP COMPLETED SUCCESSFULLY"
echo "Results available in: $SWEEP_DIR"
echo "============================================="
