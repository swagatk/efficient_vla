#!/usr/bin/env bash
# reproduce_baseline_task1.sh
# Reproduce SmolVLA Baseline on LIBERO Task 1 using the hybrid_diffusion evaluation pipeline.
#
# Task 1: "put both the cream cheese box and the butter in the basket"
# Expected Baseline Performance: 60-70% success rate across 10 evaluation episodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EPISODES="${1:-10}"
OUTPUT_DIR="${2:-./outputs/baseline_task1_reproduce_$(date +%Y%m%d_%H%M%S)}"

echo "============================================================"
echo "🎯 Reproducing Baseline SmolVLA Performance on Task 1"
echo "============================================================"
echo "  • Task ID:             1 (cream cheese box & butter into basket)"
echo "  • Policy Mode:         base (Frozen SmolVLA)"
echo "  • Episodes:            $EPISODES"
echo "  • Horizon / Episode:   520 steps"
echo "  • Camera Flip:         vertical_horizontal (agentview & wrist)"
echo "  • Output Directory:    $OUTPUT_DIR"
echo "============================================================"

bash eval_hybrid_diffusion.sh \
  --policy_mode base \
  --task_id 1 \
  --episodes "$EPISODES" \
  --max_steps 520 \
  --flip_mode vertical_horizontal \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "Evaluation Finished! Results saved in $OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/eval_results.json" ]]; then
  echo "Summary:"
  cat "$OUTPUT_DIR/eval_results.json"
fi
echo "============================================================"
