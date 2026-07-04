#!/usr/bin/env bash

# Script to evaluate the quality of Phase 1 dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/dataset_analysis_$(date +%Y%m%d_%H%M%S)}"

echo "Evaluating Phase 1 dataset quality..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the evaluation script
python "$SCRIPT_DIR/eval_phase1_dataset.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Dataset evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"