# How to Run the Gated Residual Strategy

## Phase 1: Data Collection

First, you need to collect failure data from the baseline SmolVLA policy:

```bash
# Run data collection for all tasks and seeds (recommended)
python collect_failure_data.py --run_all --num_episodes 10

# Or run for a single task and seed for testing
python collect_failure_data.py --task_id 0 --seed 0 --num_episodes 5

# or use the shell script
bash run_phase1_baselines.sh 
```



This will create files in `Gated_Residual_strategy/data/`:
- `failure_dataset_task{task_id}_seed{seed}.h5` - HDF5 files with observations, actions, and labels
- `logs_task{task_id}_seed{seed}.json` - Detailed per-step logs

### Dataset Quality Evaluation

After collecting the data, evaluate its quality:

```bash
# Evaluate dataset quality
python eval_phase1_dataset.py --data_dir ./data --output_dir ./dataset_analysis

# Or use the shell script
DATA_DIR=./outputs/run_20260624_232108 bash run_dataset_evaluation.sh
```

This will generate:
- `dataset_quality_report.json` - Detailed analysis of dataset quality
- `class_distribution.png` - Visualization of class balance
- Console output with quality assessment

## Phase 2: Train the Failure-Risk Gate

Once you have a quality dataset, train the failure-risk gate:

```bash
# Execute the Phase 2 training across multiple seeds
DATA_DIR=./outputs/run_20260624_232108 bash run_phase2_train_gate.sh
```

This will:
1. Train the lightweight binary classifier on the failure dataset
2. Log results to Weights & Biases
3. Save model checkpoints

### Evaluate Results

Finally, evaluate the trained gate:

```bash
# Evaluate the trained gate
python eval_gated_baseline.py --run_all
```

This will:
1. Run evaluation with the trained gate
2. Generate performance metrics with confidence intervals
3. Compare results to baseline

## Troubleshooting

If you encounter issues:

1. **Missing data files**: Make sure to run `collect_failure_data.py` first
2. **Import errors**: Ensure all dependencies are installed
3. **CUDA errors**: Check GPU availability and memory
4. **WandB issues**: Set up Weights & Biases account and login

## Expected Results

After successful execution, you should have:

1. **Collected Data**: 30 HDF5 files (3 seeds × 10 tasks) with observations and failure labels
2. **Quality Report**: Dataset analysis showing balanced class distribution
3. **Trained Models**: Failure-risk gate models with good validation performance
4. **Evaluation Results**: Performance metrics showing improvement over baseline