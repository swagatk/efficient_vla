# Gated Residual Strategy - Phase 1: Data Collection & Benchmark Harness

## Overview

This folder contains the infrastructure for **Phase 1** of the Gated Residual Strategy execution roadmap. The goal is to:
1. **Collect failure data** from baseline SmolVLA rollouts
2. **Establish a robust evaluation harness** with per-step success/failure logging

## Files

### `collect_failure_data.py`
**Purpose:** Runs baseline SmolVLA and collects trajectories with success/failure labels.

**Usage:**
```bash
# Single task and seed
python collect_failure_data.py --task_id 0 --seed 0 --num_episodes 10

# All tasks and seeds (recommended for full dataset)
python collect_failure_data.py --run_all --num_episodes 10
```

**Outputs:**
- `failure_dataset_task{task_id}_seed{seed}.h5`: HDF5 file with observations, actions, and labels
- `logs_task{task_id}_seed{seed}.json`: Detailed per-step logs

**Key Features:**
- Logs binary labels: `0` = success, `1` = failure
- Saves per-trajectory success flags
- Generates detailed step-level metadata

### `eval_gated_baseline.py`
**Purpose:** Enhanced evaluation script with statistical analysis and confidence intervals.

**Usage:**
```bash
# Single task and seed
python eval_gated_baseline.py --task_id 0 --seed 0

# All tasks and seeds
python eval_gated_baseline.py --run_all
```

**Outputs:**
- Per-task JSON results with confidence intervals
- Aggregate results across all tasks
- Detailed episode-level success logs

**Key Features:**
- 95% confidence interval calculations
- Per-step success/failure tracking
- Statistical analysis of success rates
- Easy comparison between baseline and future gated strategies

### `run_phase2_train_gate.sh`
**Purpose:** Phase 2 Failure-Risk Gate Training Orchestrator. Trains a lightweight binary classifier (`train_failure_risk_gate.py`) on the failure dataset collected in Phase 1.

**Usage:**
```bash
# Execute the Phase 2 training across multiple seeds
bash run_phase2_train_gate.sh
```

**Key Features:**
- Resumable execution across 3 random seeds
- Weights & Biases logging and run continuity
- Progress heartbeat logging and system power profile hardening

## Phase 1

### Step 1: Collect Failure Data
```bash
# Generate baseline failure dataset (3 seeds, 10 episodes each)
python collect_failure_data.py --run_all --num_episodes 10
```

This will create files like:
```
Gated_Residual_strategy/data/
├── failure_dataset_task0_seed0.h5
├── failure_dataset_task0_seed1.h5
├── failure_dataset_task0_seed2.h5
├── logs_task0_seed0.json
├── logs_task0_seed1.json
└── ... (for all 10 tasks)
```

### Step 2: Run Evaluation Harness
```bash
# Evaluate baseline performance across all tasks
python eval_gated_baseline.py --run_all
```

This will create:
```
Gated_Residual_strategy/eval_results/
├── task_0_results_20250601_120000.json
├── task_1_results_20250601_120000.json
├── ...
└── aggregate_results_20250601_120000.json
```

### Step 3: Analyze Results
```python
import json

# Load aggregate results
with open("Gated_Residual_strategy/eval_results/aggregate_results_*.json") as f:
    results = json.load(f)

print(f"Overall Success Rate: {results['overall_mean_success_rate']:.2%}")
print(f"95% CI: [{results['ci_95_lower']:.2%}, {results['ci_95_upper']:.2%}]")
```

## Next Steps (Phase 2)

Once you have collected the failure data:
1. **Train the Failure-Risk Gate** (Week B)
   - Use the collected labels to train a binary classifier
   - Input: observations from failure states
   - Output: probability of failure

**Execution:**
```bash
bash run_phase2_train_gate.sh
```

## Phase 3
2. **Train the Residual Corrector** (Week C)
   - Train on failure trajectories only
   - Learn corrective actions for failure states


## Phase 4
3. **Implement Gated Inference** (Week C)
   - Combine base policy + gated residual
   - Test on held-out failure cases

## Phase 5: Ablation.

## Important Notes

- **Data Quality:** Ensure your baseline rollout logic correctly identifies success/failure
- **Label Consistency:** The binary labels must be consistent across all tasks
- **Seed Diversity:** Use at least 3 seeds per task for statistical significance
- **Reproducibility:** All scripts use fixed seeds and deterministic evaluation

## Troubleshooting

### "No module named 'lerobot'"
```bash
pip install lerobot==0.4.0
```

### "Failed to import FastSAM"
See previous discussion - install via:
```bash
pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git --no-build-isolation
```

### "HDF5 file too large"
- Reduce `--num_episodes` for initial testing
- Use compression in HDF5 (add `compression='gzip'` to `h5py.File`)

## Success Criteria for Phase 1

✅ **Completed when:**
- [ ] Failure dataset collected for all 10 tasks (3 seeds each)
- [ ] Baseline success rates established with confidence intervals
- [ ] Per-step success/failure labels are consistent and accurate
- [ ] Evaluation harness can reproduce baseline results

**Go/No-Go Threshold:**
- If baseline success rate is <50% on any task, consider adjusting task difficulty
- If failure rates are too low (<5%), increase `--num_episodes`
- If failure rates are too high (>90%), the residual may not have enough to learn from

## Contact

For questions or issues, refer to the main project documentation or the `plan2_June_01.md` roadmap.
