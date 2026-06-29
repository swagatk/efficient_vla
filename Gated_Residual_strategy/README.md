# Gated Residual Strategy (Self-Correcting Edge VLA)

This folder contains the complete implementation for the **Gated Residual Strategy** on LIBERO-10. 
The core idea is to keep a frozen base VLA policy (SmolVLA) as the primary controller, train a lightweight binary **Failure-Risk Gate** to trigger whenever intervention is needed, and train a **Residual Corrector** to apply corrective control actions during these high-risk intervals.

---

## Workspace Layout & Key Files

| Phase | Python Script | Bash Orchestrator Script | Purpose |
| :--- | :--- | :--- | :--- |
| **Phase 1** | [collect_failure_data.py](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/collect_failure_data.py)<br>[eval_phase1_dataset.py](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/eval_phase1_dataset.py) | `run_phase1_baseline.sh`<br>[run_dataset_evaluation.sh](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/run_dataset_evaluation.sh) | Rollout baseline to collect failure data, then analyze dataset quality. |
| **Phase 2** | [train_failure_risk_gate.py](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/train_failure_risk_gate.py) | [run_phase2_train_gate.sh](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/run_phase2_train_gate.sh) | Train the failure-risk gate model on rollout observations and labels. |
| **Phase 3** | [train_residual_corrector.py](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/train_residual_corrector.py) | [run_phase3_train_corrector.sh](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/run_phase3_train_corrector.sh) | Train the residual corrector model to predict successful recovery actions. |
| **Phase 4** | [eval_gated_baseline.py](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/eval_gated_baseline.py) | [run_phase4_eval.sh](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/run_phase4_eval.sh) | Evaluate the full gated residual system inside the LIBERO simulator. |

---

## Step-by-Step Execution Guide

> [!IMPORTANT]
> Running the base VLA policy (SmolVLA) and simulator environments requires a GPU (ideally 16GB+ VRAM) and an active Conda environment with `lerobot_v040` (and `robosuite`/`libero` dependencies installed).

### Phase 1: Data Collection & Quality Evaluation

#### 1. Collect Rollout Data
Run SmolVLA baseline to gather HDF5 datasets containing observations, actions, and binary success/failure trajectory labels.
* **Python Command:**
  ```bash
  python collect_failure_data.py --run_all --num_episodes 10 --output_dir ./data
  ```
* **Bash Orchestrator Command:**
  ```bash
  bash run_phase1_baseline.sh
  ```

#### 2. Evaluate Dataset Quality
Verify class distribution and dataset consistency before training models.
* **Python Command:**
  ```bash
  python eval_phase1_dataset.py --data_dir ./outputs/run_20260624_232108 --output_dir ./dataset_analysis
  ```
* **Bash Orchestrator Command:**
  ```bash
  DATA_DIR=./outputs/run_20260624_232108 bash run_dataset_evaluation.sh
  ```

---

### Phase 2: Train the Failure-Risk Gate
Train a binary classifier to predict risk of impending task failure based on observation features.
* **Python Command:**
  ```bash
  python train_failure_risk_gate.py \
      --data_dir ./outputs/run_20260624_232108 \
      --output_dir ./outputs/phase2_train_single \
      --seed 42 \
      --epochs 15 \
      --wandb_project gated_residual_phase2
  ```
* **Bash Orchestrator Command (Recommended: trains across 3 seeds):**
  ```bash
  DATA_DIR=./outputs/run_20260624_232108 \
  PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python \
  bash run_phase2_train_gate.sh
  ```

---

### Phase 3: Train the Gated Residual Corrector
Train the action corrector model on successful nominal rollout steps (`target_label=0`) to predict corrective recovery control vectors.
* **Python Command:**
  ```bash
  python train_residual_corrector.py \
      --data_dir ./outputs/run_20260624_232108 \
      --output_dir ./outputs/phase3_train_single \
      --target_label 0 \
      --seed 42 \
      --epochs 15 \
      --wandb_project gated_residual_phase3
  ```
* **Bash Orchestrator Command (Recommended: trains across 3 seeds):**
  ```bash
  DATA_DIR=./outputs/run_20260624_232108 \
  PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python \
  bash run_phase3_train_corrector.sh
  ```

---

### Phase 4: Gated Residual Strategy Evaluation
Run the integrated policy (SmolVLA base + Failure Gate + Residual Corrector) in the simulator. When the gate probability exceeds `threshold`, actions are blended with the corrector using interpolation scale `alpha`.
* **Python Command (Single Task/Seed):**
  ```bash
  python eval_gated_baseline.py \
      --task_id 0 \
      --seed 0 \
      --gate_dir ./outputs/phase2_train_20260626_101134 \
      --corrector_dir ./outputs/phase3_train_20260626_110840 \
      --threshold 0.5 \
      --alpha 0.5 \
      --num_episodes 10 \
      --output_dir ./eval_results
  ```
* **Bash Orchestrator Command (Recommended: executes sequentially task-by-task to avoid memory leaks):**
  ```bash
  GATE_DIR=./outputs/phase2_train_20260626_101134 \
  CORRECTOR_DIR=./outputs/phase3_train_20260626_110840 \
  THRESHOLD=0.5 \
  ALPHA=0.5 \
  NUM_EPISODES=10 \
  PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python \
  bash run_phase4_eval.sh
  ```

---

## Troubleshooting & Key Settings

1. **Power Hardening & Auto-Sleep:** The bash orchestrator scripts dynamically query and set system power profiles to `performance` and turn off sleep-on-AC settings. These settings are registered to auto-restore on script completion or interrupt (Ctrl+C).
2. **Out of Memory / Memory Leaks:** If running evaluation across multiple seeds in a single python script crashes your GPU/RAM, use the [run_phase4_eval.sh](file:///home/swagat/GIT/efficient_vla/Gated_Residual_strategy/run_phase4_eval.sh) orchestrator which runs tasks isolated in individual subprocesses.
3. **WandB Setup:** Ensure you are logged into Weights & Biases (`wandb login`) before training so that progress maps properly.
