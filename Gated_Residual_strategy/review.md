# Review & Refined Gated Residual Strategy

This document reviews and corrects the Gated Residual Strategy implementation plan based on empirical validation of the dataset and models.

---

## Phase 1: Data Collection and Analysis
- **Protocol:** Collect failure data for all tasks (from task 0 to 9) and for 3 seeds each using the SmolVLA baseline model.
- Keep a localized `failure_window` (e.g., 30–50 steps) representing the actual divergence/critical failure region.
- Analyze the dataset, report the baseline model to be used for comparison in phase 4.

To start fresh execution
```
RESUME=0 \
PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
bash run_phase1_baseline.sh

```
To resume an interrupted run:
```
RESUME=1 \
OUTPUT_ROOT=outputs/phase1_run_20260715_121907 \
PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
bash run_phase1_baseline.sh

```
---

## Phase 2: Failure Prediction Model Training
- **Task-Specific Models:** Train 10 separate gates for the 10 tasks.
- **Epochs:** Increase to 20–30 epochs to ensure validation AUC converges (watch out for class imbalance: use weighted BCE loss).

Command to start fresh execution:
```
RESUME=0 \ 
DATA_DIR=outputs/phase1_run_20260715_121907 \ 
PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python \ 
bash run_phase2_task_specific_gate.sh
```

Command to resume execution:
```
RESUME=1 DATA_DIR=outputs/phase1_run_20260715_121907 OUTPUT_ROOT=outputs/phase2_task_specific_train_20260716_125525 PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python bash run_phase2_task_specific_gate.sh

```

---

## Phase 3: Train the Corrector
- **Model Capacity Warning:**
  > [!IMPORTANT]
  > The current `LightweightResidualCorrector` (CNN + MLP, ~244 KB) is too small to learn complex Libero action trajectories. Validation MSE on delta mode was ~0.07, which is worse than or equal to predicting a constant zero vector (Zero-Predictor MSE $\approx$ 0.05 - 0.07).
  > 
  > **To make this work, you must:**
  > 1. Increase model capacity (e.g., use a larger fusion MLP with 128/256 hidden units, or a ResNet-18 backbone).
  > 2. Increase training to **50–100 epochs**.
  > 3. Optionally use history/temporal state representations.

- **Options to Try:**
  - **Option 1: Self-Imitation Absolute:** Train task-specific correctors on Phase 1 successful rollout trajectories using `train_mode=absolute`.
  - **Option 2: Demonstration-Anchored Delta (Recommended):** Precompute baseline actions on the Libero-10 human demonstration dataset first using `precompute_base_actions.py`, then train task-specific correctors to predict actions relative to demonstrations using `train_mode=delta`.

Command for absolute mode:

```
RESUME=0 DATA_DIR=outputs/phase1_run_20260715_121907 TRAIN_MODE=absolute PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python bash run_phase3_task_specific_corrector.sh
```

Command for delta mode:

```
RESUME=0 DATA_DIR=/home/swagat/libero_dataset/libero_10 TRAIN_MODE=delta PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python bash run_phase3_task_specific_corrector.sh
```

---

## Phase 4: Evaluation & Mathematical Corrections

### Blending Formulas

1. **Delta Mode Blending (Hard/Threshold Gating):**
   ```python
   if gate_prob > threshold:
       final_action = baseline_action + alpha * delta_corrector_action
   else:
       final_action = baseline_action
   ```

2. **Absolute Mode Blending (Hard/Threshold Gating):**
   ```python
   if gate_prob > threshold:
       final_action = (1.0 - alpha) * baseline_action + alpha * corrector_action
   else:
       final_action = baseline_action
   ```
Command to run phase 4 in delta mode with adaptive gating
```
RESUME=0 ADAPTIVE_GATING=1 ACTIVE_GATING_TASKS="2 7 9" INFERENCE_MODE=delta THRESHOLD=0.5 ALPHA=0.5 GATE_DIR=outputs/phase2_task_specific_train_20260716_125525 CORRECTOR_DIR=outputs/phase3_task_specific_train_20260717_003320 PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python bash run_phase4_eval.sh
```
To resume interrupted run
```
RESUME=1 ADAPTIVE_GATING=1 ACTIVE_GATING_TASKS="2 7 9" INFERENCE_MODE=delta THRESHOLD=0.5 ALPHA=0.5 GATE_DIR=outputs/phase2_task_specific_train_20260716_125525 CORRECTOR_DIR=outputs/phase3_task_specific_train_20260717_003320 OUTPUT_DIR=outputs/phase4_eval_results_20260717_101320 PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python bash run_phase4_eval.sh
```
command to run in absolute mode with adaptive gating
```
RESUME=0 \
ADAPTIVE_GATING=1 \
ACTIVE_GATING_TASKS="2 7 9" \
INFERENCE_MODE=absolute \
THRESHOLD=0.5 \
ALPHA=0.5 \
GATE_DIR=outputs/phase2_task_specific_train_20260716_125525 \
CORRECTOR_DIR=outputs/phase3_task_specific_train_20260716_210747 \
PYTHON_BIN=/home/swagat/anaconda3/envs/lerobot_v040/bin/python \
bash run_phase4_eval.sh

```

---

## Phase 5: Hyperparameter Sweeps & Ablations
Once the corrector model capacity is upgraded and trained:
- **Sweep Threshold:** Test `threshold` values in `[0.3, 0.5, 0.7]` (determines how early/sensitively the gate triggers).
- **Sweep Alpha:** Test blending weight `alpha` in `[0.3, 0.5, 0.7, 1.0]` (determines the magnitude of the corrective offset).

---

## Crucial Recommendations for Better Performance

### 1. Pre-trained Visual Backbone (Instead of CNN from Scratch)
Training a 3-layer CNN from scratch on a small dataset (around 10k frames) leads to poor feature representations. 
* **Action:** Replace the custom CNN in `LightweightFailureGate` and `LightweightResidualCorrector` with a pre-trained feature extractor, such as:
  * A frozen **ResNet-18** from `torchvision.models` (pretrained on ImageNet).
  * Direct projection of the frozen **SigLIP/vision tower embeddings** from the base SmolVLA policy.

### 2. Failure Data Augmentation
Because SmolVLA baseline success rate is relatively high (42.33%), tasks like Task 3 have very few failed trajectories (only 1 or 2 failed runs), resulting in extreme class imbalance for the Failure Gate.
* **Action:** Inject small random Gaussian noise ($\sigma \approx 0.05$) into the baseline actions during Phase 1 data collection. This forces the base policy to fail more frequently and dynamically, generating a much larger and more diverse dataset of failure states.

### 3. Action Loss Normalization
Standard actions control both end-effector position/rotation changes (typically very small, e.g., $\pm 0.02$) and the gripper (binary $\pm 1.0$). A simple MSE loss will be heavily dominated by the gripper state, neglecting the position control.
* **Action:** Weight the loss function or normalize action dimensions during corrector training so position, orientation, and gripper errors contribute more evenly to gradients.
