# Walkthrough - Supervised Task-Specific LoRA Adapters

We have implemented and successfully verified the **Supervised Task-Specific LoRA Adapters** workflow for SmolVLA. This framework replaces external residual architectures with direct weight adaptations on the SmolVLA Action Expert model using human demonstration datasets.

---

## 1. Accomplishments & Code Additions

### Dependency Installation
Installed the `peft` parameter-efficient fine-tuning library inside the conda environment `/home/swagat/anaconda3/envs/lerobot_v040`.

### Training Pipeline
* **[train_supervised_lora.py](file:///home/swagat/GIT/efficient_vla/train_supervised_lora.py)**:
  * Implemented `SupervisedLoRADataset` to parse nested LIBERO-10 demonstration files. It extracts target action chunks (size 50) and active observation streams.
  * Configured PEFT `LoraConfig` (rank=16, alpha=32) targeting all attention and MLP projection layers `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` of the Gemma-based Action Expert (`lm_expert`).
  * Structured the collate function to preprocess data with LeRobot's formatting.
  * Added mixed-precision training using `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` to guarantee compatiblity and prevent VRAM exhaustion.
  * Added `--grad_accum_steps` and `--gradient_checkpointing` for memory efficiency. Setting `--batch_size 8 --grad_accum_steps 2` maintains the effective batch size of 16 while significantly reducing VRAM/RAM overhead.
  * Implemented garbage collection (`gc.collect()`, `torch.cuda.empty_cache()`) and HDF5 dataset closing (`train_dataset.close()`, `val_dataset.close()`) after each epoch to prevent memory leaks.
  * Set up logging to Weight & Biases and adapter checkpoint saving.

### Evaluation Pipeline
* **[eval_lora_policy.py](file:///home/swagat/GIT/efficient_vla/eval_lora_policy.py)**:
  * Implemented an off-screen rollout simulator evaluator.
  * Integrates saved adapter weights dynamically back into the base policy expert on a per-task basis using `PeftModel.from_pretrained()`.
  * Runs closed-loop rollouts (using `autocast` and `postprocessor`) and aggregates success statistics across tasks.

### Orchestration Script
* **[run_lora_training.sh](file:///home/swagat/GIT/efficient_vla/run_lora_training.sh)**: 
  * A helper script that sequences training across all target tasks and supports quick validation via `DRY_RUN=1`.
  * Updated to default to memory-optimized training settings (`BATCH_SIZE=8`, `GRAD_ACCUM_STEPS=2`, `GRADIENT_CHECKPOINTING=0`).
  * Note: `GRADIENT_CHECKPOINTING` is disabled by default because PyTorch's gradient checkpointing engine conflicts with models where most parameters are frozen (like the base VLM in our PEFT setup), triggering Segmentation Faults. Fortunately, with `BATCH_SIZE=8`, VRAM usage is already extremely low (~3.9 GB out of 16 GB), making checkpointing completely unnecessary.
  * Added support for task completion tracking via `.completed` files inside each task folder, allowing training to resume exactly at the interrupted task.

---

## 2. Validation & Verification Results

### A. Dry-Run Supervised Training
Tested the dataset loading, PEFT model wrapping, backprop step, and logging with:
```bash
DRY_RUN=1 TASKS="0" bash run_lora_training.sh
```
* **Status:** Passed successfully.
* **Outputs:** 
  * Target trainable parameters: `4,915,200` (~4.8% of base VLA model).
  * Dataset parsed `12,495` training steps and `2,205` validation steps on Task 0.
  * Computed first epoch under autocast (Train Loss: `0.05967`, Val Loss: `0.08237`).
  * Synced training run correctly with Weights & Biases dashboard.

### B. Dry-Run Simulation Rollouts
Tested the simulator execution and step actions with:
```bash
/home/swagat/anaconda3/envs/lerobot_v040/bin/python eval_lora_policy.py --task_id 0 --seed 0 --num_episodes 1
```
* **Status:** Passed successfully.
* **Outputs:**
  * Loaded the base SmolVLA policy and successfully initialized Robosuite/MuJoCo engine.
  * Executed a complete closed-loop episode for 520 steps.
  * Completed and returned evaluation summary results without errors.
