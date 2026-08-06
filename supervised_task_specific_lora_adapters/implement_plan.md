# Implementation Plan - Supervised Task-Specific LoRA Adapters

This plan outlines the implementation of **Supervised Task-Specific LoRA Adapters** for SmolVLA. Instead of training external residual heads (which can introduce high-frequency adversarial noise and require a separate Failure Gate), we will train Low-Rank Adaptation (LoRA) layers directly inside SmolVLA's Action Expert (`lm_expert`) using human demonstrations from `libero_10`. 

This approach preserves the pre-trained VLM's visual attention and representations while adapting the final action generation directly to the target task's geometry.

---

## User Review Required

> [!IMPORTANT]
> **Dependency Installation:**
> Implementing LoRA requires the `peft` library. Since `peft` is not currently installed in the conda environment, we will need to run:
> ```bash
> /home/swagat/anaconda3/envs/lerobot_v040/bin/python -m pip install peft
> ```
> Please confirm if you approve running this installation command.

> [!NOTE]
> **LoRA Target Modules:**
> We will target all key linear projections in the Gemma-based Action Expert (`lm_expert`), specifically:
> `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
> This guarantees maximum representational capacity for fine-tuning while leaving the frozen VLM vision encoder and core weights completely untouched, saving significant VRAM (easily fitting under 10 GB).

---

## Open Questions

1. **Rank & Alpha Selection:**
   We propose a baseline configuration of rank $r=16$ and scaling factor $\alpha=32$ with a dropout of $0.05$. Should we include arguments to sweep these hyperparameters later?
2. **Epoch Count & Batch Size:**
   We propose training for $20$ epochs per task (batch size $64$, learning rate $1\times 10^{-4}$) to match the offline training scale of the residual correctors. Does this timeline align with your training limits?

---

## Proposed Changes

### Supervised LoRA Components

#### [NEW] [train_supervised_lora.py](file:///home/swagat/GIT/efficient_vla/train_supervised_lora.py)
A custom supervised training script that will:
* **Load Dataset:** Parse task-specific LIBERO-10 demonstration HDF5 files from `/home/swagat/libero_dataset/libero_10` using `h5py` and format them to match LeRobot/SmolVLA input expectations.
* **Integrate LoRA:** Load `SmolVLAPolicy` and wrap its `self.model.vlm_with_expert.lm_expert` module using `peft.get_peft_model` with a `LoraConfig`.
* **Supervised Training (BC):** Optimize only the trainable LoRA adapters against the standard Flow Matching loss (`F.mse_loss(predicted_flow, target_flow)`) computed by calling the policy.
* **Metrics Logging:** Log training/validation loss trends to Weight & Biases (wandb).
* **Model Checkpoints:** Save task-specific PEFT LoRA adapter weights (which are small, typically ~15-20MB) under `checkpoints_lora_task_{task_id}/`.

#### [NEW] [eval_lora_policy.py](file:///home/swagat/GIT/efficient_vla/eval_lora_policy.py)
A closed-loop evaluation script (similar to `eval_gated_baseline.py` but for LoRA) that will:
* Load the base `SmolVLAPolicy`.
* Load the trained task-specific LoRA weights from the selected checkpoint and merge/load them into the policy.
* Spin up the LIBERO-10 OffScreen simulator environment.
* Run standard rollout evaluations (10 episodes per task, 3 seeds) and generate overall success rates.

#### [NEW] [run_lora_training.sh](file:///home/swagat/GIT/efficient_vla/run_lora_training.sh)
A helper shell script to launch supervised LoRA training sequentially across the target tasks.

---

## Verification Plan

### Automated Tests
* **Installation Test:** Ensure `peft` is installed and can be imported correctly.
* **Dataset Dry-Run:** Run a single-epoch test of `train_supervised_lora.py` on Task 0, Seed 0 with a subset of data to verify loss reduction and checkpoint saving.
  ```bash
  python train_supervised_lora.py --task_id 0 --epochs 1 --batch_size 16 --dry_run
  ```
* **Evaluation Dry-Run:** Run closed-loop rollouts for 1 episode on Task 0, Seed 0 to ensure the PEFT model runs successfully in the simulator without execution or shape mismatches.
  ```bash
  python eval_lora_policy.py --task_id 0 --seed 0 --num_episodes 1
  ```
