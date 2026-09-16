# Supervised Task-Specific LoRA Adapters for SmolVLA

This module implements **Supervised Task-Specific Low-Rank Adaptation (LoRA)** on SmolVLA for the LIBERO-10 benchmark.

Instead of training external residual networks or diffusion heads, this approach applies LoRA adapters ($r=16, \alpha=32$) directly to the linear projections of SmolVLA's Action Expert (`lm_expert`), preserving full token-level vision-language cross-attention while adapting trajectory generation to task geometry with **zero added inference latency**.

---

## 1. Environment & Prerequisites

Activate the environment before executing training or evaluation scripts:

```bash
conda activate lerobot_env
cd /home/swagat/GIT/efficient_vla/supervised_task_specific_lora_adapters
```

* Ensure `peft`, `h5py`, and `wandb` are installed in the conda environment.
* Demonstration datasets are loaded from `DATA_DIR` (defaults to `/home/swagat/libero_dataset/libero_10`).
* System power management is handled automatically:
  * **On Windows WSL**: The script dynamically switches Windows 11 host power to *High Performance* and sets standby sleep timeout to 0 (never sleep) via `powercfg.exe`, automatically restoring the original scheme and timeouts upon completion or exit.
  * **On Native Linux**: Handled via `systemd-inhibit` or `gnome-session-inhibit`.

---

## 2. Training Workflows

### A. Train a Single Task (e.g., Task 0)
To train a LoRA adapter for a single difficult task before running everything:

```bash
TASKS="0" bash run_lora_training.sh
```

### B. Train All Target Tasks
To sequentially train adapters across the standard 8 evaluated tasks (`0, 1, 2, 4, 6, 7, 8, 9`):

```bash
bash run_lora_training.sh
```

### C. Resume an Interrupted Training Run
The training script saves PEFT adapter weights and updates `config.json` at the end of **every single epoch**. If training is interrupted:

* **Resume latest run automatically:**
  ```bash
  RESUME=1 bash run_lora_training.sh
  ```

* **Resume a specific run directory:**
  ```bash
  RESUME_DIR=outputs/run_lora_training_20260828_215901 bash run_lora_training.sh
  ```

### D. Customizing Hyperparameters via Environment Variables
You can override training hyperparameters directly when invoking the shell runner:

```bash
EPOCHS=25 BATCH_SIZE=4 LR=1e-4 LORA_RANK=16 TASKS="0 1" bash run_lora_training.sh
```

---

## 3. Evaluation Workflows

Closed-loop evaluation runs the policy in the LIBERO simulation environment and reports success rates.

### A. Evaluate Single Task (e.g., Task 0 Across 3 Seeds)
Evaluate Task 0 over 3 seeds (10 episodes per seed, 30 episodes total):

```bash
python eval_lora_policy.py \
  --task_id 0 \
  --lora_dir outputs/run_lora_training_YYYYMMDD_HHMMSS \
  --num_episodes 10 \
  --output_dir lora_eval_results_task0
```

### B. Evaluate All 10 Tasks and 3 Seeds
Run full evaluation across all tasks with trained LoRA adapters:

```bash
python eval_lora_policy.py \
  --lora_dir outputs/run_lora_training_YYYYMMDD_HHMMSS \
  --run_all \
  --num_episodes 10 \
  --output_dir lora_eval_results_all
```

### C. Evaluate Baseline SmolVLA (No LoRA)
To benchmark baseline SmolVLA without applying LoRA adapters:

```bash
python eval_lora_policy.py \
  --task_id 0 \
  --num_episodes 10 \
  --output_dir baseline_eval_results
```

---

## 4. Key Files & Structure

```
supervised_task_specific_lora_adapters/
├── README.md               # User guide and command instructions
├── train_supervised_lora.py # Supervised BC training script with PEFT LoRA
├── eval_lora_policy.py     # Closed-loop LIBERO simulator evaluation harness
├── run_lora_training.sh    # Multi-task sequential runner with resume support
├── outputs/                # Timestamped run checkpoints and configs
│   └── run_lora_training_YYYYMMDD_HHMMSS/
│       ├── config.json
│       ├── task_0/
│       │   ├── adapter_model.safetensors # Compact LoRA weights (~19MB)
│       │   ├── adapter_config.json
│       │   └── config.json
│       └── ...
└── implement_plan.md       # Technical design and architecture specification
```

---

## 5. Summary of Hyperparameters

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `EPOCHS` | `20` | Training epochs per task |
| `BATCH_SIZE` | `2` | Batch size per step |
| `GRAD_ACCUM_STEPS`| `8` | Effective batch size = `16` |
| `LR` | `1e-4` | AdamW learning rate |
| `LORA_RANK` ($r$) | `16` | LoRA rank on `lm_expert` |
| `LORA_ALPHA` ($\alpha$) | `32` | LoRA scaling factor |
| Target Modules | `q, k, v, o, gate, up, down` | Linear projections in Gemma Action Expert |
