#!/usr/bin/env python3
import os
os.environ["WANDB_SYSTEM_MONITOR"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/home/swagat/lerobot/src")

# Pre-import transformers
import transformers
import argparse
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import wandb
from datetime import datetime

# Import LeRobot/SmolVLA modules
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Safe out-of-place RoPE implementation to prevent CUDA slice mutation segfaults
import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_expert_module

_TIMESCALE_CACHE = {}

def _get_cached_timescale(dim, device, max_wavelength=10000):
    key = (dim, str(device), max_wavelength)
    if key not in _TIMESCALE_CACHE:
        _TIMESCALE_CACHE[key] = max_wavelength ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    return _TIMESCALE_CACHE[key]

def _safe_apply_rope(x, positions, max_wavelength=10000):
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    timescale = _get_cached_timescale(x.shape[-1], x.device, max_wavelength)

    if positions.ndim == 1:
        positions = positions.unsqueeze(0)

    radians = (positions[..., None].to(torch.float32) / timescale)
    if x.ndim == 4:
        if x.shape[1] == positions.shape[1]:
            radians = radians.unsqueeze(2)
        elif x.shape[2] == positions.shape[1]:
            radians = radians.unsqueeze(1)
        else:
            radians = radians.unsqueeze(2)
    else:
        radians = radians.unsqueeze(-2)

    sin = torch.sin(radians).to(dtype)
    cos = torch.cos(radians).to(dtype)

    x1, x2 = x.chunk(2, dim=-1)
    res1 = x1 * cos - x2 * sin
    res2 = x2 * cos + x1 * sin
    return torch.cat([res1, res2], dim=-1)

smolvlm_expert_module.apply_rope = _safe_apply_rope

# Import PEFT modules
from peft import LoraConfig, get_peft_model, PeftModel

# Constants for LeRobot keys
ACTION = "action"
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

class SupervisedLoRADataset(Dataset):
    def __init__(self, data_dir, task_id, split="train", val_ratio=0.15, benchmark_name="libero_10"):
        self.file_paths = sorted(list(Path(data_dir).rglob("*.h5")) + list(Path(data_dir).rglob("*.hdf5")))
        
        # Get task name and language instruction from LIBERO benchmark
        try:
            from libero.libero.benchmark import get_benchmark_dict
            benchmark = get_benchmark_dict()[benchmark_name]()
            task = benchmark.get_task(task_id)
            self.task_name = task.name
            self.instruction = task.language
        except Exception as e:
            print(f"Warning: Could not get task name from benchmark: {e}")
            self.task_name = None
            self.instruction = ""

        # Filter file paths for this specific task
        filtered_paths = []
        for p in self.file_paths:
            p_str = p.as_posix()
            if f"_t{task_id}_" in p_str:
                filtered_paths.append(p)
            elif self.task_name is not None and (self.task_name in p.name or self.task_name.lower() in p.name.lower()):
                filtered_paths.append(p)
        self.file_paths = filtered_paths
        
        if not self.file_paths:
            raise ValueError(f"No demonstration files found for task {task_id} in {data_dir}")

        # Preload demonstration data into memory for zero HDF5 disk churn and max throughput
        self.samples = []
        for f_path in self.file_paths:
            try:
                with h5py.File(f_path, 'r') as h5:
                    if "data" not in h5:
                        continue
                    demo_keys = sorted(list(h5["data"].keys()))
                    temp_demos = []
                    for dk in demo_keys:
                        prefix = f"data/{dk}"
                        agent_rgb = h5[f"{prefix}/obs/agentview_rgb"][:]
                        # Flip H and W to align with SmolVLA standard orientation
                        agent_rgb = agent_rgb[:, ::-1, ::-1, :].copy()
                        
                        wrist_rgb = h5[f"{prefix}/obs/eye_in_hand_rgb"][:]
                        wrist_rgb = wrist_rgb[:, ::-1, ::-1, :].copy()
                        
                        ee_pos = h5[f"{prefix}/obs/ee_pos"][:]
                        ee_ori = h5[f"{prefix}/obs/ee_ori"][:]
                        gripper = h5[f"{prefix}/obs/gripper_states"][:]
                        states = np.concatenate([ee_pos, ee_ori, gripper], axis=-1).astype(np.float32)
                        
                        actions = h5[f"{prefix}/actions"][:].astype(np.float32)
                        temp_demos.append((agent_rgb, wrist_rgb, states, actions))
                        
                    n_demos = len(temp_demos)
                    val_count = max(1, int(n_demos * val_ratio))
                    train_count = n_demos - val_count
                    selected_demos = temp_demos[:train_count] if split == "train" else temp_demos[train_count:]
                    
                    for agent_rgb, wrist_rgb, states, actions in selected_demos:
                        n_steps = actions.shape[0]
                        for step_idx in range(n_steps):
                            chunk_len = min(50, n_steps - step_idx)
                            action_chunk = np.zeros((50, 7), dtype=np.float32)
                            actions_is_pad = np.zeros(50, dtype=bool)
                            action_chunk[:chunk_len] = actions[step_idx : step_idx + chunk_len]
                            actions_is_pad[chunk_len:] = True
                            if chunk_len < 50:
                                action_chunk[chunk_len:] = actions[-1]  # pad using last action
                                
                            self.samples.append({
                                "img_agent": agent_rgb[step_idx],
                                "img_wrist": wrist_rgb[step_idx],
                                "agent_pos": states[step_idx],
                                "action": action_chunk,
                                "actions_is_pad": actions_is_pad,
                                "task": self.instruction,
                            })
            except Exception as e:
                print(f"Failed to load {f_path}: {e}")

        print(f"[{split.upper()}] Supervised Dataset initialized with {len(self.samples)} samples for Task {task_id} (instruction: '{self.instruction}').")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def make_collate_fn(preprocessor):
    def collate_fn(batch):
        B = len(batch)
        img1_np = np.stack([item["img_agent"] for item in batch], axis=0)
        img2_np = np.stack([item["img_wrist"] for item in batch], axis=0)
        state_np = np.stack([item["agent_pos"] for item in batch], axis=0)
        
        # Images: convert [B, H, W, C] in [0, 255] to [B, C, H, W] in [0, 1] as float32 contiguous on CPU
        img_agent = torch.from_numpy(img1_np).permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0
        img_wrist = torch.from_numpy(img2_np).permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0
        state = torch.from_numpy(state_np).contiguous().to(dtype=torch.float32)
        
        action = torch.stack([torch.from_numpy(item["action"]) for item in batch], dim=0).contiguous().to(dtype=torch.float32)
        actions_is_pad = torch.stack([torch.from_numpy(item["actions_is_pad"]) for item in batch], dim=0).contiguous().to(dtype=torch.bool)
        tasks = [item["task"] for item in batch]
        
        raw_batch = {
            "observation.images.image": img_agent,
            "observation.images.image2": img_wrist,
            "observation.state": state,
            "task": tasks,
            "action": action,
        }
        
        processed_batch = preprocessor(raw_batch)
        processed_batch["actions_is_pad"] = actions_is_pad
        return processed_batch
    return collate_fn

def main():
    parser = argparse.ArgumentParser(description="Supervised Task-Specific LoRA Finetuning of SmolVLA")
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO task ID (0-9)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--data_dir", type=str, default="/home/swagat/libero_dataset/libero_10", help="Path to LIBERO-10 demonstrations directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints_lora", help="Root directory for saving checkpoints")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save VRAM")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick training step validation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate checkpoint folder
    task_checkpoint_dir = os.path.join(args.output_dir, f"task_{args.task_id}")
    os.makedirs(task_checkpoint_dir, exist_ok=True)
    config_path = os.path.join(task_checkpoint_dir, "config.json")
    
    # --- Check for resumption ---
    is_resuming = False
    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = wandb.util.generate_id()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            # Verify if LoRA adapter weights actually exist before resuming
            if os.path.exists(os.path.join(task_checkpoint_dir, "adapter_model.safetensors")) or \
               os.path.exists(os.path.join(task_checkpoint_dir, "adapter_model.bin")):
                is_resuming = True
                wandb_run_id = saved_config.get("wandb_run_id", wandb_run_id)
                start_epoch = saved_config.get("epoch", 0)
                best_val_loss = saved_config.get("best_val_loss", float("inf"))
                print(f"[Resume] Found checkpoint. Resuming from epoch {start_epoch} (wandb run id: {wandb_run_id})")
        except Exception as e:
            print(f"Warning: Could not parse saved config to resume: {e}. Starting fresh.")

    # Initialize wandb (with support for resume and fallback if deleted)
    try:
        wandb.init(
            project="smolvla_supervised_lora",
            name=f"task_{args.task_id}_lora_r{args.r}",
            id=wandb_run_id,
            resume="allow",
            config={
                "task_id": args.task_id,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lora_rank": args.r,
                "lora_alpha": args.alpha,
                "data_dir": args.data_dir
            }
        )
    except Exception as e:
        print(f"Warning: Could not resume wandb run {wandb_run_id}: {e}. Creating a new wandb run.")
        wandb_run_id = wandb.util.generate_id()
        wandb.init(
            project="smolvla_supervised_lora",
            name=f"task_{args.task_id}_lora_r{args.r}",
            id=wandb_run_id,
            config={
                "task_id": args.task_id,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lora_rank": args.r,
                "lora_alpha": args.alpha,
                "data_dir": args.data_dir
            }
        )

    # 1. Load Base Policy
    print("Loading baseline SmolVLA policy...")
    policy_name = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(policy_name).to(device)
    preprocessor, _ = make_pre_post_processors(policy.config, policy_name)

    # 2. Wrap lm_expert with Peft LoRA
    if is_resuming:
        print(f"Loading existing LoRA adapters from {task_checkpoint_dir}...")
        policy.model.vlm_with_expert.lm_expert = PeftModel.from_pretrained(
            policy.model.vlm_with_expert.lm_expert,
            task_checkpoint_dir,
            is_trainable=True
        ).to(device)
    else:
        print(f"Integrating LoRA (rank={args.r}, alpha={args.alpha}) on Action Expert...")
        peft_config = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        policy.model.vlm_with_expert.lm_expert = get_peft_model(policy.model.vlm_with_expert.lm_expert, peft_config).to(device)

    policy.model.vlm_with_expert.lm_expert.print_trainable_parameters()
    
    # Enable Gradient Checkpointing if requested
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing for Action Expert...")
        if hasattr(policy.model.vlm_with_expert.lm_expert, "gradient_checkpointing_enable"):
            policy.model.vlm_with_expert.lm_expert.gradient_checkpointing_enable()
        else:
            print("Warning: Action Expert does not support gradient checkpointing.")

    # Freeze the entire policy except LoRA layers
    for p in policy.parameters():
        p.requires_grad = False
        
    # Explicitly ensure only LoRA parameters have requires_grad=True
    trainable_params = []
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False
            
    optimizer = optim.AdamW(trainable_params, lr=args.lr)

    # Save initial config.json with all passed arguments (including default parameters)
    if not is_resuming:
        initial_config = vars(args).copy()
        initial_config["wandb_run_id"] = wandb_run_id
        initial_config["epoch"] = 0
        initial_config["best_val_loss"] = float("inf")
        with open(config_path, "w") as f:
            json.dump(initial_config, f, indent=4)

    # 3. Create Dataset and DataLoader
    print("Loading datasets...")
    train_dataset = SupervisedLoRADataset(args.data_dir, task_id=args.task_id, split="train")
    val_dataset = SupervisedLoRADataset(args.data_dir, task_id=args.task_id, split="val")
    
    collate_fn = make_collate_fn(preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Keep base frozen backbone in eval mode; only train the LoRA Action Expert
        policy.eval()
        policy.model.vlm_with_expert.vlm.eval()
        policy.model.vlm_with_expert.lm_expert.train()
        
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        train_pbar = tqdm(
            train_loader,
            desc=f"Task {args.task_id} | Epoch {epoch+1:02d}/{args.epochs:02d} [Train]",
            dynamic_ncols=True,
            mininterval=1.0,
            unit="batch"
        )
        for step, batch in enumerate(train_pbar):
            # Transfer batch tensors to GPU inside main loop (blocking transfer to guarantee host page pin safety)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # Forward pass through VLA policy to calculate flow loss
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, loss_dict = policy(batch)
                loss = loss / args.grad_accum_steps
            
            loss.backward()
            
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            loss_val = loss.item() * args.grad_accum_steps
            train_losses.append(loss_val)
            train_pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg_loss": f"{np.mean(train_losses):.4f}"})
            
            # Synchronize CUDA stream at every step to prevent autograd thread race conditions with CUDA memory allocator
            torch.cuda.synchronize()
            
            if (step + 1) % 100 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            if args.dry_run:
                print(f"Dry-run step loss: {train_losses[-1]}")
                break
                
        avg_train_loss = np.mean(train_losses)
        
        # Validation pass
        policy.eval()
        policy.model.vlm_with_expert.lm_expert.eval()
        val_losses = []
        val_pbar = tqdm(
            val_loader,
            desc=f"Task {args.task_id} | Epoch {epoch+1:02d}/{args.epochs:02d} [Val]",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, loss_dict = policy(batch)
                val_losses.append(loss.item())
                val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
                torch.cuda.synchronize()
                if args.dry_run:
                    break
                    
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch + 1})
        
        # Save checkpoint every epoch for seamless interruption resumption
        if not args.dry_run:
            # 1. Always save latest checkpoint weights and update resume state
            policy.model.vlm_with_expert.lm_expert.save_pretrained(task_checkpoint_dir)
            readme_path = os.path.join(task_checkpoint_dir, "README.md")
            if os.path.exists(readme_path):
                os.remove(readme_path)

            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"★ New best validation loss: {best_val_loss:.5f}")

            # Update config.json with current epoch and best loss
            with open(config_path, "w") as f:
                updated_config = vars(args).copy()
                updated_config["wandb_run_id"] = wandb_run_id
                updated_config["epoch"] = epoch + 1
                updated_config["best_val_loss"] = best_val_loss
                json.dump(updated_config, f, indent=4)
            print(f"Saved LoRA checkpoint (epoch {epoch+1}) to {task_checkpoint_dir}")
            
            # Clean up memory caches at epoch boundary
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        if args.dry_run:
            print("Dry-run completed successfully.")
            break
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    if hasattr(val_dataset, "close"):
        val_dataset.close()
    wandb.finish()

if __name__ == "__main__":
    main()
