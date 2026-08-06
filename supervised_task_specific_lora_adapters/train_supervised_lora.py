#!/usr/bin/env python3
import os
os.environ["WANDB_SYSTEM_MONITOR"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import argparse
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import wandb
from datetime import datetime

# Import LeRobot/SmolVLA modules
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors

# Import PEFT modules
from peft import LoraConfig, get_peft_model, PeftModel

# Import power inhibitor
from linux_inhibit import LinuxInhibit

# Constants for LeRobot keys
ACTION = "action"
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language_tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language_tokens_attention_mask"

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

        self.index_map = []
        
        # Build dataset step index map
        for f_idx, f_path in enumerate(self.file_paths):
            try:
                with h5py.File(f_path, 'r') as h5:
                    if "data" in h5:
                        demo_keys = sorted(list(h5["data"].keys()))
                        temp_indices = []
                        for dk in demo_keys:
                            n_steps = h5[f"data/{dk}/actions"].shape[0]
                            for step_idx in range(n_steps):
                                temp_indices.append((dk, step_idx, n_steps))
                                
                        n_valid = len(temp_indices)
                        val_size = int(n_valid * val_ratio)
                        train_size = n_valid - val_size
                        selected_indices = temp_indices[:train_size] if split == "train" else temp_indices[train_size:]
                        for dk, step_idx, total_steps in selected_indices:
                            self.index_map.append((f_idx, dk, step_idx, total_steps))
            except Exception as e:
                print(f"Failed to index {f_path}: {e}")
                
        self.open_files = {}
        print(f"[{split.upper()}] Supervised Dataset initialized with {len(self.index_map)} samples for Task {task_id} (instruction: '{self.instruction}').")

    def close(self):
        for f_idx, f_handle in list(self.open_files.items()):
            try:
                f_handle.close()
            except Exception:
                pass
        self.open_files.clear()

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        f_idx, demo_key, i, total_steps = self.index_map[idx]
        
        if f_idx not in self.open_files:
            self.open_files[f_idx] = h5py.File(self.file_paths[f_idx], 'r')
            
        h5 = self.open_files[f_idx]
        demo_group = h5[f"data/{demo_key}"]
        
        # 1. Read agentview and eye-in-hand cameras (flipped horizontally and vertically to match standard alignment)
        raw_img1 = demo_group["obs"]["agentview_rgb"][i]
        img1_np = raw_img1[::-1, ::-1, :].copy()
        
        raw_img2 = demo_group["obs"]["eye_in_hand_rgb"][i]
        img2_np = raw_img2[::-1, ::-1, :].copy()
        
        # 2. Read proprioceptive state: pos (3) + ori (3 axis-angle) + gripper (2)
        ee_pos = demo_group["obs"]["ee_pos"][i]
        ee_ori = demo_group["obs"]["ee_ori"][i]
        gripper_states = demo_group["obs"]["gripper_states"][i]
        state_np = np.concatenate([ee_pos, ee_ori, gripper_states]).astype(np.float32)
        
        # 3. Read action chunk (size 50 matching SmolVLA's flow chunk size)
        action_chunk = np.zeros((50, 7), dtype=np.float32)
        actions_is_pad = np.zeros(50, dtype=bool)
        
        chunk_len = min(50, total_steps - i)
        action_chunk[:chunk_len] = demo_group["actions"][i : i + chunk_len]
        actions_is_pad[chunk_len:] = True
        if chunk_len < 50:
            action_chunk[chunk_len:] = demo_group["actions"][total_steps - 1]  # pad using last action
            
        return {
            "img_agent": img1_np,
            "img_wrist": img2_np,
            "agent_pos": state_np,
            "action": action_chunk,
            "actions_is_pad": actions_is_pad,
            "task": self.instruction
        }

def make_collate_fn(preprocessor, device):
    def collate_fn(batch):
        batch_img_agent = np.stack([item["img_agent"] for item in batch], axis=0)
        batch_img_wrist = np.stack([item["img_wrist"] for item in batch], axis=0)
        batch_agent_pos = np.stack([item["agent_pos"] for item in batch], axis=0)
        
        batch_action = torch.stack([torch.from_numpy(item["action"]) for item in batch], dim=0).to(device, dtype=torch.bfloat16)
        batch_actions_is_pad = torch.stack([torch.from_numpy(item["actions_is_pad"]) for item in batch], dim=0).to(device)
        tasks = [item["task"] for item in batch]
        
        # Put observations into LeRobot preprocess observation format (using numpy arrays)
        raw_obs_batch = {
            "pixels": {
                "image": batch_img_agent,
                "image2": batch_img_wrist,
            },
            "agent_pos": batch_agent_pos,
        }
        
        policy_obs = preprocess_observation(raw_obs_batch)
        policy_obs["task"] = tasks
        
        # Preprocess using policy text/image pipeline
        batch_obs = preprocessor(policy_obs)
        
        # Move all preprocessed tensors to the target device and cast floats to bfloat16
        batch_obs = {
            k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device))
            if isinstance(v, torch.Tensor) else v
            for k, v in batch_obs.items()
        }
        
        # Inject target action chunk and padding masks
        batch_obs["action"] = batch_action
        batch_obs["actions_is_pad"] = batch_actions_is_pad
        
        return batch_obs
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

    # Initialize wandb (with support for resume)
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

    # 1. Load Base Policy
    print("Loading baseline SmolVLA policy...")
    policy_name = "HuggingFaceVLA/smolvla_libero"
    policy = SmolVLAPolicy.from_pretrained(policy_name).to(torch.bfloat16).to(device)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, policy_name)

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
    
    collate_fn = make_collate_fn(preprocessor, device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        policy.train()
        
        optimizer.zero_grad()
        train_losses = []
        for step, batch in enumerate(train_loader):
            # Forward pass through VLA policy to calculate flow loss
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, loss_dict = policy(batch)
                loss = loss / args.grad_accum_steps
            
            loss.backward()
            
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * args.grad_accum_steps)
            del loss, loss_dict, batch
            
            if args.dry_run:
                print(f"Dry-run step loss: {train_losses[-1]}")
                break
                
        avg_train_loss = np.mean(train_losses)
        train_dataset.close()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Validation pass
        policy.eval()
        policy.model.vlm_with_expert.lm_expert.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, loss_dict = policy(batch)
                val_losses.append(loss.item())
                del loss, loss_dict
                if args.dry_run:
                    break
                    
        avg_val_loss = np.mean(val_losses)
        val_dataset.close()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch + 1})
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss and not args.dry_run:
            best_val_loss = avg_val_loss
            # Save the small PEFT model adapters weights
            policy.model.vlm_with_expert.lm_expert.save_pretrained(task_checkpoint_dir)
            readme_path = os.path.join(task_checkpoint_dir, "README.md")
            if os.path.exists(readme_path):
                os.remove(readme_path)
            
            # Save config.json for training parameters
            with open(config_path, "w") as f:
                updated_config = vars(args).copy()
                updated_config["wandb_run_id"] = wandb_run_id
                updated_config["epoch"] = epoch + 1
                updated_config["best_val_loss"] = best_val_loss
                json.dump(updated_config, f, indent=4)
            print(f"Saved best LoRA checkpoint to {task_checkpoint_dir}")
            

            
        if args.dry_run:
            print("Dry-run completed successfully.")
            break
    train_dataset.close()
    val_dataset.close()
    wandb.finish()

if __name__ == "__main__":
    # Wrap in power profile inhibitor and signal handler context manager
    with LinuxInhibit("Supervised LoRA Training"):
        main()
