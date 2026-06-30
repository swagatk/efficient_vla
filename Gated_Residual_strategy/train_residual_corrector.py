#!/usr/bin/env python3
"""
Phase 3: Train the Gated Residual Corrector

This script trains a lightweight multimodal neural network to predict successful 
control actions from observation streams, trained offline on successful segments
(nominal data, label=0) of the collected Phase 1 trajectories.
"""

import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import wandb
from robosuite.utils.transform_utils import quat2axisangle

class CorrectorDataset(Dataset):
    def __init__(self, data_dir, split="train", val_ratio=0.15, target_label=0, train_mode="absolute"):
        """
        Lazy-loading HDF5 dataset for Phase 3 Gated Residual Corrector training.
        Filters for steps matching target_label (default 0: successful trajectories).
        """
        self.file_paths = sorted(list(Path(data_dir).rglob("*.h5")))
        if not self.file_paths:
            raise ValueError(f"No .h5 files found in {data_dir}")
            
        self.index_map = []
        self.target_label = target_label
        self.train_mode = train_mode
        
        for f_idx, f_path in enumerate(self.file_paths):
            try:
                with h5py.File(f_path, 'r') as h5:
                    if "labels" not in h5:
                        continue
                    labels = h5["labels"][:]
                    n = len(labels)
                    
                    # Filter indices matching the target label
                    valid_indices = [idx for idx in range(n) if labels[idx] == self.target_label]
                    n_valid = len(valid_indices)
                    if n_valid == 0:
                        continue
                        
                    val_size = int(n_valid * val_ratio)
                    train_size = n_valid - val_size
                    
                    selected_indices = valid_indices[:train_size] if split == "train" else valid_indices[train_size:]
                    
                    for i in selected_indices:
                        self.index_map.append((f_idx, i))
            except Exception as e:
                print(f"Failed to read {f_path}: {e}")
                
        self.open_files = {}
        print(f"[{split.upper()}] Dataset initialized with {len(self.index_map)} samples (target_label={target_label}, train_mode={train_mode}).")

    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, idx):
        f_idx, i = self.index_map[idx]
        
        if f_idx not in self.open_files:
            self.open_files[f_idx] = h5py.File(self.file_paths[f_idx], 'r')
            
        h5 = self.open_files[f_idx]
        
        # Read and prepare image tensors
        raw_img1 = h5["observations"]["agentview_image"][i]
        img1_np = raw_img1[::-1, ::-1, :].copy()
        img1 = torch.from_numpy(img1_np).float().permute(2, 0, 1) / 255.0
        
        raw_img2 = h5["observations"]["robot0_eye_in_hand_image"][i]
        img2_np = raw_img2[::-1, ::-1, :].copy()
        img2 = torch.from_numpy(img2_np).float().permute(2, 0, 1) / 255.0
        
        # Read and prepare proprioceptive state
        eef_pos = h5["observations"]["robot0_eef_pos"][i]
        eef_quat = h5["observations"]["robot0_eef_quat"][i]
        gripper_qpos = h5["observations"]["robot0_gripper_qpos"][i]
        
        state_np = np.concatenate([eef_pos, quat2axisangle(eef_quat), gripper_qpos])
        state = torch.from_numpy(state_np).float()
        
        # Read and prepare action target
        action_np = h5["actions"][i]
        
        if self.train_mode == "delta":
            if "observations/base_actions" in h5:
                base_action_np = h5["observations/base_actions"][i]
            elif "observations" in h5 and "base_actions" in h5["observations"]:
                base_action_np = h5["observations"]["base_actions"][i]
            else:
                base_action_np = action_np
            target_np = action_np - base_action_np
        else:
            target_np = action_np
            
        action = torch.from_numpy(target_np).float()
        
        return img1, img2, state, action

class LightweightResidualCorrector(nn.Module):
    def __init__(self, state_dim=8, action_dim=7):
        """
        Multimodal architecture outputting continuous actions for correction.
        """
        super().__init__()
        
        def create_cnn():
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            
        self.img1_cnn = create_cnn()
        self.img2_cnn = create_cnn()
        
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion layer: 64 (img1) + 64 (img2) + 32 (state) = 160 input features
        self.fusion = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim)
        )

    def forward(self, img1, img2, state):
        x1 = self.img1_cnn(img1)
        x2 = self.img2_cnn(img2)
        xs = self.state_mlp(state)
        x = torch.cat([x1, x2, xs], dim=1)
        return self.fusion(x)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img1, img2, state, action in val_loader:
            img1, img2, state, action = img1.to(device), img2.to(device), state.to(device), action.to(device)
            pred_action = model(img1, img2, state)
            loss = criterion(pred_action, action)
            total_loss += loss.item()
            
    return {
        "val/loss": total_loss / len(val_loader)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Gated_Residual_strategy/data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_label", type=int, default=0, help="Label to train on (0: success trajectories)")
    parser.add_argument("--train_mode", type=str, choices=["absolute", "delta"], default="absolute", help="Training mode (absolute actions or delta actions)")
    parser.add_argument("--wandb_project", type=str, default="gated_residual_phase3")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Dataset
    test_dataset = CorrectorDataset(args.data_dir, split="train", val_ratio=0.15, target_label=args.target_label, train_mode=args.train_mode)
    if len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    _, _, sample_state, sample_action = test_dataset[0]
    state_dim = sample_state.shape[0]
    action_dim = sample_action.shape[0]
    print(f"Inferred Proprioceptive State Dimension: {state_dim}, Action Dimension: {action_dim}")

    train_dataset = test_dataset
    val_dataset = CorrectorDataset(args.data_dir, split="val", val_ratio=0.15, target_label=args.target_label, train_mode=args.train_mode)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightResidualCorrector(state_dim=state_dim, action_dim=action_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume=args.wandb_resume,
        config=vars(args)
    )
    
    # Write WandB run ID for progress tracking
    run_id_file = os.environ.get("WANDB_RUN_ID_FILE")
    if run_id_file and wandb.run is not None:
        with open(run_id_file, "w") as f:
            f.write(wandb.run.id)
            
    best_val_loss = float("inf")
    val_metrics = {}
    
    print(f"Starting training on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (img1, img2, state, action) in enumerate(train_loader):
            img1, img2, state, action = img1.to(device), img2.to(device), state.to(device), action.to(device)
            
            optimizer.zero_grad()
            pred_action = model(img1, img2, state)
            loss = criterion(pred_action, action)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Train MSE Loss: {loss.item():.6f}")
                wandb.log({"train/batch_loss": loss.item()})
                
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_metrics["train/epoch_loss"] = total_loss / len(train_loader)
        val_metrics["epoch"] = epoch
        
        print(f"Epoch {epoch} Results: Val MSE Loss: {val_metrics['val/loss']:.6f} | Train Loss: {val_metrics['train/epoch_loss']:.6f}")
        wandb.log(val_metrics)
        
        # Save checkpoints
        checkpoint_path = os.path.join(args.output_dir, "latest_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best model (Val MSE: {best_val_loss:.6f})")
            
    # Save training metrics for shell orchestrator summary
    if val_metrics:
        metrics_file = os.path.join(args.output_dir, "training_metrics.csv")
        with open(metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=val_metrics.keys())
            writer.writeheader()
            writer.writerow(val_metrics)
        print(f"Saved training metrics to {metrics_file}")
        
    wandb.finish()

if __name__ == "__main__":
    main()
