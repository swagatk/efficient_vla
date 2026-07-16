"""
Phase 2: Train the Failure-Risk Gate

This script trains a lightweight binary classifier on the failure dataset collected in Phase 1.
It predicts the probability of failure based on the agent's observation streams.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from robosuite.utils.transform_utils import quat2axisangle

class FailureDataset(Dataset):
    def __init__(self, data_dir, split="train", val_ratio=0.15, task_id=None):
        """
        Lazy-loading HDF5 dataset for Phase 1 failure trajectories.
        """
        self.file_paths = sorted(list(Path(data_dir).rglob("*.h5")) + list(Path(data_dir).rglob("*.hdf5")))
        if task_id is not None:
            self.file_paths = [p for p in self.file_paths if f"_t{task_id}_" in p.as_posix()]
        if not self.file_paths:
            raise ValueError(f"No .h5 or .hdf5 files found in {data_dir}")
            
        self.index_map = []
        self.num_pos = 0
        self.num_neg = 0
        
        for f_idx, f_path in enumerate(self.file_paths):
            try:
                with h5py.File(f_path, 'r') as h5:
                    if "labels" not in h5:
                        continue
                    labels = h5["labels"][:]
                    n = len(labels)
                    
                    val_size = int(n * val_ratio)
                    train_size = n - val_size
                    
                    start_idx = 0 if split == "train" else train_size
                    end_idx = train_size if split == "train" else n
                    
                    split_labels = labels[start_idx:end_idx]
                    self.num_pos += int(np.sum(split_labels == 1))
                    self.num_neg += int(np.sum(split_labels == 0))
                    
                    for i in range(start_idx, end_idx):
                        self.index_map.append((f_idx, i))
            except Exception as e:
                print(f"Failed to read {f_path}: {e}")
                
        self.open_files = {}
        print(f"[{split.upper()}] Dataset initialized with {len(self.index_map)} samples (neg: {self.num_neg}, pos: {self.num_pos}).")

    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, idx):
        f_idx, i = self.index_map[idx]
        
        if f_idx not in self.open_files:
            self.open_files[f_idx] = h5py.File(self.file_paths[f_idx], 'r')
            
        h5 = self.open_files[f_idx]
        
        # Read and prepare tensors from native LIBERO observation keys
        raw_img1 = h5["observations"]["agentview_image"][i]
        img1_np = raw_img1[::-1, ::-1, :].copy()
        img1 = torch.from_numpy(img1_np).float().permute(2, 0, 1) / 255.0
        
        raw_img2 = h5["observations"]["robot0_eye_in_hand_image"][i]
        img2_np = raw_img2[::-1, ::-1, :].copy()
        img2 = torch.from_numpy(img2_np).float().permute(2, 0, 1) / 255.0
        
        eef_pos = h5["observations"]["robot0_eef_pos"][i]
        eef_quat = h5["observations"]["robot0_eef_quat"][i]
        gripper_qpos = h5["observations"]["robot0_gripper_qpos"][i]
        
        state_np = np.concatenate([eef_pos, quat2axisangle(eef_quat), gripper_qpos])
        state = torch.from_numpy(state_np).float()
        
        label = torch.tensor(h5["labels"][i], dtype=torch.float32).unsqueeze(0)
        
        return img1, img2, state, label

class LightweightFailureGate(nn.Module):
    def __init__(self, state_dim=8):
        """
        Multimodal MLP that processes precomputed SigLIP visual embeddings and proprioceptive state.
        """
        super().__init__()
        
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # 768 (img1 SigLIP pooler) + 768 (img2 SigLIP pooler) + 32 (state) = 1568
        self.fusion = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Outputs logits (unnormalized log probabilities)
        )

    def forward(self, feat1, feat2, state):
        xs = self.state_mlp(state)
        x = torch.cat([feat1, feat2, xs], dim=1)
        return self.fusion(x)

def evaluate(model, val_loader, criterion, device, vision_tower):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for img1, img2, state, label in val_loader:
            img1, img2, state, label = img1.to(device), img2.to(device), state.to(device), label.to(device)
            
            # 1. Resize and normalize images dynamically
            img1_224 = torch.nn.functional.interpolate(img1, size=(224, 224), mode="bilinear", align_corners=False)
            img1_norm = (img1_224 - 0.5) / 0.5
            
            img2_224 = torch.nn.functional.interpolate(img2, size=(224, 224), mode="bilinear", align_corners=False)
            img2_norm = (img2_224 - 0.5) / 0.5
            
            # 2. Extract embeddings
            v_dtype = vision_tower.dtype
            feat1 = vision_tower(img1_norm.to(dtype=v_dtype)).last_hidden_state.mean(dim=1)
            feat2 = vision_tower(img2_norm.to(dtype=v_dtype)).last_hidden_state.mean(dim=1)
            
            logits = model(feat1.to(torch.float32), feat2.to(torch.float32), state)
            loss = criterion(logits, label)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            total_loss += loss.item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    metrics = {
        "val/loss": total_loss / len(val_loader),
        "val/accuracy": accuracy_score(all_labels, all_preds),
        "val/precision": precision_score(all_labels, all_preds, zero_division=0),
        "val/recall": recall_score(all_labels, all_preds, zero_division=0),
    }
    
    # AUC requires varied labels
    try:
        metrics["val/auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["val/auc"] = 0.0
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Gated_Residual_strategy/data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None, help="Train specifically on this task ID (0-9)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="gated_residual_phase2")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load baseline SmolVLA policy for vision features
    print("Loading base SmolVLA policy for frozen vision tower...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy_name = "HuggingFaceVLA/smolvla_libero"
    base_policy = SmolVLAPolicy.from_pretrained(policy_name).to(device)
    vision_tower = base_policy.model.vlm_with_expert.get_vlm_model().vision_model
    vision_tower.eval()
    for p in vision_tower.parameters():
        p.requires_grad = False
    
    # Determine State Dimension dynamically from the first file
    test_dataset = FailureDataset(args.data_dir, split="train", val_ratio=0.15, task_id=args.task_id)
    if len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    _, _, sample_state, _ = test_dataset[0]
    state_dim = sample_state.shape[0]
    print(f"Inferred Proprioceptive State Dimension: {state_dim}")

    train_dataset = test_dataset
    val_dataset = FailureDataset(args.data_dir, split="val", val_ratio=0.15, task_id=args.task_id)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = LightweightFailureGate(state_dim=state_dim).to(device)
    
    n_neg = train_dataset.num_neg
    n_pos = train_dataset.num_pos
    if n_pos > 0:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"BCE pos_weight set to: {pos_weight.item():.4f} (neg: {n_neg}, pos: {n_pos})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Warning: No positive class samples found in train dataset. Using unweighted BCE loss.")
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume=args.wandb_resume,
        config=vars(args)
    )
    
    # Ensure Bash script receives the run ID
    run_id_file = os.environ.get("WANDB_RUN_ID_FILE")
    if run_id_file and wandb.run is not None:
        with open(run_id_file, "w") as f:
            f.write(wandb.run.id)

    best_val_auc = 0.0
    best_metrics = {}
    val_metrics = {}
    
    print(f"Starting training on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (img1, img2, state, label) in enumerate(train_loader):
            img1, img2, state, label = img1.to(device), img2.to(device), state.to(device), label.to(device)
            
            # 1. Resize and normalize images dynamically
            img1_224 = torch.nn.functional.interpolate(img1, size=(224, 224), mode="bilinear", align_corners=False)
            img1_norm = (img1_224 - 0.5) / 0.5
            
            img2_224 = torch.nn.functional.interpolate(img2, size=(224, 224), mode="bilinear", align_corners=False)
            img2_norm = (img2_224 - 0.5) / 0.5
            
            # 2. Extract embeddings
            with torch.no_grad():
                v_dtype = vision_tower.dtype
                feat1 = vision_tower(img1_norm.to(dtype=v_dtype)).last_hidden_state.mean(dim=1)
                feat2 = vision_tower(img2_norm.to(dtype=v_dtype)).last_hidden_state.mean(dim=1)
            
            optimizer.zero_grad()
            logits = model(feat1.to(torch.float32), feat2.to(torch.float32), state)
            loss = criterion(logits, label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f}")
                wandb.log({"train/batch_loss": loss.item()})
                
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device, vision_tower)
        val_metrics["train/epoch_loss"] = total_loss / len(train_loader)
        val_metrics["epoch"] = epoch
        
        print(f"Epoch {epoch} Results: Val Loss: {val_metrics['val/loss']:.4f} | Val AUC: {val_metrics['val/auc']:.4f} | Val Acc: {val_metrics['val/accuracy']:.4f}")
        wandb.log(val_metrics)
        
        # Checkpoint
        checkpoint_path = os.path.join(args.output_dir, "latest_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        if not best_metrics or val_metrics["val/auc"] > best_val_auc:
            best_val_auc = val_metrics["val/auc"]
            best_metrics = val_metrics
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best model (AUC: {best_val_auc:.4f})")
            
    # Save final epoch metrics to CSV for bash script aggregation
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