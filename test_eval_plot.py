import torch
import sys
import os
import argparse
import random
import numpy as np

from hybrid_diffusion_agent import HybridFrozenBrainDiffusionHands
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from train_hybrid_diffusion import plot_diffusion_trajectories
from torch.utils.data import DataLoader

def run_debug():
    device = torch.device('cuda')
    # Load dataset
    dataset = LeRobotDataset("lerobot/libero_10", root="~/.cache/huggingface/lerobot/libero_10")
    
    # We must patch delts_timestamps if required, but let's just use dataloader
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=8,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    
    print("[DEBUG] Initialized Dataloader.")
    
    model = HybridFrozenBrainDiffusionHands(
        base_policy_path="HuggingFaceVLA/smolvla_libero",
        action_dim=7, chunk_size=16,
        device=device
    )
    
    print("[DEBUG] Calling plot_diffusion_trajectories...")
    # This will trigger the plotting
    plot_diffusion_trajectories(model, batch, device, 1)

if __name__ == "__main__":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"
    run_debug()
