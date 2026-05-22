import torch
import sys
from train_hybrid_diffusion import plot_diffusion_trajectories

def get_base_policy():
    from hybrid_diffusion_agent import HybridFrozenBrainDiffusionHands
    model = HybridFrozenBrainDiffusionHands(
        base_policy_path="HuggingFaceVLA/smolvla_libero",
        action_dim=7, chunk_size=1, device="cuda"
    )
    return model

if __name__ == "__main__":
    pass
