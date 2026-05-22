import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from hybrid_diffusion_agent import HybridFrozenBrainDiffusionHands
from torch.utils.data import DataLoader
from train_hybrid_diffusion import plot_diffusion_trajectories

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LeRobotDataset("lerobot/libero_10", root="~/.cache/huggingface/lerobot/libero_10")
    
    # Needs to apply transforms or delta_timestamps normally...?
    # Just grab vis_batch the way train_hybrid_diffusion.py does
    from lerobot.common.datasets.lerobot_dataset import MultiEpochSampler
    dataloader = DataLoader(dataset, num_workers=0, batch_size=8, shuffle=False)
    vis_batch = next(iter(dataloader))
    
    model = HybridFrozenBrainDiffusionHands(
        base_policy_path="HuggingFaceVLA/smolvla_libero",
        action_dim=7, chunk_size=16,
        device=device
    )
    # The device printed by train script
    plot_diffusion_trajectories(model, vis_batch, device, 10, num_samples=4)

