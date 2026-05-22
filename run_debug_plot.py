import torch
import sys
from train_hybrid_diffusion import parse_args, train
# Hack sys.argv to force eval frequency to match 1
sys.argv = ['train_hybrid_diffusion.py', '--epochs', '2', '--vis_freq', '1', '--dataset_repo_id', 'lerobot/libero_10']

# We just want to run plot_diffusion_trajectories, but training loop has dataset load.
