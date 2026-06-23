#!/usr/bin/env python3
"""
Simple script to check if Phase 1 data collection worked properly
"""

import os
import h5py
import json
from pathlib import Path

def check_data_files(data_dir="Gated_Residual_strategy/data"):
    """Check if data files exist and are readable"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory {data_dir} does not exist")
        return False
    
    # Check for HDF5 files
    h5_files = list(data_path.rglob("failure_dataset_task*_seed*.h5"))
    log_files = list(data_path.rglob("logs_task*_seed*.json"))
    
    print(f"Found {len(h5_files)} HDF5 files and {len(log_files)} log files")
    
    if not h5_files:
        print("❌ No HDF5 files found")
        return False
    
    # Check first few files
    print("\nChecking first few files:")
    for i, h5_file in enumerate(h5_files[:3]):
        try:
            with h5py.File(h5_file, 'r') as f:
                print(f"  File {h5_file.name}:")
                print(f"    Keys: {list(f.keys())}")
                if 'labels' in f:
                    labels = f['labels'][:]
                    print(f"    Labels shape: {labels.shape}")
                    print(f"    Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
                if 'actions' in f:
                    actions = f['actions'][:]
                    print(f"    Actions shape: {actions.shape}")
                if 'observations' in f:
                    obs = f['observations']
                    print(f"    Observation keys: {list(obs.keys())}")
                    for key in obs.keys():
                        if hasattr(obs[key], 'shape'):
                            print(f"      {key}: {obs[key].shape}")
        except Exception as e:
            print(f"  ❌ Error reading {h5_file}: {e}")
    
    return True

def check_logs(log_files):
    """Check log files for basic information"""
    print("\nChecking log files:")
    for i, log_file in enumerate(log_files[:3]):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                print(f"  File {log_file.name}:")
                print(f"    Number of episodes: {len(data)}")
                if data:
                    first_episode = data[0]
                    print(f"    First episode keys: {list(first_episode.keys())}")
        except Exception as e:
            print(f"  ❌ Error reading {log_file}: {e}")

if __name__ == "__main__":
    # Import numpy here to avoid issues if not needed
    import numpy as np
    check_data_files()