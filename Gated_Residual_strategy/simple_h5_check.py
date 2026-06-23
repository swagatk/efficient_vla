#!/usr/bin/env python3
"""
Very simple check to see if we can read HDF5 files
"""

# Try to import required modules
try:
    import h5py
    import numpy as np
    print("✅ Successfully imported h5py and numpy")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Try to read a file
filepath = "Gated_Residual_strategy/outputs/run_20260603_214930/unit_collect_t0_s0/failure_dataset_task0_seed0.h5"

print(f"Attempting to read: {filepath}")

try:
    with h5py.File(filepath, 'r') as f:
        print("✅ Successfully opened HDF5 file")
        print(f"Root keys: {list(f.keys())}")
        
        if 'labels' in f:
            labels = f['labels'][:]
            print(f"Labels shape: {labels.shape}")
            print(f"Labels dtype: {labels.dtype}")
            unique, counts = np.unique(labels, return_counts=True)
            print(f"Unique labels: {unique}")
            print(f"Label counts: {counts}")
            print(f"Failure rate: {counts[1]/sum(counts):.2%}" if len(counts) > 1 else "Only one class")
        else:
            print("No labels dataset found")
            
        if 'actions' in f:
            actions = f['actions'][:]
            print(f"Actions shape: {actions.shape}")
            
        if 'observations' in f:
            obs = f['observations']
            print(f"Observation keys: {list(obs.keys())}")
            for key in list(obs.keys())[:3]:  # First 3 keys
                if key in obs:
                    data = obs[key]
                    print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                    
    print("✅ File analysis completed successfully")
    
except FileNotFoundError:
    print(f"❌ File not found: {filepath}")
except Exception as e:
    print(f"❌ Error reading file: {e}")
    import traceback
    traceback.print_exc()