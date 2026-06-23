#!/usr/bin/env python3
"""
Examine the structure of HDF5 files to understand the data format
"""

import h5py
import numpy as np

def examine_h5_file(filepath):
    """Examine a single HDF5 file structure"""
    print(f"Examining {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            
            # Examine metadata
            if 'metadata' in f:
                print(f"Metadata attributes: {dict(f['metadata'].attrs)}")
            
            # Examine labels
            if 'labels' in f:
                labels = f['labels'][:]
                print(f"Labels shape: {labels.shape}")
                print(f"Labels dtype: {labels.dtype}")
                unique, counts = np.unique(labels, return_counts=True)
                print(f"Label distribution: {dict(zip(unique, counts))}")
            
            # Examine actions
            if 'actions' in f:
                actions = f['actions'][:]
                print(f"Actions shape: {actions.shape}")
                print(f"Actions dtype: {actions.dtype}")
                print(f"Actions range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
            
            # Examine observations
            if 'observations' in f:
                obs_group = f['observations']
                print(f"Observation keys: {list(obs_group.keys())}")
                
                for key in obs_group.keys():
                    obs_data = obs_group[key]
                    print(f"  {key}: shape={obs_data.shape}, dtype={obs_data.dtype}")
                    
                    # Show some sample values for first few observations
                    if obs_data.shape[0] > 0:
                        sample_idx = min(3, obs_data.shape[0])  # First 3 samples
                        print(f"    Sample values (first {sample_idx} entries):")
                        for i in range(sample_idx):
                            if len(obs_data[i].shape) == 0:  # Scalar
                                print(f"      [{i}]: {obs_data[i]}")
                            elif len(obs_data[i].shape) == 1 and obs_data[i].shape[0] <= 10:  # Small vector
                                print(f"      [{i}]: {obs_data[i][:5]}...")  # First 5 elements
                            else:
                                print(f"      [{i}]: shape={obs_data[i].shape}")
            
            # Check a few sample entries
            if 'labels' in f and 'actions' in f and 'observations' in f:
                print(f"\nSample entries (first 3):")
                labels = f['labels'][:]
                for i in range(min(3, len(labels))):
                    print(f"  Entry {i}: label={labels[i]}, action={f['actions'][i][:3]}...")  # First 3 action dims
                    
    except Exception as e:
        print(f"Error examining {filepath}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Examine one of the HDF5 files
    filepath = "outputs/run_20260603_214930/unit_collect_t0_s0/failure_dataset_task0_seed0.h5"
    examine_h5_file(filepath)