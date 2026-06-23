#!/usr/bin/env python3
"""
Simple script to check if we can read the Phase 1 data
"""

import h5py
import numpy as np
import os

def check_single_file(filepath):
    """Check if we can read a single HDF5 file"""
    print(f"Checking {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  File does not exist")
        return False
        
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"  Successfully opened file")
            print(f"  Keys: {list(f.keys())}")
            
            if 'labels' in f:
                labels = f['labels'][:]
                print(f"  Labels shape: {labels.shape}")
                unique, counts = np.unique(labels, return_counts=True)
                print(f"  Label distribution: {dict(zip(unique, counts))}")
                return True
            else:
                print(f"  No labels found")
                return False
                
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False

def main():
    # Try to check a few files
    test_files = [
        "outputs/run_20260603_214930/unit_collect_t0_s0/failure_dataset_task0_seed0.h5",
        "outputs/run_20260603_214930/unit_collect_t0_s1/failure_dataset_task0_seed1.h5",
        "outputs/run_20260603_214930/unit_collect_t1_s0/failure_dataset_task1_seed0.h5"
    ]
    
    print("Checking Phase 1 data files...")
    successful_reads = 0
    
    for test_file in test_files:
        full_path = os.path.join("Gated_Residual_strategy", test_file)
        if check_single_file(full_path):
            successful_reads += 1
    
    print(f"\nSuccessfully read {successful_reads}/{len(test_files)} files")
    
    if successful_reads > 0:
        print("\n✅ Phase 1 data is available and readable!")
        print("Dataset is ready for Phase 2 training.")
    else:
        print("\n❌ Could not read Phase 1 data files.")
        print("Check if data collection completed successfully.")

if __name__ == "__main__":
    main()