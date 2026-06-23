#!/usr/bin/env python3
"""
Check if we can access the Phase 1 data for training
"""

import os
from pathlib import Path

def check_data_access():
    """Check if we can access the Phase 1 data"""
    # Check the data directory
    data_dirs = [
        "Gated_Residual_strategy/outputs/run_20260603_214930",
        "Gated_Residual_strategy/data",
        "outputs/run_20260603_214930"
    ]
    
    found_data = False
    
    for data_dir in data_dirs:
        print(f"Checking directory: {data_dir}")
        if os.path.exists(data_dir):
            print(f"  Directory exists")
            # Look for HDF5 files
            h5_files = list(Path(data_dir).rglob("*.h5"))
            print(f"  Found {len(h5_files)} HDF5 files")
            
            if h5_files:
                print(f"  Sample files:")
                for f in h5_files[:3]:
                    print(f"    {f}")
                found_data = True
                break
        else:
            print(f"  Directory does not exist")
    
    if found_data:
        print("\n✅ Phase 1 data is accessible!")
        print("You can now run Phase 2 training:")
        print("  bash Gated_Residual_strategy/run_phase2_train_gate.sh")
    else:
        print("\n❌ Phase 1 data not found")
        print("Run Phase 1 data collection first:")
        print("  bash Gated_Residual_strategy/run_phase1_baseline.sh")

if __name__ == "__main__":
    check_data_access()