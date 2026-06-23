#!/usr/bin/env python3
"""
Setup data for Phase 2 training by creating a symlink or copying data
"""

import os
import shutil
from pathlib import Path

def setup_phase2_data():
    """Setup data for Phase 2 training"""
    
    # Source directory with Phase 1 data
    source_dir = "Gated_Residual_strategy/outputs/run_20260603_214930"
    
    # Target directory for Phase 2
    target_dir = "Gated_Residual_strategy/data"
    
    print(f"Setting up Phase 2 data...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    # Check if source exists
    if not os.path.exists(source_dir):
        print(f"❌ Source directory does not exist: {source_dir}")
        return False
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all HDF5 files in source directory
    h5_files = list(Path(source_dir).rglob("failure_dataset_task*_seed*.h5"))
    print(f"Found {len(h5_files)} HDF5 files")
    
    if len(h5_files) == 0:
        print("❌ No HDF5 files found in source directory")
        return False
    
    # Create symlinks or copy files
    success_count = 0
    for h5_file in h5_files:
        # Get relative path from source_dir
        relative_path = h5_file.relative_to(source_dir)
        target_path = Path(target_dir) / relative_path
        
        # Create parent directories
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink
        if target_path.exists():
            target_path.unlink()
            
        try:
            target_path.symlink_to(h5_file)
            success_count += 1
        except Exception as e:
            print(f"Failed to create symlink for {h5_file}: {e}")
    
    print(f"✅ Successfully set up {success_count}/{len(h5_files)} files")
    
    if success_count > 0:
        print("\n✅ Phase 2 data setup complete!")
        print("You can now run Phase 2 training:")
        print("  cd Gated_Residual_strategy && bash run_phase2_train_gate.sh")
        return True
    else:
        print("\n❌ Failed to set up Phase 2 data")
        return False

if __name__ == "__main__":
    setup_phase2_data()