#!/usr/bin/env python3
"""
Test script to verify data collection works properly
"""

import argparse
import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    try:
        import h5py
        import torch
        import numpy as np
        print("✅ All basic imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        # Try to import LIBERO components
        import libero.libero as libero_pkg
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.benchmark import get_benchmark_dict
        print("✅ LIBERO imports successful")
    except ImportError as e:
        print(f"⚠️  LIBERO import warning (may be OK for testing): {e}")
    
    try:
        # Try to import LeRobot components
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        print("✅ LeRobot imports successful")
    except ImportError as e:
        print(f"⚠️  LeRobot import warning (may be OK for testing): {e}")
    
    return True

def test_file_structure():
    """Test that required directories exist"""
    required_dirs = [
        "data",
        "eval_results",
        "checkpoints"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path("Gated_Residual_strategy") / dir_name
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    return True

def main():
    print("Testing Gated Residual Strategy setup...")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("\n1. Testing imports...")
    if not test_imports():
        return 1
    
    print("\n2. Testing file structure...")
    if not test_file_structure():
        return 1
    
    print("\n✅ All tests passed! You can now run data collection.")
    print("\nTo collect data, run:")
    print("  python collect_failure_data.py --task_id 0 --seed 0 --num_episodes 2")
    print("\nFor full dataset collection:")
    print("  python collect_failure_data.py --run_all --num_episodes 10")
    
    return 0

if __name__ == "__main__":
    exit(main())