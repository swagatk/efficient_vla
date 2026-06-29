#!/usr/bin/env python3
"""
Phase 1 Dataset Quality Evaluator

This script evaluates the quality of the failure dataset collected in Phase 1
to ensure it's suitable for training the failure-risk gate in Phase 2.

Key evaluation metrics:
1. Class distribution (success/failure balance)
2. Data consistency checks
3. Per-task analysis
4. Cross-validation split quality
"""

import os
import argparse
import h5py
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_h5_file(filepath):
    """Analyze a single HDF5 file for data quality."""
    try:
        with h5py.File(filepath, 'r') as h5:
            # Check required keys
            required_keys = ["observations", "labels", "actions"]
            missing_keys = [key for key in required_keys if key not in h5]
            if missing_keys:
                return {"error": f"Missing keys: {missing_keys}"}
            
            # Get data shapes
            labels = h5["labels"][:]
            n_samples = len(labels)
            
            # Class distribution
            unique, counts = np.unique(labels, return_counts=True)
            class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
            
            # Check for data consistency
            observations = h5["observations"]
            obs_keys = list(observations.keys())
            
            # Image shape consistency
            img_shapes = {}
            if "agentview_image" in obs_keys:
                img_shapes["agentview"] = observations["agentview_image"].shape
            if "robot0_eye_in_hand_image" in obs_keys:
                img_shapes["eye_in_hand"] = observations["robot0_eye_in_hand_image"].shape
                
            # State dimensions
            state_dims = {}
            state_keys = [k for k in obs_keys if "eef" in k or "gripper" in k]
            for key in state_keys:
                state_dims[key] = observations[key].shape
                
            return {
                "n_samples": n_samples,
                "class_distribution": class_dist,
                "img_shapes": img_shapes,
                "state_dims": state_dims,
                "obs_keys": obs_keys
            }
    except Exception as e:
        return {"error": str(e)}

def analyze_log_file(filepath):
    """Analyze a log file for additional metadata."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Extract episode-level information
        episode_outcomes = []
        step_outcomes = []
        
        for episode in data.get("episodes", []):
            episode_outcomes.append(episode.get("success", False))
            for step in episode.get("steps", []):
                step_outcomes.append(step.get("failure", 0))
                
        return {
            "n_episodes": len(episode_outcomes),
            "episode_success_rate": np.mean(episode_outcomes),
            "n_steps": len(step_outcomes),
            "step_failure_rate": np.mean(step_outcomes)
        }
    except Exception as e:
        return {"error": str(e)}

def evaluate_dataset_quality(data_dir, output_dir=None):
    """Evaluate the overall quality of the Phase 1 dataset."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Find all dataset files
    h5_files = list(data_path.rglob("failure_dataset_task*_seed*.h5"))
    log_files = list(data_path.rglob("logs_task*_seed*.json"))
    
    print(f"Found {len(h5_files)} HDF5 files and {len(log_files)} log files")
    
    if not h5_files:
        raise ValueError("No HDF5 files found in the data directory")
    
    # Analyze each file
    file_results = {}
    task_stats = defaultdict(list)
    overall_stats = {
        "total_samples": 0,
        "class_distribution": Counter(),
        "file_errors": 0
    }
    
    print("\n=== FILE-LEVEL ANALYSIS ===")
    for h5_file in h5_files:
        file_result = analyze_h5_file(h5_file)
        file_results[str(h5_file)] = file_result
        
        # Extract task and seed info
        filename = h5_file.name
        parts = filename.replace("failure_dataset_", "").replace(".h5", "").split("_")
        task_id = parts[0].replace("task", "")
        seed = parts[1].replace("seed", "")
        
        if "error" in file_result:
            print(f"❌ {filename}: {file_result['error']}")
            overall_stats["file_errors"] += 1
            continue
            
        print(f"✅ {filename}: {file_result['n_samples']} samples")
        
        # Update overall stats
        overall_stats["total_samples"] += file_result["n_samples"]
        for cls, count in file_result["class_distribution"].items():
            overall_stats["class_distribution"][cls] += count
            
        # Store per-task stats
        task_stats[task_id].append({
            "seed": seed,
            "samples": file_result["n_samples"],
            "class_dist": file_result["class_distribution"]
        })
    
    # Per-task analysis
    print("\n=== PER-TASK ANALYSIS ===")
    task_summary = {}
    for task_id, seed_data in task_stats.items():
        total_samples = sum(d["samples"] for d in seed_data)
        class_counts = Counter()
        for d in seed_data:
            for cls, count in d["class_dist"].items():
                class_counts[cls] += count
                
        task_summary[task_id] = {
            "total_samples": total_samples,
            "n_seeds": len(seed_data),
            "class_distribution": dict(class_counts)
        }
        
        # Calculate class balance
        if len(class_counts) >= 2:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            balance_ratio = min_count / max_count if max_count > 0 else 0
        else:
            balance_ratio = 0
            
        print(f"Task {task_id}: {total_samples} samples, {len(seed_data)} seeds")
        print(f"  Class distribution: {dict(class_counts)}")
        print(f"  Balance ratio: {balance_ratio:.3f}")
        
        # Quality assessment
        if balance_ratio >= 0.2 and total_samples > 50:
            print(f"  🟢 Good quality dataset")
        elif balance_ratio >= 0.1 and total_samples > 20:
            print(f"  🟡 Acceptable quality dataset")
        else:
            print(f"  🔴 Poor quality dataset")
    
    # Overall dataset quality
    print("\n=== OVERALL DATASET QUALITY ===")
    print(f"Total samples: {overall_stats['total_samples']}")
    print(f"Total files with errors: {overall_stats['file_errors']}")
    print(f"Class distribution: {dict(overall_stats['class_distribution'])}")
    
    # Calculate overall balance
    if len(overall_stats["class_distribution"]) >= 2:
        min_count = min(overall_stats["class_distribution"].values())
        max_count = max(overall_stats["class_distribution"].values())
        overall_balance = min_count / max_count if max_count > 0 else 0
        print(f"Overall balance ratio: {overall_balance:.3f}")
    else:
        overall_balance = 0
        print("⚠️  Only one class present in dataset!")
        
    # Final quality assessment
    print("\n=== QUALITY ASSESSMENT ===")
    if overall_stats["file_errors"] > 0:
        print("❌ Dataset has file errors that need to be addressed")
    elif overall_balance < 0.1:
        print("❌ Dataset is highly imbalanced - not suitable for training")
    elif overall_balance < 0.2:
        print("⚠️  Dataset is moderately imbalanced - may need augmentation")
    elif overall_stats["total_samples"] < 100:
        print("⚠️  Dataset is small - may need more data collection")
    else:
        print("✅ Dataset appears to be of good quality for Phase 2 training")
        
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results = {
            "file_results": file_results,
            "task_summary": task_summary,
            "overall_stats": {
                "total_samples": int(overall_stats["total_samples"]),
                "class_distribution": {int(k): int(v) for k, v in overall_stats["class_distribution"].items()},
                "file_errors": int(overall_stats["file_errors"])
            },
            "overall_balance": float(overall_balance)
        }
        
        results_file = output_path / "dataset_quality_report.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nDetailed report saved to {results_file}")
        
        # Create a simple visualization
        if len(overall_stats["class_distribution"]) >= 2:
            classes = list(overall_stats["class_distribution"].keys())
            counts = list(overall_stats["class_distribution"].values())
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(classes, counts, color=['blue', 'red'])
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.title('Dataset Class Distribution')
            plt.xticks(classes, [f'Success ({classes[0]})', f'Failure ({classes[1]})'])
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
            
            plot_file = output_path / "class_distribution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Class distribution plot saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 dataset quality")
    parser.add_argument("--data_dir", type=str, default="Gated_Residual_strategy/data",
                        help="Path to the Phase 1 dataset directory")
    parser.add_argument("--output_dir", type=str, default="Gated_Residual_strategy/dataset_analysis",
                        help="Path to save analysis results")
    
    args = parser.parse_args()
    
    try:
        evaluate_dataset_quality(args.data_dir, args.output_dir)
    except Exception as e:
        print(f"Error evaluating dataset: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())