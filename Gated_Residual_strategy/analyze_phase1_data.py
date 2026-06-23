#!/usr/bin/env python3
"""
Analyze Phase 1 data to evaluate its quality for training the failure-risk gate
"""

import h5py
import json
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_h5_file(filepath):
    """Analyze a single HDF5 file"""
    print(f"\nAnalyzing {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"  Keys: {list(f.keys())}")
            
            # Check metadata
            if 'metadata' in f:
                print(f"  Metadata: {dict(f['metadata'].attrs)}")
            
            # Check labels
            if 'labels' in f:
                labels = f['labels'][:]
                unique, counts = np.unique(labels, return_counts=True)
                label_dist = dict(zip(unique, counts))
                print(f"  Label distribution: {label_dist}")
                print(f"  Total samples: {len(labels)}")
                print(f"  Failure rate: {label_dist.get(1, 0) / len(labels):.2%}")
            
            # Check observations
            if 'observations' in f:
                obs = f['observations']
                print(f"  Observation keys: {list(obs.keys())}")
                for key in obs.keys():
                    if hasattr(obs[key], 'shape'):
                        print(f"    {key}: {obs[key].shape}")
            
            # Check actions
            if 'actions' in f:
                actions = f['actions'][:]
                print(f"  Actions shape: {actions.shape}")
                
    except Exception as e:
        print(f"  Error reading file: {e}")

def analyze_log_file(filepath):
    """Analyze a log file"""
    print(f"\nAnalyzing {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        print(f"  Number of episodes: {len(data)}")
        
        if data:
            successes = sum(1 for ep in data if ep.get('success', False))
            print(f"  Success rate: {successes/len(data):.2%}")
            print(f"  First episode keys: {list(data[0].keys()) if data else []}")
            
    except Exception as e:
        print(f"  Error reading log file: {e}")

def analyze_dataset_quality(data_dir):
    """Analyze the overall quality of the Phase 1 dataset"""
    print("Analyzing Phase 1 Dataset Quality...")
    
    # Find all HDF5 files
    h5_files = list(Path(data_dir).rglob("failure_dataset_task*_seed*.h5"))
    log_files = list(Path(data_dir).rglob("logs_task*_seed*.json"))
    
    print(f"Found {len(h5_files)} HDF5 files and {len(log_files)} log files")
    
    # Collect statistics
    all_labels = []
    task_stats = {}
    
    for h5_file in h5_files[:5]:  # Analyze first 5 files
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'labels' in f:
                    labels = f['labels'][:]
                    all_labels.extend(labels)
                    
                    # Extract task_id and seed from filename
                    filename = os.path.basename(h5_file)
                    parts = filename.replace('failure_dataset_task', '').replace('.h5', '').split('_seed')
                    if len(parts) == 2:
                        task_id, seed = parts[0], parts[1]
                        task_key = f"task{task_id}"
                        if task_key not in task_stats:
                            task_stats[task_key] = {'success': 0, 'failure': 0, 'total': 0}
                        
                        unique, counts = np.unique(labels, return_counts=True)
                        label_dict = dict(zip(unique, counts))
                        task_stats[task_key]['success'] += label_dict.get(0, 0)
                        task_stats[task_key]['failure'] += label_dict.get(1, 0)
                        task_stats[task_key]['total'] += len(labels)
        except Exception as e:
            print(f"Error analyzing {h5_file}: {e}")
    
    # Overall statistics
    if all_labels:
        unique, counts = np.unique(all_labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        total_samples = len(all_labels)
        
        print(f"\nOverall Dataset Statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Success labels (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/total_samples:.2%})")
        print(f"  Failure labels (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/total_samples:.2%})")
        
        # Class balance ratio (min/max)
        if 0 in label_dist and 1 in label_dist:
            balance_ratio = min(label_dist[0], label_dist[1]) / max(label_dist[0], label_dist[1])
            print(f"  Class balance ratio: {balance_ratio:.3f}")
            
            if balance_ratio > 0.2:
                print("  ✅ Class balance is good (ratio > 0.2)")
            else:
                print("  ⚠️  Class balance may be problematic (ratio <= 0.2)")
        
        # Per-task statistics
        print(f"\nPer-Task Statistics:")
        for task, stats in task_stats.items():
            if stats['total'] > 0:
                failure_rate = stats['failure'] / stats['total']
                print(f"  {task}: {stats['failure']}/{stats['total']} failures ({failure_rate:.2%})")
    
    # Check log files for success rates
    print(f"\nEpisode Success Rates from Logs:")
    for log_file in log_files[:3]:  # Check first 3 log files
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                
            if data:
                successes = sum(1 for ep in data if ep.get('success', False))
                success_rate = successes / len(data) if data else 0
                print(f"  {os.path.basename(log_file)}: {success_rate:.2%} ({successes}/{len(data)})")
        except Exception as e:
            print(f"  Error reading {log_file}: {e}")
    
    return {
        'total_samples': len(all_labels),
        'success_count': label_dist.get(0, 0),
        'failure_count': label_dist.get(1, 0),
        'success_rate': label_dist.get(0, 0) / len(all_labels) if all_labels else 0,
        'failure_rate': label_dist.get(1, 0) / len(all_labels) if all_labels else 0,
        'task_stats': task_stats
    }

def main():
    # Change to the correct directory
    os.chdir('Gated_Residual_strategy')
    
    data_dir = "outputs/run_20260603_214930"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found")
        return
    
    # Analyze dataset quality
    stats = analyze_dataset_quality(data_dir)
    
    # Save quality report
    report = {
        'dataset_statistics': stats,
        'quality_assessment': {
            'sufficient_data': stats['total_samples'] > 1000,
            'balanced_classes': stats.get('failure_rate', 0) > 0.1 and stats.get('failure_rate', 0) < 0.9,
            'adequate_failure_examples': stats.get('failure_count', 0) > 100
        }
    }
    
    print(f"\nQuality Assessment:")
    print(f"  Sufficient data (>1000 samples): {report['quality_assessment']['sufficient_data']}")
    print(f"  Balanced classes (10-90% failure rate): {report['quality_assessment']['balanced_classes']}")
    print(f"  Adequate failure examples (>100): {report['quality_assessment']['adequate_failure_examples']}")
    
    # Determine if dataset is suitable for Phase 2
    suitable = all(report['quality_assessment'].values())
    print(f"\nDataset suitable for Phase 2 training: {'✅ YES' if suitable else '❌ NO'}")
    
    if suitable:
        print("\n✅ Dataset quality is good for training the failure-risk gate!")
        print("Next step: Run 'bash run_phase2_train_gate.sh' to train the gate")
    else:
        print("\n⚠️  Dataset quality may need improvement before training")

if __name__ == "__main__":
    main()