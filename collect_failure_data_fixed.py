"""
Phase 1: Data Collection for Gated Residual Strategy

This script runs the baseline SmolVLA policy on all LIBERO-10 tasks (3 seeds)
and collects trajectories, specifically logging which steps lead to success or failure.

Usage:
    python collect_failure_data.py --task_id 0 --seed 0
    python collect_failure_data.py --task_id all --seed all

Outputs:
    - failure_dataset.h5: Contains observations, actions, and binary labels (0=success, 1=failure)
    - per_step_logs.json: Detailed per-step success/failure logs
"""

import argparse
import json
import os
import h5py
import numpy as np
import torch
from pathlib import Path
import sys

# Add project paths
project_root = Path("/home/swagat/GIT/efficient_vla")
sys.path.insert(0, str(project_root))
lerobot_path = Path("/home/swagat/lerobot")
sys.path.insert(0, str(lerobot_path))

# Import your SmolVLA agent
# from rl_residual_agent import SmolVLA_Agent  # Adjust as needed
# from eval_week2_visual_prompting import evaluate_task

# TODO: Import your actual baseline evaluation components here
# These imports should match your actual project structure


def run_baseline_rollout(task_id, seed, num_episodes=10):
    """
    Runs the baseline SmolVLA policy for a given task and seed.
    
    Returns:
        list: List of trajectories, each containing:
            - observations: list of observation dicts
            - actions: list of actions
            - success: bool (whether the task was completed successfully)
    """
    trajectories = []
    
    # TODO: Implement your actual baseline rollout logic here
    # This is a placeholder structure
    
    for ep in range(num_episodes):
        trajectory = {
            "observations": [],
            "actions": [],
            "success": False
        }
        
        # Reset environment
        # env = reset_env(task_id, seed)
        
        # Run rollout
        # for step in range(max_steps):
        #     obs = env.get_observation()
        #     action = smolvla_agent.predict(obs)
        #     next_obs, reward, done, info = env.step(action)
        #     
        #     trajectory["observations"].append(obs)
        #     trajectory["actions"].append(action)
        #     
        #     if done:
        #         trajectory["success"] = info.get("success", False)
        #         break
        
        trajectories.append(trajectory)
    
    return trajectories


def save_failure_dataset(trajectories, output_path, task_id, seed):
    """
    Saves the collected trajectories to an HDF5 file.
    
    Args:
        trajectories: List of trajectory dicts
        output_path: Path to save the HDF5 file
        task_id: Task identifier
        seed: Random seed used
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create dataset groups
        f.create_group("metadata")
        f["metadata"].attrs["task_id"] = task_id
        f["metadata"].attrs["seed"] = seed
        f["metadata"].attrs["num_trajectories"] = len(trajectories)
        
        # Flatten all data
        all_obs = []
        all_actions = []
        all_labels = []
        all_success_flags = []
        
        for i, traj in enumerate(trajectories):
            for j, (obs, action) in enumerate(zip(traj["observations"], traj["actions"])):
                all_obs.append(obs)
                all_actions.append(action)
                
                # Label: 0 if step leads to success, 1 if step leads to failure
                # This is a simplified logic - adjust based on your actual success criteria
                step_label = 1 if not traj["success"] else 0
                all_labels.append(step_label)
                
                # Flag if the entire trajectory was successful
                all_success_flags.append(traj["success"])
        
        # Save as datasets
        f.create_dataset("observations", data=np.array(all_obs))
        f.create_dataset("actions", data=np.array(all_actions))
        f.create_dataset("labels", data=np.array(all_labels))
        f.create_dataset("trajectory_success", data=np.array(all_success_flags))
        
        # Save per-step metadata
        f.create_dataset("step_indices", data=np.arange(len(all_obs)))
        
    print(f"Saved {len(all_obs)} steps to {output_path}")
    print(f"  - Successful trajectories: {sum(all_success_flags)}")
    print(f"  - Failed trajectories: {len(all_success_flags) - sum(all_success_flags)}")
    print(f"  - Total failure steps: {sum(all_labels)}")


def save_per_step_logs(trajectories, task_id, seed, output_path):
    """
    Saves detailed per-step logs to a JSON file.
    """
    logs = []
    
    for i, traj in enumerate(trajectories):
        log_entry = {
            "trajectory_id": i,
            "task_id": task_id,
            "seed": seed,
            "success": traj["success"],
            "num_steps": len(traj["observations"]),
            "steps": []
        }
        
        for j, (obs, action) in enumerate(zip(traj["observations"], traj["actions"])):
            step_log = {
                "step": j,
                "obs_shape": obs.shape if hasattr(obs, 'shape') else "unknown",
                "action_shape": action.shape if hasattr(action, 'shape') else "unknown"
            }
            log_entry["steps"].append(step_log)
        
        logs.append(log_entry)
    
    with open(output_path, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Saved per-step logs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect failure data for Gated Residual Strategy")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID (0-9)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes per task/seed")
    parser.add_argument("--output_dir", type=str, default="Gated_Residual_strategy/data", help="Output directory")
    parser.add_argument("--run_all", action="store_true", help="Run all tasks and seeds")
    
    args = parser.parse_args()
    
    if args.run_all:
        print("Running all tasks (0-9) and seeds (0-2)...")
        for task_id in range(10):
            for seed in range(3):
                output_path = f"{args.output_dir}/failure_dataset_task{task_id}_seed{seed}.h5"
                logs_path = f"{args.output_dir}/logs_task{task_id}_seed{seed}.json"
                
                print(f"\nProcessing Task {task_id}, Seed {seed}")
                trajectories = run_baseline_rollout(task_id, seed, args.num_episodes)
                save_failure_dataset(trajectories, output_path, task_id, seed)
                save_per_step_logs(trajectories, task_id, seed, logs_path)
    else:
        print(f"Processing Task {args.task_id}, Seed {args.seed}")
        output_path = f"{args.output_dir}/failure_dataset_task{args.task_id}_seed{args.seed}.h5"
        logs_path = f"{args.output_dir}/logs_task{args.task_id}_seed{args.seed}.json"
        
        trajectories = run_baseline_rollout(args.task_id, args.seed, args.num_episodes)
        save_failure_dataset(trajectories, output_path, args.task_id, args.seed)
        save_per_step_logs(trajectories, args.task_id, args.seed, logs_path)


if __name__ == "__main__":
    main()
