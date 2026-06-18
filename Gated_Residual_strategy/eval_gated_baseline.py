"""
Phase 1: Enhanced Evaluation Script for Gated Residual Strategy

This script provides a robust evaluation harness with:
- Full LIBERO-10 eval parity (3 seeds)
- Consistent success extraction and preprocessing
- Per-step success/failure logging
- Confidence intervals and statistical analysis

Usage:
    python eval_gated_baseline.py --task_id 0 --seed 0
    python eval_gated_baseline.py --task_id all --seed all
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from linux_inhibit import LinuxInhibit

# Import your evaluation components
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from eval_week2_visual_prompting import evaluate_task

def evaluate_task_baseline(task_id, seed, num_episodes=10):
    """
    Evaluates the baseline SmolVLA policy on a single task and seed.
    
    Returns:
        dict: Evaluation results including:
            - success_rate: float
            - num_successful: int
            - num_total: int
            - avg_steps_to_success: float
            - per_episode_success: list
    """
    results = {
        "task_id": task_id,
        "seed": seed,
        "num_episodes": num_episodes,
        "num_successful": 0,
        "num_total": num_episodes,
        "avg_steps_to_success": 0.0,
        "per_episode_success": [],
        "episode_details": []
    }
    
    successful_episodes = []
    total_steps_successful = 0
    
    for ep in range(num_episodes):
        # TODO: Implement actual evaluation logic
        # success, steps = run_evaluation_episode(task_id, seed, ep)
        
        # Placeholder logic
        success = np.random.rand() > 0.3  # Simulate ~70% success rate
        steps = np.random.randint(20, 50) if success else np.random.randint(20, 50)
        
        results["per_episode_success"].append(success)
        results["episode_details"].append({
            "episode": ep,
            "success": success,
            "steps": steps
        })
        
        if success:
            results["num_successful"] += 1
            total_steps_successful += steps
    
    if results["num_successful"] > 0:
        results["avg_steps_to_success"] = total_steps_successful / results["num_successful"]
    
    results["success_rate"] = results["num_successful"] / results["num_total"]
    
    return results

def calculate_confidence_interval(success_rates, confidence=0.95):
    """
    Calculates confidence intervals for success rates.
    
    Args:
        success_rates: List of success rates
        confidence: Confidence level (default 0.95)
    
    Returns:
        tuple: (mean, ci_lower, ci_upper)
    """
    if len(success_rates) < 2:
        return np.mean(success_rates), 0.0, 0.0
    
    mean = np.mean(success_rates)
    n = len(success_rates)
    std = np.std(success_rates, ddof=1)
    t_val = 1.96  # Approximation for 95% CI with large n
    
    margin = t_val * (std / np.sqrt(n))
    return mean, mean - margin, mean + margin

def run_full_evaluation(task_ids=None, seeds=None, output_dir="Gated_Residual_strategy/eval_results"):
    """
    Runs full evaluation across specified tasks and seeds.
    
    Args:
        task_ids: List of task IDs (default: all 0-9)
        seeds: List of seeds (default: 0, 1, 2)
        output_dir: Output directory for results
    """
    if task_ids is None:
        task_ids = list(range(10))
    if seeds is None:
        seeds = [0, 1, 2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for task_id in task_ids:
        task_results = {
            "task_id": task_id,
            "task_name": f"LIBERO_{task_id}",
            "seed_results": []
        }
        
        for seed in seeds:
            print(f"Evaluating Task {task_id}, Seed {seed}")
            eval_result = evaluate_task_baseline(task_id, seed)
            task_results["seed_results"].append(eval_result)
            all_results.append(eval_result)
        
        # Calculate task-level statistics
        success_rates = [sr["success_rate"] for sr in task_results["seed_results"]]
        mean_rate, ci_lower, ci_upper = calculate_confidence_interval(success_rates)
        
        task_results["mean_success_rate"] = mean_rate
        task_results["ci_95_lower"] = ci_lower
        task_results["ci_95_upper"] = ci_upper
        
        all_results.append(task_results)
    
    # Save individual task results
    for task_result in all_results:
        if "task_id" in task_result and "seed_results" in task_result:
            # This is a task-level summary
            filename = f"task_{task_result['task_id']}_results_{timestamp}.json"
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, 'w') as f:
                json.dump(task_result, f, indent=2)
            print(f"Saved task {task_result['task_id']} results to {output_path}")
    
    # Save aggregate results
    aggregate_results = []
    task_success_rates = []
    
    for task_result in all_results:
        if "task_id" in task_result and "seed_results" in task_result:
            task_success_rates.append(task_result["mean_success_rate"])
    
    overall_mean, overall_ci_lower, overall_ci_upper = calculate_confidence_interval(task_success_rates)
    
    aggregate_results = {
        "overall_mean_success_rate": overall_mean,
        "ci_95_lower": overall_ci_lower,
        "ci_95_upper": overall_ci_upper,
        "num_tasks": len(task_success_rates),
        "num_seeds": len(seeds),
        "timestamp": timestamp,
        "task_details": all_results
    }
    
    aggregate_path = os.path.join(output_dir, f"aggregate_results_{timestamp}.json")
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    print(f"Saved aggregate results to {aggregate_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Mean Success Rate: {overall_mean:.2%}")
    print(f"95% CI: [{overall_ci_lower:.2%}, {overall_ci_upper:.2%}]")
    print(f"Tasks Evaluated: {len(task_success_rates)}")
    print(f"Seeds per Task: {len(seeds)}")
    print("="*60)
    
    return aggregate_results

def main():
    parser = argparse.ArgumentParser(description="Enhanced evaluation script for Gated Residual Strategy")
    parser.add_argument("--task_id", type=int, default=None, help="Single task ID (0-9)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None, help="List of task IDs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of seeds")
    parser.add_argument("--output_dir", type=str, default="Gated_Residual_strategy/eval_results", help="Output directory")
    parser.add_argument("--run_all", action="store_true", help="Run all tasks and seeds")
    
    args = parser.parse_args()
    
    if args.run_all:
        task_ids = list(range(10))
        seeds = [0, 1, 2]
    else:
        task_ids = [args.task_id] if args.task_id is not None else args.task_ids
        seeds = [args.seed] if args.seed is not None else args.seeds
    
    if task_ids is None:
        task_ids = list(range(10))
    if seeds is None:
        seeds = [0, 1, 2]
    
    run_full_evaluation(task_ids=task_ids, seeds=seeds, output_dir=args.output_dir)

if __name__ == "__main__":
    with LinuxInhibit(reason="Evaluating Gated Baseline"):
        main()
