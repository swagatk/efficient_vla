#!/usr/bin/env python3
"""
Phase 4: Gated Residual Strategy Evaluation Harness

This script performs full LIBERO-10 evaluation (multi-seed, multi-episode)
for both the baseline SmolVLA policy and the Gated Residual Corrector policy.
"""

import argparse
import json
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from linux_inhibit import LinuxInhibit
import libero.libero as libero_pkg
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from robosuite.utils.transform_utils import quat2axisangle

# Redefine architectures locally to avoid import path complexities
class LightweightFailureGate(nn.Module):
    def __init__(self, state_dim=8):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        # 768 (img1 SigLIP pooler) + 768 (img2 SigLIP pooler) + 32 (state) = 1568
        self.fusion = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, feat1, feat2, state):
        xs = self.state_mlp(state)
        x = torch.cat([feat1, feat2, xs], dim=1)
        return self.fusion(x)

class LightweightResidualCorrector(nn.Module):
    def __init__(self, state_dim=8, action_dim=7):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        # 768 (img1 SigLIP pooler) + 768 (img2 SigLIP pooler) + 32 (state) = 1568
        self.fusion = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, feat1, feat2, state):
        xs = self.state_mlp(state)
        x = torch.cat([feat1, feat2, xs], dim=1)
        return self.fusion(x)

def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

def evaluate_task(task_id, seed, base_policy, preprocessor, postprocessor, 
                  gate_model=None, corrector_model=None, 
                  threshold=0.5, alpha=0.5, num_episodes=10, max_steps=400, device="cuda",
                  inference_mode="absolute", benchmark_name="libero_10"):
    """
    Evaluates policy on a single LIBERO task. Blends corrector output when gate triggers.
    """
    benchmark_dict = get_benchmark_dict()
    benchmark = benchmark_dict[benchmark_name]()
    
    try:
        task = benchmark.get_task(task_id)
        task_name = task.name
    except IndexError:
        print(f"Error: Invalid task_id {task_id}.")
        return {}

    print(f"\n--- Starting Evaluation: Task '{task_name}' (ID: {task_id}), Seed: {seed} ---")
    if gate_model and corrector_model:
        print(f"Gating Active (Threshold: {threshold}, Alpha: {alpha}, Mode: {inference_mode})")
    else:
        print("Running Baseline SmolVLA (No Gating/Corrector)")

    # Seed all sources
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    benchmark_root = os.path.dirname(libero_pkg.__file__)
    bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=256,
        camera_widths=256,
    )

    results = {
        "task_id": task_id,
        "task_name": task_name,
        "seed": seed,
        "num_episodes": num_episodes,
        "num_successful": 0,
        "avg_steps_to_success": 0.0,
        "intervention_rate": 0.0,
        "episode_details": []
    }

    total_steps_successful = 0
    total_steps_run = 0
    total_interventions = 0

    init_states = benchmark.get_task_init_states(task_id)

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}...")
        env.reset()
        if init_states is not None and len(init_states) > ep:
            env.set_init_state(init_states[ep])
        obs = env.reset()

        # Warmup
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())

        instruction = task.language
        if hasattr(base_policy, 'reset'):
            base_policy.reset()

        done = False
        step = 0
        ep_success = False
        ep_interventions = 0

        while not done and step < max_steps:
            img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
            img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()
            state_np = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
            
            # 1. Forward through Base SmolVLA Policy
            raw_obs = {
                "pixels": {
                    "image": img_agent,
                    "image2": img_wrist,
                },
                "agent_pos": state_np.astype(np.float32),
            }
            policy_obs = preprocess_observation(raw_obs)
            policy_obs["task"] = [instruction]
            batch_obs = preprocessor(policy_obs)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base_action_tensor = base_policy.select_action(batch_obs)
            
            env_action = postprocessor(base_action_tensor)
            action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]

            # 2. Check risk gate if loaded
            triggered = False
            if gate_model is not None and corrector_model is not None:
                state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    # Extract visual features from policy preprocessed inputs
                    vision_tower = base_policy.model.vlm_with_expert.get_vlm_model().vision_model
                    v_dtype = vision_tower.dtype
                    feat1 = vision_tower(batch_obs["observation.images.image"].to(dtype=v_dtype)).last_hidden_state.mean(dim=1)
                    feat2 = vision_tower(batch_obs["observation.images.image2"].to(dtype=v_dtype)).last_hidden_state.mean(dim=1)
                    
                    gate_logits = gate_model(feat1.to(torch.float32), feat2.to(torch.float32), state_tensor)
                    gate_prob = torch.sigmoid(gate_logits).item()

                if gate_prob > threshold:
                    triggered = True
                    ep_interventions += 1
                    total_interventions += 1
                    
                    # 3. Predict correction
                    with torch.no_grad():
                        corr_action_tensor = corrector_model(feat1.to(torch.float32), feat2.to(torch.float32), state_tensor)
                        corr_action_np = corr_action_tensor.cpu().numpy()[0]
                    
                    # Blend base action and corrective action
                    if inference_mode == "absolute":
                        action_np = (1.0 - alpha) * action_np + alpha * corr_action_np
                    elif inference_mode == "delta":
                        action_np = action_np + alpha * corr_action_np

            # Step environment
            next_obs, reward, done_env, info = env.step(action_np)

            step_success = False
            if hasattr(env, "check_success") and env.check_success():
                step_success = True
            elif isinstance(info, dict) and info.get("success", False):
                step_success = True

            if step_success:
                ep_success = True
                done = True
            elif done_env:
                done = True

            obs = next_obs
            step += 1

        print(f"  Episode finished. Steps: {step} | Success: {ep_success} | Interventions: {ep_interventions}")
        total_steps_run += step
        if ep_success:
            results["num_successful"] += 1
            total_steps_successful += step

        results["episode_details"].append({
            "episode": ep,
            "success": ep_success,
            "steps": step,
            "interventions": ep_interventions
        })

    env.close()
    
    results["success_rate"] = results["num_successful"] / num_episodes
    if results["num_successful"] > 0:
        results["avg_steps_to_success"] = total_steps_successful / results["num_successful"]
    results["intervention_rate"] = total_interventions / total_steps_run if total_steps_run > 0 else 0.0
    
    print(f"Task Results: Success Rate = {results['success_rate']:.2%}, Avg Steps = {results['avg_steps_to_success']:.1f}, Intervention Rate = {results['intervention_rate']:.2%}")
    return results

def calculate_confidence_interval(success_rates, confidence=0.95):
    if len(success_rates) < 2:
        return np.mean(success_rates), 0.0, 0.0
    mean = np.mean(success_rates)
    std = np.std(success_rates, ddof=1)
    # 95% confidence interval multiplier (approximate)
    margin = 1.96 * (std / np.sqrt(len(success_rates)))
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA Gated Residual Strategy")
    parser.add_argument("--task_id", type=int, default=None, help="Evaluate a single task (0-9)")
    parser.add_argument("--seed", type=int, default=None, help="Evaluate a single seed (0-2)")
    parser.add_argument("--run_all", action="store_true", help="Evaluate all 10 tasks and 3 seeds")
    parser.add_argument("--gate_dir", type=str, default=None, help="Path to Phase 2 training outputs directory (loads corresponding seed checkpoint)")
    parser.add_argument("--corrector_dir", type=str, default=None, help="Path to Phase 3 training outputs directory (loads corresponding seed checkpoint)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Gating confidence threshold")
    parser.add_argument("--alpha", type=float, default=0.5, help="Residual interpolation scaling factor")
    parser.add_argument("--num_episodes", type=int, default=10, help="Episodes per task-seed run")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--inference_mode", type=str, choices=["absolute", "delta"], default="absolute", help="Inference action blending mode")
    parser.add_argument("--output_dir", type=str, default="Gated_Residual_strategy/eval_results", help="Directory to save evaluation reports")
    parser.add_argument("--adaptive_gating", action="store_true", help="Enable task-adaptive gating: only use gate/corrector on specified tasks")
    parser.add_argument("--active_gating_tasks", type=int, nargs="+", default=[2, 7, 9], help="Task IDs where Gated Residual correction is active")
    parser.add_argument("--benchmark", type=str, default="libero_10", help="Libero benchmark name (e.g. libero_10, libero_goal)")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Base Policy once
    print("Loading baseline SmolVLA policy...")
    policy_name = "HuggingFaceVLA/smolvla_libero"
    base_policy = SmolVLAPolicy.from_pretrained(policy_name).to(torch.bfloat16).to(device)
    base_policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(base_policy.config, policy_name)

    # Define tasks and seeds to run
    task_ids = list(range(10)) if (args.run_all or args.task_id is None) else [args.task_id]
    seeds = [0, 1, 2] if (args.run_all or args.seed is None) else [args.seed]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_runs = []

    for seed in seeds:
        # Load Gate and Corrector checkpoints matching the evaluation seed if available
        # Note: mapping eval seeds [0, 1, 2] to training seeds [42, 123, 999]
        train_seed_map = {0: 42, 1: 123, 2: 999}
        train_seed = train_seed_map.get(seed, 42)
        
        for task_id in task_ids:
            gate_model = None
            corrector_model = None

            if args.gate_dir:
                gate_path = Path(args.gate_dir) / f"task_{task_id}" / f"unit_train_seed_{train_seed}" / "best_model.pth"
                if not gate_path.exists():
                    gate_path = Path(args.gate_dir) / f"unit_train_seed_{train_seed}" / "best_model.pth"
                
                if gate_path.exists():
                    print(f"Loading Failure Gate from {gate_path}")
                    # Failure Gate uses state_dim=8
                    gate_model = LightweightFailureGate(state_dim=8).to(device)
                    gate_model.load_state_dict(torch.load(gate_path, map_location=device))
                    gate_model.eval()
                else:
                    print(f"[Warning] Failure Gate checkpoint not found: {gate_path}")

            if args.corrector_dir:
                corrector_path = Path(args.corrector_dir) / f"task_{task_id}" / f"unit_train_seed_{train_seed}" / "best_model.pth"
                if not corrector_path.exists():
                    corrector_path = Path(args.corrector_dir) / f"unit_train_seed_{train_seed}" / "best_model.pth"
                
                if corrector_path.exists():
                    print(f"Loading Residual Corrector from {corrector_path}")
                    # Corrector uses state_dim=8, action_dim=7
                    corrector_model = LightweightResidualCorrector(state_dim=8, action_dim=7).to(device)
                    corrector_model.load_state_dict(torch.load(corrector_path, map_location=device))
                    corrector_model.eval()
                else:
                    print(f"[Warning] Residual Corrector checkpoint not found: {corrector_path}")

            active_gate = gate_model
            active_corr = corrector_model
            if args.adaptive_gating and task_id not in args.active_gating_tasks:
                active_gate = None
                active_corr = None

            run_result = evaluate_task(
                task_id=task_id,
                seed=seed,
                base_policy=base_policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                gate_model=active_gate,
                corrector_model=active_corr,
                threshold=args.threshold,
                alpha=args.alpha,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                device=device,
                inference_mode=args.inference_mode,
                benchmark_name=args.benchmark
            )
            all_runs.append(run_result)

    # 2. Compile and save results
    task_success_rates = []
    task_summaries = {}
    
    # Group runs by task to calculate task-level stats
    for task_id in task_ids:
        task_runs = [r for r in all_runs if r["task_id"] == task_id]
        if not task_runs:
            continue
        rates = [r["success_rate"] for r in task_runs]
        mean, ci_lower, ci_upper = calculate_confidence_interval(rates)
        task_success_rates.append(mean)
        
        task_summaries[str(task_id)] = {
            "task_name": task_runs[0]["task_name"],
            "success_rate_mean": mean,
            "success_rate_ci_95": [ci_lower, ci_upper],
            "runs": task_runs
        }

    overall_mean, overall_ci_lower, overall_ci_upper = calculate_confidence_interval(task_success_rates)
    
    aggregate_report = {
        "overall_mean_success_rate": overall_mean,
        "ci_95_lower": overall_ci_lower,
        "ci_95_upper": overall_ci_upper,
        "timestamp": timestamp,
        "config": {
            "gate_dir": args.gate_dir,
            "corrector_dir": args.corrector_dir,
            "threshold": args.threshold,
            "alpha": args.alpha,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "inference_mode": args.inference_mode,
            "adaptive_gating": args.adaptive_gating,
            "active_gating_tasks": args.active_gating_tasks
        },
        "tasks": task_summaries
    }

    report_path = os.path.join(args.output_dir, f"evaluation_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(aggregate_report, f, indent=2)

    print("\n" + "="*60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Overall Success Rate: {overall_mean:.2%}")
    print(f"95% Confidence Interval: [{overall_ci_lower:.2%}, {overall_ci_upper:.2%}]")
    print(f"Detailed results saved to: {report_path}")
    print("="*60)

if __name__ == "__main__":
    with LinuxInhibit(reason="Gated Residual Evaluation"):
        main()
