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
import random
import sys
import gc

# Prevent OpenGL/EGL context teardown crashes during MuJoCo garbage collection
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

# Fix for PyTorch 2.6+ weights_only=True default when loading libero init states
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Add project root to path to allow importing from other modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# LeRobot and LIBERO imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from robosuite.utils.transform_utils import quat2axisangle
import libero.libero as libero_pkg
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark_dict
from linux_inhibit import LinuxInhibit

# Get all task names from LIBERO-10 benchmark
BENCHMARK_NAME = "libero_10"
try:
    benchmark_dict = get_benchmark_dict()
    TASK_SUITE = benchmark_dict[BENCHMARK_NAME]()
    ALL_TASK_NAMES = TASK_SUITE.get_task_names()
except Exception as e:
    print(f"Could not initialize LIBERO benchmark '{BENCHMARK_NAME}': {e}")
    print("Please ensure LIBERO is installed and configured correctly.")
    sys.exit(1)

def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

def run_baseline_rollout(task_id, seed, num_episodes=10, max_steps=400):
    """
    Runs the baseline SmolVLA policy for a given task and seed.
    
    Returns:
        list: List of trajectories, each containing:
            - observations: list of observation dicts
            - actions: list of actions
            - success: bool (whether the task was completed successfully)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the baseline SmolVLA policy
    policy_name = "HuggingFaceVLA/smolvla_libero"
    print(f"Loading policy: {policy_name}")
    policy = SmolVLAPolicy.from_pretrained(policy_name).to(torch.bfloat16).to(device)
    policy.eval()

    # Get task name from task_id
    try:
        task = TASK_SUITE.get_task(task_id)
        task_name = task.name
    except IndexError:
        print(f"Error: Invalid task_id {task_id}. Must be between 0 and {len(ALL_TASK_NAMES) - 1}.")
        return []

    print(f"Initializing environment for task: {task_name}")

    random.seed(seed)
    np.random.seed(seed)

    benchmark_root = os.path.dirname(libero_pkg.__file__)
    bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=256,
        camera_widths=256,
    )

    trajectories = []
    processor = policy.model.vlm_with_expert.processor

    for ep in range(num_episodes):
        print(f"Running episode {ep + 1}/{num_episodes} for task '{task_name}' (id: {task_id}), seed: {seed}")

        # Reset environment and policy
        env.reset()
        init_states = TASK_SUITE.get_task_init_states(task_id)
        if init_states is not None and len(init_states) > 0:
            env.set_init_state(random.choice(init_states))
        obs = env.reset()

        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())

        instruction = task.language
        if hasattr(policy, 'reset'):
            policy.reset() # Clear any internal recurrent state if present

        trajectory = {
            "observations": [],
            "actions": [],
            "success": False
        }

        done = False
        step = 0
        while not done and step < max_steps:
            img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
            img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()

            img_tensor = torch.from_numpy(img_agent).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            img_tensor_wrist = torch.from_numpy(img_wrist).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            state_np = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
            state_tensor = torch.from_numpy(state_np).to(torch.bfloat16).unsqueeze(0).to(device)

            text_out = processor(text=instruction, return_tensors='pt')

            batch_obs = {
                'observation.images.image': img_tensor,
                'observation.images.image2': img_tensor_wrist,
                'observation.state': state_tensor,
                'observation.language.tokens': text_out['input_ids'].to(device),
                'observation.language.attention_mask': text_out['attention_mask'].to(device).bool(),
                'language_instruction': [instruction]
            }

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    action = policy.select_action(batch_obs)

            action_np = action.detach().cpu().to(torch.float32).numpy()[0]

            # Store raw observation dict and action
            trajectory["observations"].append(obs)
            trajectory["actions"].append(action_np)

            # Step environment
            next_obs, reward, done_env, info = env.step(action_np)

            step_success = info.get("success", False)
            if hasattr(step_success, 'item'):
                step_success = step_success.item()
            elif isinstance(step_success, list) or isinstance(step_success, np.ndarray):
                step_success = bool(step_success[0])
            else:
                step_success = bool(step_success)

            if step_success:
                trajectory["success"] = True
                done = True
            elif done_env:
                done = True

            obs = next_obs
            step += 1

        print(f"  Episode finished. Success: {trajectory['success']}")

        trajectories.append(trajectory)

    env.close()
    del env
    gc.collect()
    return trajectories

def save_failure_dataset(trajectories, output_path, task_id, seed):
    """
    Saves the collected trajectories to an HDF5 file, handling dictionary observations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_steps = sum(len(t['observations']) for t in trajectories)
    if total_steps == 0:
        print(f"No steps recorded for task {task_id}, seed {seed}. Skipping HDF5 creation.")
        return

    with h5py.File(output_path, 'w') as f:
        f.create_group("metadata")
        f["metadata"].attrs["task_id"] = task_id
        f["metadata"].attrs["seed"] = seed
        f["metadata"].attrs["num_trajectories"] = len(trajectories)

        first_obs = trajectories[0]["observations"][0]
        # Filter out any non-numpy array metadata (like strings) from obs
        obs_keys = [k for k, v in first_obs.items() if isinstance(v, np.ndarray)]
        flat_obs = {k: [] for k in obs_keys}
        all_actions = []
        all_labels = []
        successful_trajectories = sum(1 for t in trajectories if t["success"])

        for traj in trajectories:
            step_label = 0 if traj["success"] else 1
            for obs, action in zip(traj["observations"], traj["actions"]):
                for k in obs_keys:
                    flat_obs[k].append(obs[k])
                all_actions.append(action)
                all_labels.append(step_label)

        obs_group = f.create_group("observations")
        for k in obs_keys:
            obs_group.create_dataset(k, data=np.array(flat_obs[k]))
        f.create_dataset("actions", data=np.array(all_actions))
        f.create_dataset("labels", data=np.array(all_labels, dtype=np.uint8))

    print(f"Saved {total_steps} steps to {output_path}")
    print(f"  - Successful trajectories: {successful_trajectories}")
    print(f"  - Failed trajectories: {len(trajectories) - successful_trajectories}")
    print(f"  - Total failure-labeled steps: {sum(all_labels)}")

def save_per_step_logs(trajectories, task_id, seed, output_path):
    """
    Saves detailed per-step logs to a JSON file, handling dictionary observations.
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
            if isinstance(obs, dict):
                obs_shape = {k: v.shape for k, v in obs.items() if hasattr(v, 'shape')}
            else:
                obs_shape = obs.shape if hasattr(obs, 'shape') else "unknown"

            step_log = {
                "step": j,
                "obs_shape": str(obs_shape),
                "action_shape": action.shape if hasattr(action, 'shape') else "unknown"
            }
            log_entry["steps"].append(step_log)

        logs.append(log_entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, tuple): return list(obj)
            return super(NumpyEncoder, self).default(obj)

    with open(output_path, 'w') as f:
        json.dump(logs, f, indent=2, cls=NumpyEncoder)

    print(f"Saved per-step logs to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Collect failure data for Gated Residual Strategy")
    parser.add_argument("--task_id", type=str, default="0", help="Task ID (0-9) or 'all'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes per task/seed")
    parser.add_argument("--output_dir", type=str, default="Gated_Residual_strategy/data", help="Output directory")
    parser.add_argument("--run_all", action="store_true", help="Run all tasks and seeds")
    
    args = parser.parse_args()
    
    if args.task_id == "all" or args.run_all:
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
        task_id = int(args.task_id)
        print(f"Processing Task {task_id}, Seed {args.seed}")
        output_path = f"{args.output_dir}/failure_dataset_task{task_id}_seed{args.seed}.h5"
        logs_path = f"{args.output_dir}/logs_task{task_id}_seed{args.seed}.json"

        trajectories = run_baseline_rollout(task_id, args.seed, args.num_episodes)
        save_failure_dataset(trajectories, output_path, task_id, args.seed)
        save_per_step_logs(trajectories, task_id, args.seed, logs_path)

if __name__ == "__main__":
    with LinuxInhibit(reason="Data Collection"):
        main()
