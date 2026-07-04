#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from eval_gated_baseline import make_pre_post_processors, preprocess_observation
from libero.libero import benchmark

def find_task_in_suites(file_stem):
    task_name_candidate = file_stem
    if task_name_candidate.endswith("_demo"):
        task_name_candidate = task_name_candidate[:-5]
        
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name, suite_fn in benchmark_dict.items():
        suite = suite_fn()
        for task in suite.tasks:
            if task.name.lower() == task_name_candidate.lower():
                return task, suite_name
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Precompute SmolVLA base policy actions on human demonstrations")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .hdf5 or .h5 human demonstration files")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only 1 episode of 1 task")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading base SmolVLA policy...")
    policy_name = "HuggingFaceVLA/smolvla_libero"
    base_policy = SmolVLAPolicy.from_pretrained(policy_name).to(torch.bfloat16).to(device)
    base_policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(base_policy.config, policy_name)

    hdf5_files = sorted(list(Path(args.data_dir).rglob("*.hdf5")) + list(Path(args.data_dir).rglob("*.h5")))
    if not hdf5_files:
        print(f"No HDF5 files found in {args.data_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files.")

    for f_path in hdf5_files:
        file_stem = f_path.stem
        task, suite_name = find_task_in_suites(file_stem)
        if task is None:
            print(f"Could not map file {f_path.name} to any LIBERO task suite. Skipping.")
            continue

        instruction = task.language
        print(f"\nProcessing {f_path.name} -> Task: '{task.name}' | Instruction: '{instruction}'")

        # Open in read-write mode
        with h5py.File(f_path, "r+") as h5:
            if "data" not in h5:
                print(f"File {f_path.name} is not in Robomimic/standard LIBERO format. Skipping.")
                continue

            demo_keys = sorted(list(h5["data"].keys()))
            if args.debug:
                demo_keys = demo_keys[:1]
                print(f"[DEBUG] Processing only first episode: {demo_keys[0]}")

            BATCH_SIZE = 32

            for dk in tqdm(demo_keys, desc=f"Inference on {f_path.name}"):
                demo_group = h5["data"][dk]
                
                # Check if base_actions already exists, delete it if so
                if "base_actions" in demo_group["obs"]:
                    del demo_group["obs/base_actions"]

                # Load raw data
                agentview_rgb = demo_group["obs"]["agentview_rgb"][:]
                eye_in_hand_rgb = demo_group["obs"]["eye_in_hand_rgb"][:]
                ee_pos = demo_group["obs"]["ee_pos"][:]
                ee_ori = demo_group["obs"]["ee_ori"][:]
                gripper_states = demo_group["obs"]["gripper_states"][:]
                
                n_steps = agentview_rgb.shape[0]
                base_actions = [None] * n_steps

                # Process in batches
                for start_idx in range(0, n_steps, BATCH_SIZE):
                    end_idx = min(start_idx + BATCH_SIZE, n_steps)
                    batch_obs_list = []

                    for i in range(start_idx, end_idx):
                        img_agent = agentview_rgb[i][::-1, ::-1, :].copy()
                        img_wrist = eye_in_hand_rgb[i][::-1, ::-1, :].copy()
                        state_np = np.concatenate([ee_pos[i], ee_ori[i], gripper_states[i]])

                        raw_obs = {
                            "pixels": {
                                "image": img_agent,
                                "image2": img_wrist,
                            },
                            "agent_pos": state_np.astype(np.float32),
                        }

                        policy_obs = preprocess_observation(raw_obs)
                        policy_obs["task"] = [instruction]
                        batch_obs_i = preprocessor(policy_obs)
                        batch_obs_list.append(batch_obs_i)

                    # Stack keys to form batched inputs
                    keys = batch_obs_list[0].keys()
                    stacked_batch_obs = {}
                    for k in keys:
                        if torch.is_tensor(batch_obs_list[0][k]):
                            tensors = [b[k].to(device) for b in batch_obs_list]
                            stacked_batch_obs[k] = torch.cat(tensors, dim=0)
                        elif batch_obs_list[0][k] is None:
                            stacked_batch_obs[k] = None
                        elif isinstance(batch_obs_list[0][k], list):
                            stacked_batch_obs[k] = []
                            for b in batch_obs_list:
                                if b[k] is not None:
                                    stacked_batch_obs[k].extend(b[k])
                        else:
                            stacked_batch_obs[k] = [b[k] for b in batch_obs_list]

                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            base_action_tensor = base_policy.select_action(stacked_batch_obs)
                    
                    env_action = postprocessor(base_action_tensor)
                    actions_np = env_action.detach().cpu().to(torch.float32).numpy()
                    
                    for idx, i in enumerate(range(start_idx, end_idx)):
                        base_actions[i] = actions_np[idx]

                base_actions_np = np.array(base_actions, dtype=np.float32)
                demo_group.create_dataset("obs/base_actions", data=base_actions_np)

    print("\nPrecomputation successfully completed!")

if __name__ == "__main__":
    main()
