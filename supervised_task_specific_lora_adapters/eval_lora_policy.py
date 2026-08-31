#!/usr/bin/env python3
import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/home/swagat/lerobot/src")

import transformers
import argparse
import json
import numpy as np
import torch

# Fix for PyTorch 2.6+ weights_only=True default when loading libero init states
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import random
from datetime import datetime

# LIBERO/Robosuite imports
import libero.libero as libero_pkg
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

# LeRobot/SmolVLA imports
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors

# PEFT import
from peft import PeftModel
import gc

# Safe out-of-place RoPE implementation to prevent CUDA slice mutation segfaults
import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_expert_module

_TIMESCALE_CACHE = {}

def _get_cached_timescale(dim, device, max_wavelength=10000):
    key = (dim, str(device), max_wavelength)
    if key not in _TIMESCALE_CACHE:
        _TIMESCALE_CACHE[key] = max_wavelength ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    return _TIMESCALE_CACHE[key]

def _safe_apply_rope(x, positions, max_wavelength=10000):
    dtype = x.dtype
    timescale = _get_cached_timescale(x.shape[-1], x.device, max_wavelength)

    if positions.ndim == 1:
        positions = positions.unsqueeze(0)

    radians = (positions[..., None].to(torch.float32) / timescale)
    seq_len = positions.shape[-1]
    if x.ndim == 4:
        if x.shape[1] == seq_len:
            radians = radians.unsqueeze(2)
        elif x.shape[2] == seq_len:
            radians = radians.unsqueeze(1)
        else:
            radians = radians.unsqueeze(2)
    else:
        radians = radians.unsqueeze(-2)

    sin = torch.sin(radians).to(dtype)
    cos = torch.cos(radians).to(dtype)

    x1, x2 = x.chunk(2, dim=-1)
    res1 = x1 * cos - x2 * sin
    res2 = x2 * cos + x1 * sin
    return torch.cat([res1, res2], dim=-1)

smolvlm_expert_module.apply_rope = _safe_apply_rope

# Safe KV cache handling in forward_attn_layer to prevent dangling CUDA memory pointers
_orig_forward_attn = smolvlm_expert_module.SmolVLMWithExpertModel.forward_attn_layer

def _safe_forward_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, head_dim, use_cache=True, fill_kv_cache=True, past_key_values=None):
    if use_cache and past_key_values is not None and not fill_kv_cache and layer_idx in past_key_values:
        past_key_values[layer_idx]["key_states"] = past_key_values[layer_idx]["key_states"].contiguous()
        past_key_values[layer_idx]["value_states"] = past_key_values[layer_idx]["value_states"].contiguous()
    res, past_key_values = _orig_forward_attn(self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, head_dim, use_cache, fill_kv_cache, past_key_values)
    if use_cache and fill_kv_cache and past_key_values is not None and layer_idx in past_key_values:
        past_key_values[layer_idx]["key_states"] = past_key_values[layer_idx]["key_states"].detach().clone()
        past_key_values[layer_idx]["value_states"] = past_key_values[layer_idx]["value_states"].detach().clone()
    return res, past_key_values

smolvlm_expert_module.SmolVLMWithExpertModel.forward_attn_layer = _safe_forward_attn_layer

def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

def extract_success(info):
    if not isinstance(info, dict):
        return False
    for key in ("success", "is_success", "task_success", "episode_success"):
        if key not in info:
            continue
        value = info[key]
        if hasattr(value, "item"):
            value = value.item()
        return bool(value)
    return False

def evaluate_task(task_id, seed, policy, preprocessor, postprocessor, 
                  num_episodes=10, max_steps=520, device="cuda", benchmark_name="libero_10"):
    """
    Evaluates the LoRA-adapted policy on a single LIBERO task.
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
        ignore_done=True,
    )

    results = {
        "task_id": task_id,
        "task_name": task_name,
        "seed": seed,
        "num_episodes": num_episodes,
        "num_successful": 0,
        "avg_steps_to_success": 0.0,
        "episode_details": []
    }

    total_steps_successful = 0
    total_steps_run = 0

    init_states = benchmark.get_task_init_states(task_id)

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}...", flush=True)
        env.reset()
        if init_states is not None and len(init_states) > ep:
            env.set_init_state(init_states[ep])
        obs = env.reset()

        # Warmup environment
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())

        instruction = task.language
        if hasattr(policy, 'reset'):
            policy.reset()

        done = False
        step = 0
        ep_success = False

        while not done and step < max_steps:
            img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
            img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()
            state_np = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)
            
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
            
            # Predict actions using policy (which contains the active task-specific LoRA adapters)
            torch.cuda.synchronize()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    action_tensor = policy.select_action(batch_obs)
            
            env_action = postprocessor(action_tensor)
            action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]
            
            # Step the environment
            torch.cuda.synchronize()
            obs, reward, done, info = env.step(action_np)
            step += 1
            if step % 100 == 0:
                print(f"    Step {step}/{max_steps}...", flush=True)
            
            # Extract success flag
            if extract_success(info) or reward > 0.9:
                ep_success = True
                done = True

        print(f"  Episode finished. Steps: {step} | Success: {ep_success}", flush=True)
        
        if ep_success:
            results["num_successful"] += 1
            total_steps_successful += step
        total_steps_run += step

        results["episode_details"].append({
            "episode": ep,
            "success": ep_success,
            "steps": step
        })

        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        gc.collect()

    env.close()
    torch.cuda.empty_cache()
    gc.collect()
    
    success_rate = results["num_successful"] / num_episodes
    avg_steps = total_steps_successful / results["num_successful"] if results["num_successful"] > 0 else float(max_steps)
    
    results["success_rate"] = success_rate
    results["avg_steps_to_success"] = avg_steps
    
    print(f"Task Results: Success Rate = {success_rate:.2%}, Avg Steps = {avg_steps:.1f}", flush=True)
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy with task-specific LoRA adapters")
    parser.add_argument("--task_id", type=int, default=None, help="Evaluate a single task (0-9)")
    parser.add_argument("--seed", type=int, default=None, help="Evaluate a single seed (0-2)")
    parser.add_argument("--run_all", action="store_true", help="Evaluate all 10 tasks and 3 seeds")
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to checkpoints root directory containing task folders (e.g. checkpoints_lora)")
    parser.add_argument("--num_episodes", type=int, default=10, help="Episodes per task-seed run")
    parser.add_argument("--max_steps", type=int, default=520, help="Max steps per episode")
    parser.add_argument("--output_dir", type=str, default="lora_eval_results", help="Directory to save evaluation reports")
    parser.add_argument("--benchmark", type=str, default="libero_10", help="Libero benchmark name (e.g. libero_10)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Base Policy once
    print("Loading baseline SmolVLA policy...")
    policy_name = "HuggingFaceVLA/smolvla_libero"
    base_policy = SmolVLAPolicy.from_pretrained(policy_name).to(device)
    base_policy.eval()
    if hasattr(base_policy, "model") and hasattr(base_policy.model, "vlm_with_expert"):
        if hasattr(base_policy.model.vlm_with_expert, "config"):
            base_policy.model.vlm_with_expert.config._attn_implementation = "eager"
    preprocessor, postprocessor = make_pre_post_processors(base_policy.config, policy_name)

    # Define tasks and seeds to run
    task_ids = list(range(10)) if (args.run_all or args.task_id is None) else [args.task_id]
    seeds = [0, 1, 2] if (args.run_all or args.seed is None) else [args.seed]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_runs = []

    for task_id in task_ids:
        # Load task-specific LoRA weights if provided
        active_policy = base_policy
        if args.lora_dir is not None:
            saved_adapter_path = os.path.join(args.lora_dir, f"task_{task_id}")
            if os.path.exists(saved_adapter_path):
                print(f"Loading LoRA adapters for task {task_id} from {saved_adapter_path}...")
                
                # Wrap the action expert using PEFT loading saved adapter weights
                active_policy = base_policy
                if hasattr(active_policy, "model") and hasattr(active_policy.model, "vlm_with_expert"):
                    if hasattr(active_policy.model.vlm_with_expert, "config"):
                        active_policy.model.vlm_with_expert.config._attn_implementation = "eager"
                active_policy.model.vlm_with_expert.lm_expert = PeftModel.from_pretrained(
                    active_policy.model.vlm_with_expert.lm_expert,
                    saved_adapter_path
                ).to(device)
                target_dtype = next(active_policy.model.vlm_with_expert.vlm.parameters()).dtype
                active_policy.model.vlm_with_expert.lm_expert.to(target_dtype)
                active_policy.eval()
            else:
                print(f"[Warning] LoRA checkpoint not found for task {task_id}: {saved_adapter_path}. Running base policy.")

        task_runs = []
        for seed in seeds:
            run_output_dir = os.path.join(args.output_dir, f"task_{task_id}_seed_{seed}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            # Check if evaluation report already exists in run_output_dir
            existing_reports = sorted(list(Path(run_output_dir).glob("evaluation_report_*.json")))
            if existing_reports:
                latest_report = existing_reports[-1]
                try:
                    with open(latest_report, "r") as f:
                        saved_result = json.load(f)
                    if saved_result.get("num_episodes", 0) >= args.num_episodes:
                        print(f"--- Task {task_id}, Seed {seed} already completed (found {latest_report.name}). Skipping and loading saved result. ---")
                        task_runs.append(saved_result)
                        all_runs.append(saved_result)
                        continue
                except Exception as e:
                    print(f"Warning: Could not read existing report {latest_report}: {e}. Re-running evaluation.")

            run_result = evaluate_task(
                task_id=task_id,
                seed=seed,
                policy=active_policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                device=device,
                benchmark_name=args.benchmark
            )
            
            # Save individual run report
            report_path = os.path.join(run_output_dir, f"evaluation_report_{timestamp}.json")
            with open(report_path, "w") as f:
                json.dump(run_result, f, indent=2)
            
            task_runs.append(run_result)
            all_runs.append(run_result)

    # Compute overall metrics
    if all_runs:
        success_rates = [r["success_rate"] for r in all_runs]
        mean_rate = np.mean(success_rates)
        print("\n" + "="*60)
        print("LORA EVALUATION COMPLETED")
        print("="*60)
        print(f"Overall Success Rate: {mean_rate:.2%}")
        print("="*60)

if __name__ == "__main__":
    main()
