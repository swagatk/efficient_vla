#!/usr/bin/env python3
import faulthandler
faulthandler.enable()

import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/home/swagat/lerobot/src")

# Add LIBERO repository path if present locally
_libero_repo = os.path.expanduser("~/LIBERO")
if os.path.isdir(_libero_repo) and _libero_repo not in sys.path:
    sys.path.insert(0, _libero_repo)


import transformers
import argparse
import json
import numpy as np
import torch

# Fix for PyAV 15+ missing av.option.Option type hint in LeRobot
try:
    import av
    import types
    if not hasattr(av, "option"):
        av.option = types.SimpleNamespace(Option=object)
except Exception:
    pass


# Fix for PyTorch 2.6+ weights_only=True default when loading libero init states
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import random
from datetime import datetime
import yaml

# Ensure ~/.libero/config.yaml exists and points to valid existing folders
_libero_cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
_libero_cfg_file = _libero_cfg_dir / "config.yaml"

def _ensure_valid_libero_config():
    try:
        candidates = [
            Path(os.path.expanduser("~/LIBERO/libero/libero")),
            Path(os.path.expanduser("~/LIBERO/libero")),
        ]
        import importlib.util
        for mod in ["libero.libero", "libero"]:
            try:
                spec = importlib.util.find_spec(mod)
                if spec and spec.origin:
                    p = Path(spec.origin).resolve().parent
                    candidates.extend([p, p / "libero"])
            except Exception:
                pass

        bench_root = None
        for c in candidates:
            if (c / "init_files").is_dir() and (c / "bddl_files").is_dir():
                bench_root = c
                break

        if bench_root is None:
            # Fallback to any directory that has init_files
            for c in candidates:
                if (c / "init_files").is_dir():
                    bench_root = c
                    break

        if bench_root is None:
            bench_root = Path(os.path.expanduser("~/LIBERO/libero/libero"))

        needs_write = True
        if _libero_cfg_file.exists():
            try:
                with open(_libero_cfg_file, "r") as f:
                    curr = yaml.safe_load(f)
                if isinstance(curr, dict) and Path(curr.get("init_states", "")).is_dir():
                    needs_write = False
            except Exception:
                pass

        if needs_write:
            dataset_root = os.environ.get("DATASET_ROOT", os.path.expanduser("~/lerobot_datasets/libero_10"))
            cfg = {
                "benchmark_root": str(bench_root),
                "bddl_files": str(bench_root / "bddl_files"),
                "init_states": str(bench_root / "init_files"),
                "datasets": str(Path(dataset_root).resolve()),
                "assets": str(bench_root / "assets"),
            }
            _libero_cfg_dir.mkdir(parents=True, exist_ok=True)
            with open(_libero_cfg_file, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[setup] Updated LIBERO config with verified benchmark root: {_libero_cfg_file}")
            print(f"[setup]   init_states: {bench_root / 'init_files'}")
    except Exception as e:
        print(f"[setup] Warning: LIBERO config validation: {e}")

_ensure_valid_libero_config()

try:
    import libero.libero as libero_pkg
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    import robosuite.utils.binding_utils as bu
    import mujoco

    # Fix for NumPy 2.x & MuJoCo 3.x enum/scalar comparison in robosuite binding_utils
    def _create_patched_qpos(orig_fn):
        def _patched(self, name):
            joint_id = self.joint_name2id(name)
            if joint_id < 0:
                for alt_name in [f"robot0_{name}", name.replace("robot0_", "")]:
                    alt_id = self.joint_name2id(alt_name)
                    if alt_id >= 0:
                        joint_id = alt_id
                        break
            if joint_id < 0:
                raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
            joint_type = int(self.jnt_type[joint_id])
            joint_addr = int(self.jnt_qposadr[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                ndim = 7
            elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                ndim = 4
            else:
                assert joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)), \
                    f"Joint '{name}' (id {joint_id}) has unsupported type {joint_type}"
                ndim = 1

            if ndim == 1:
                return joint_addr
            return (joint_addr, joint_addr + ndim)
        return _patched

    def _create_patched_qvel(orig_fn):
        def _patched(self, name):
            joint_id = self.joint_name2id(name)
            if joint_id < 0:
                for alt_name in [f"robot0_{name}", name.replace("robot0_", "")]:
                    alt_id = self.joint_name2id(alt_name)
                    if alt_id >= 0:
                        joint_id = alt_id
                        break
            if joint_id < 0:
                raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
            joint_type = int(self.jnt_type[joint_id])
            joint_addr = int(self.jnt_dofadr[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                ndim = 6
            elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                ndim = 3
            else:
                assert joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)), \
                    f"Joint '{name}' (id {joint_id}) has unsupported type {joint_type}"
                ndim = 1

            if ndim == 1:
                return joint_addr
            return (joint_addr, joint_addr + ndim)
        return _patched

    for target in [bu, getattr(bu, "MjModel", None), getattr(bu, "MjModelWrapper", None)]:
        if target is not None:
            if hasattr(target, "get_joint_qpos_addr"):
                target.get_joint_qpos_addr = _create_patched_qpos(target.get_joint_qpos_addr)
            if hasattr(target, "get_joint_qvel_addr"):
                target.get_joint_qvel_addr = _create_patched_qvel(target.get_joint_qvel_addr)

    # Fix for MuJoCo 3.10+ where qM was renamed to M in MjData
    class _MArray(np.ndarray):
        """NumPy array view that retains a reference to the MjData struct."""
        _mj_data = None

    def _get_qM(self):
        d = getattr(self, "_data", self)
        raw = getattr(d, "M", None)
        if raw is None:
            raw = getattr(d, "qM", None)
        if raw is None:
            raise AttributeError("'MjData' object has no attribute 'qM' or 'M'")
        arr = np.asarray(raw).view(_MArray)
        arr._mj_data = d
        return arr

    for data_cls in [getattr(bu, "MjData", None), getattr(bu, "MjDataWrapper", None), getattr(mujoco, "MjData", None)]:
        if data_cls is not None:
            try:
                setattr(data_cls, "qM", property(_get_qM))
            except Exception:
                pass

    # Safety wrapper for mj_fullM compatible with robosuite 1.4 + MuJoCo 3.x
    _orig_mj_fullM = mujoco.mj_fullM

    def _unpack_M_fallback(m, dst, M):
        nv = int(m.nv)
        dst.fill(0.0)
        for i in range(nv):
            adr = int(m.dof_Madr[i])
            j = i
            while j >= 0:
                val = float(M[adr])
                dst[i, j] = val
                dst[j, i] = val
                adr += 1
                j = int(m.dof_parentid[j])

    def _patched_mj_fullM(*args, **kwargs):
        if len(args) == 3:
            m, a1, a2 = args
            model = getattr(m, "_model", m)

            # Case 1: Called as mj_fullM(model, data, dst) [Standard MuJoCo 3.x]
            if not isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
                dst = a2
                d = getattr(a1, "_data", a1)
                raw_M = getattr(d, "M", getattr(d, "qM", None))
                if raw_M is not None:
                    return _unpack_M_fallback(model, dst, raw_M)
                try:
                    return _orig_mj_fullM(model, d, dst)
                except Exception:
                    pass

            # Case 2: Called as mj_fullM(model, dst, qM_or_data) [Robosuite 1.4 API]
            if isinstance(a1, np.ndarray):
                dst = a1
                qM_or_data = a2
                raw_M = getattr(qM_or_data, "M", getattr(qM_or_data, "qM", qM_or_data))
                if isinstance(raw_M, np.ndarray):
                    return _unpack_M_fallback(model, dst, raw_M)
                d = getattr(qM_or_data, "_mj_data", getattr(qM_or_data, "_data", None))
                if d is not None:
                    raw_M = getattr(d, "M", getattr(d, "qM", None))
                    if raw_M is not None:
                        return _unpack_M_fallback(model, dst, raw_M)
                if raw_M is not None:
                    return _unpack_M_fallback(model, dst, raw_M)

        elif len(args) == 2:
            m, a1 = args
            model = getattr(m, "_model", m)
            if hasattr(a1, "_mj_data") or hasattr(a1, "_data") or hasattr(a1, "qpos"):
                d = getattr(a1, "_mj_data", getattr(a1, "_data", a1))
                return _orig_mj_fullM(model, d)

        return _orig_mj_fullM(*args, **kwargs)

    mujoco.mj_fullM = _patched_mj_fullM
except ImportError as _err:
    print("=" * 80)
    print(f"⚠️ ERROR: Missing simulation dependency ({_err}).")
    print("Please install the LIBERO simulation package in your environment:")
    print("  conda activate lerobot_env")
    print("  pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git")
    print("  pip install robosuite bddl easydict")
    print("=" * 80)
    sys.exit(1)


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



def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

def extract_success(info, env=None):
    if env is not None:
        if hasattr(env, "check_success"):
            try:
                if env.check_success():
                    return True
            except Exception:
                pass
        if hasattr(env, "env") and hasattr(env.env, "check_success"):
            try:
                if env.env.check_success():
                    return True
            except Exception:
                pass
    if not isinstance(info, dict):
        return False
    for key in ("success", "is_success", "task_success", "episode_success"):
        if key in info:
            value = info[key]
            if hasattr(value, "item"):
                value = value.item()
            if bool(value):
                return True
    return False

def evaluate_task(task_id, seed, policy, preprocessor, postprocessor, 
                  num_episodes=10, max_steps=520, device="cuda", benchmark_name="libero_10",
                  run_output_dir=None):
    """
    Evaluates the LoRA-adapted policy on a single LIBERO task.
    Supports resuming mid-evaluation from saved eval_progress.json.
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
    start_ep = 0
    progress_file = None

    if run_output_dir is not None:
        os.makedirs(run_output_dir, exist_ok=True)
        progress_file = os.path.join(run_output_dir, "eval_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    saved_prog = json.load(f)
                if saved_prog.get("task_id") == task_id and saved_prog.get("seed") == seed:
                    saved_episodes = saved_prog.get("episode_details", [])
                    if 0 < len(saved_episodes) < num_episodes:
                        results = saved_prog
                        results["num_episodes"] = num_episodes
                        start_ep = len(saved_episodes)
                        total_steps_successful = sum(d["steps"] for d in saved_episodes if d.get("success", False))
                        total_steps_run = sum(d["steps"] for d in saved_episodes)
                        print(f"--- [Resume] Found existing progress ({start_ep}/{num_episodes} episodes completed, {results['num_successful']} successful). Resuming from Episode {start_ep + 1}... ---", flush=True)
            except Exception as e:
                print(f"Warning: Could not read progress file {progress_file}: {e}. Starting from episode 1.", flush=True)

    init_states = benchmark.get_task_init_states(task_id)

    for ep in range(start_ep, num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}...", flush=True)
        print("  Initializing fresh environment for episode...", flush=True)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_path,
            camera_heights=256,
            camera_widths=256,
            ignore_done=True,
        )
        try:
            obs = env.reset()
            if init_states is not None and len(init_states) > ep:
                print("  Applying task initial state...", flush=True)
                env.set_init_state(init_states[ep])
                if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
                    obs = env.env._get_observations()
                elif hasattr(env, "_get_observations"):
                    obs = env._get_observations()

            # Warmup environment
            print("  Warming up environment with dummy actions...", flush=True)
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
                if extract_success(info, env=env) or reward > 0.9:
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

            # Save atomic intermediate progress after every episode
            if progress_file is not None:
                try:
                    temp_progress = progress_file + ".tmp"
                    with open(temp_progress, "w") as f:
                        json.dump(results, f, indent=2)
                    os.replace(temp_progress, progress_file)
                except Exception as e:
                    print(f"Warning: Could not save progress: {e}", flush=True)

        finally:
            try:
                env.close()
            except Exception:
                pass
            del env
            gc.collect()
            torch.cuda.empty_cache()
    
    success_rate = results["num_successful"] / num_episodes
    avg_steps = total_steps_successful / results["num_successful"] if results["num_successful"] > 0 else float(max_steps)
    
    results["success_rate"] = success_rate
    results["avg_steps_to_success"] = avg_steps
    results["completed"] = True

    # Finalize progress file
    if progress_file is not None:
        try:
            with open(progress_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass
    
    print(f"Task Results: Success Rate = {success_rate:.2%}, Avg Steps = {avg_steps:.1f}", flush=True)
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy with task-specific LoRA adapters")
    parser.add_argument("--task_id", type=int, default=None, help="Evaluate a single task (0-9)")
    parser.add_argument("--seed", type=int, default=None, help="Evaluate a single seed (0-2)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of seeds to evaluate (e.g. --seeds 0 1 2)")
    parser.add_argument("--run_all", action="store_true", help="Evaluate all 10 tasks and 3 seeds")
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to checkpoints root directory or specific task adapter folder")
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
    del base_policy
    gc.collect()
    torch.cuda.empty_cache()

    # Define tasks and seeds to run
    task_ids = list(range(10)) if (args.run_all or args.task_id is None) else [args.task_id]
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [0, 1, 2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_runs = []

    for task_id in task_ids:
        # Load clean policy for this task to avoid nested PeftModel adapters
        print(f"\nInitializing policy for Task {task_id}...")
        active_policy = SmolVLAPolicy.from_pretrained(policy_name).to(device)
        active_policy.eval()
        if hasattr(active_policy, "model") and hasattr(active_policy.model, "vlm_with_expert"):
            if hasattr(active_policy.model.vlm_with_expert, "config"):
                active_policy.model.vlm_with_expert.config._attn_implementation = "eager"

        # Load task-specific LoRA weights if provided
        if args.lora_dir is not None:
            candidate_paths = [
                os.path.join(args.lora_dir, f"task_{task_id}"),
                os.path.join(args.lora_dir, f"task_{task_id}", "best"),
                args.lora_dir,
            ]
            saved_adapter_path = None
            for cp in candidate_paths:
                if os.path.exists(os.path.join(cp, "adapter_config.json")) or os.path.exists(os.path.join(cp, "adapter_model.safetensors")):
                    saved_adapter_path = cp
                    break
            
            if saved_adapter_path is not None:
                print(f"Loading LoRA adapters for task {task_id} from {saved_adapter_path}...")
                active_policy.model.vlm_with_expert.lm_expert = PeftModel.from_pretrained(
                    active_policy.model.vlm_with_expert.lm_expert,
                    saved_adapter_path
                ).to(device)
                target_dtype = next(active_policy.model.vlm_with_expert.vlm.parameters()).dtype
                active_policy.model.vlm_with_expert.lm_expert.to(target_dtype)
                
                # Check for and load custom action heads if present
                action_heads_file = os.path.join(saved_adapter_path, "action_heads.pt")
                if os.path.exists(action_heads_file):
                    print(f"Loading custom action heads for task {task_id} from {action_heads_file}...")
                    heads_dict = torch.load(action_heads_file, map_location=device, weights_only=False)
                    for head_name in ["action_out_proj", "action_time_mlp_in", "action_time_mlp_out", "action_in_proj"]:
                        if head_name in heads_dict and hasattr(active_policy.model, head_name):
                            getattr(active_policy.model, head_name).load_state_dict(heads_dict[head_name])
                            getattr(active_policy.model, head_name).to(target_dtype)
                active_policy.eval()
            else:
                print(f"[Warning] LoRA checkpoint not found for task {task_id} in '{args.lora_dir}'. Running base policy.")

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
                benchmark_name=args.benchmark,
                run_output_dir=run_output_dir,
            )
            
            # Save individual run report
            report_path = os.path.join(run_output_dir, f"evaluation_report_{timestamp}.json")
            with open(report_path, "w") as f:
                json.dump(run_result, f, indent=2)
            
            task_runs.append(run_result)
            all_runs.append(run_result)
            
            # Clean up memory and cache before next seed
            gc.collect()
            torch.cuda.empty_cache()

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
    try:
        from linux_inhibit import LinuxInhibit
        with LinuxInhibit(reason="SmolVLA LoRA Evaluation"):
            main()
    except ImportError:
        main()
