import faulthandler
faulthandler.enable()

import sys
import json
import wandb

import argparse
import os
from pathlib import Path

# Set headless MuJoCo / PyOpenGL rendering backends
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import torch

# Fix for PyTorch 2.6+ weights_only=True default when loading libero init states
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Fix for PyAV 15+ missing av.option.Option type hint in LeRobot
try:
    import av
    import types
    if not hasattr(av, "option"):
        av.option = types.SimpleNamespace(Option=object)
except Exception:
    pass

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
            for c in candidates:
                if (c / "init_files").is_dir():
                    bench_root = c
                    break

        if bench_root is None:
            bench_root = Path(os.path.expanduser("~/LIBERO/libero/libero"))

        needs_write = True
        if _libero_cfg_file.exists():
            try:
                import yaml
                with open(_libero_cfg_file, "r") as f:
                    curr = yaml.safe_load(f)
                if isinstance(curr, dict) and Path(curr.get("init_states", "")).is_dir():
                    needs_write = False
            except Exception:
                pass

        if needs_write:
            import yaml
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
            print(f"[setup] Verified LIBERO config: {_libero_cfg_file}")
    except Exception as e:
        print(f"[setup] Notice: LIBERO config validation: {e}")

_ensure_valid_libero_config()

# Robosuite / NumPy 2.x & MuJoCo 3.x compatibility patch
try:
    import robosuite.utils.binding_utils as bu
    import mujoco
    import numpy as np

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

        elif len(args) == 2:
            m, a1 = args
            model = getattr(m, "_model", m)
            if hasattr(a1, "_mj_data") or hasattr(a1, "_data") or hasattr(a1, "qpos"):
                d = getattr(a1, "_mj_data", getattr(a1, "_data", a1))
                return _orig_mj_fullM(model, d)

        return _orig_mj_fullM(*args, **kwargs)

    mujoco.mj_fullM = _patched_mj_fullM
except Exception as _patch_err:
    print(f"[Warning] Failed to patch robosuite / MuJoCo binding_utils: {_patch_err}")


import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Iterable
import re
from pathlib import Path

# Pre-import transformers to prevent sys.path modifications from causing import_utils.py generator TypeErrors
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_diffusion_agent import HybridFrozenBrainDiffusionHands
from linux_inhibit import LinuxInhibit

# LIBERO imports for evaluation
try:
    import libero.libero as libero_pkg
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    has_libero = True
except ImportError:
    has_libero = False
    print("Warning: Could not import LIBERO. Evaluation loops will be skipped.")

import sys
sys.path.insert(0, "/home/swagat/lerobot/src")

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Warning: Could not import LeRobotDataset from lerobot.")

try:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    has_lerobot_processors = True
except ImportError:
    has_lerobot_processors = False
    print("Warning: Could not import LeRobot pre/post processors or env factory.")

from torch.utils.data import Dataset

class HDF5LiberoDataset(Dataset):
    """Dataset for training directly on local LIBERO HDF5 demonstration files."""
    def __init__(self, data_dir, chunk_size=16, benchmark_name="libero_10", train_task_ids=None):
        import h5py
        self.chunk_size = chunk_size
        all_file_paths = sorted(list(Path(data_dir).rglob("*.h5")) + list(Path(data_dir).rglob("*.hdf5")))
        if not all_file_paths:
            raise ValueError(f"No .hdf5 / .h5 demonstration files found in '{data_dir}'.")
        
        task_name_to_instr = {}
        task_id_to_name = {}
        try:
            from libero.libero.benchmark import get_benchmark_dict
            benchmark = get_benchmark_dict()[benchmark_name]()
            for idx, t in enumerate(benchmark.tasks):
                task_name_to_instr[t.name] = t.language
                task_id_to_name[idx] = t.name
        except Exception:
            pass

        if train_task_ids is not None:
            target_task_names = [task_id_to_name[tid] for tid in train_task_ids if tid in task_id_to_name]
            self.file_paths = [
                f for f in all_file_paths
                if any(tname in f.name for tname in target_task_names)
            ]
            print(f"[HDF5 Dataset] Filtered to task(s) {train_task_ids} ({target_task_names}): found {len(self.file_paths)} demo file(s).")
            if not self.file_paths:
                print(f"[HDF5 Dataset] Warning: No files matched task IDs {train_task_ids}. Falling back to all {len(all_file_paths)} files.")
                self.file_paths = all_file_paths
        else:
            self.file_paths = all_file_paths

        self.samples = []
        for f_path in self.file_paths:
            try:
                with h5py.File(f_path, 'r') as h5:
                    if "data" not in h5:
                        continue
                    demo_keys = sorted(list(h5["data"].keys()))
                    for dk in demo_keys:
                        prefix = f"data/{dk}"
                        agent_rgb = h5[f"{prefix}/obs/agentview_rgb"][:]
                        # Orientation: flip both H and W to align with SmolVLA and OffScreenRenderEnv
                        agent_rgb = agent_rgb[:, ::-1, ::-1, :].copy()
                        
                        wrist_rgb = h5[f"{prefix}/obs/eye_in_hand_rgb"][:]
                        wrist_rgb = wrist_rgb[:, ::-1, ::-1, :].copy()
                        
                        ee_pos = h5[f"{prefix}/obs/ee_pos"][:]
                        ee_ori = h5[f"{prefix}/obs/ee_ori"][:]
                        gripper = h5[f"{prefix}/obs/gripper_states"][:]
                        states = np.concatenate([ee_pos, ee_ori, gripper], axis=-1).astype(np.float32)
                        actions = h5[f"{prefix}/actions"][:].astype(np.float32)

                        instr = ""
                        for tname, ti in task_name_to_instr.items():
                            if tname in f_path.name:
                                instr = ti
                                break
                        if not instr:
                            fname = f_path.stem.replace("_demo", "")
                            instr = fname.split("_put_")[-1] if "_put_" in fname else fname

                        n_steps = actions.shape[0]
                        for step_idx in range(n_steps):
                            chunk_len = min(chunk_size, n_steps - step_idx)
                            action_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
                            action_chunk[:chunk_len] = actions[step_idx : step_idx + chunk_len]
                            if chunk_len < chunk_size:
                                action_chunk[chunk_len:] = actions[-1]
                                
                            self.samples.append({
                                "img_agent": agent_rgb[step_idx],
                                "img_wrist": wrist_rgb[step_idx],
                                "agent_pos": states[step_idx],
                                "action": action_chunk,
                                "task": instr,
                            })
            except Exception as e:
                print(f"Warning: Failed to load {f_path}: {e}")

        print(f"[HDF5 Dataset] Initialized with {len(self.samples)} samples from {len(self.file_paths)} demo files in '{data_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def hdf5_collate_fn(batch):
    img1_np = np.stack([item["img_agent"] for item in batch], axis=0)
    img2_np = np.stack([item["img_wrist"] for item in batch], axis=0)
    state_np = np.stack([item["agent_pos"] for item in batch], axis=0)

    img_agent = torch.from_numpy(img1_np).permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0
    img_wrist = torch.from_numpy(img2_np).permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0

    if img_agent.shape[-2:] != (256, 256):
        img_agent = torch.nn.functional.interpolate(img_agent, size=(256, 256), mode="bilinear", align_corners=False)
    if img_wrist.shape[-2:] != (256, 256):
        img_wrist = torch.nn.functional.interpolate(img_wrist, size=(256, 256), mode="bilinear", align_corners=False)

    state = torch.from_numpy(state_np).contiguous().to(dtype=torch.float32)
    action = torch.stack([torch.from_numpy(item["action"]) for item in batch], dim=0).contiguous().to(dtype=torch.float32)
    tasks = [item["task"] for item in batch]

    return {
        "observation.images.image": img_agent,
        "observation.images.image2": img_wrist,
        "observation.state": state,
        "action": action,
        "task": tasks,
    }


def clear_preprocessor_state(preprocessor):
    """Clear intermediate EnvTransition references stored in LeRobot processor steps to prevent host RAM memory leaks."""
    if preprocessor is None:
        return
    if hasattr(preprocessor, "steps"):
        for step in preprocessor.steps:
            if hasattr(step, "_current_transition"):
                step._current_transition = None
    if hasattr(preprocessor, "reset"):
        preprocessor.reset()


def clear_video_decoder_cache():
    """Clear cached VideoDecoder C++ instances and close file handles in lerobot to prevent memory leaks and segfaults."""
    try:
        import lerobot.datasets.video_utils as vu
        if hasattr(vu, "_default_decoder_cache"):
            vu._default_decoder_cache.clear()
    except Exception:
        pass
    import gc
    gc.collect()



def extract_success(info, env=None):
    """Best-effort extraction of task success from env info."""
    if env is not None and hasattr(env, "check_success"):
        try:
            return bool(env.check_success())
        except Exception:
            pass
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success", "episode_success"):
            if key in info:
                val = info[key]
                return bool(val.item() if hasattr(val, "item") else val)
    return False


def set_base_policy_num_steps(base_policy, num_steps: int):
    if base_policy is None:
        return
    if hasattr(base_policy, "config"):
        base_policy.config.num_steps = num_steps
    if hasattr(base_policy, "model") and hasattr(base_policy.model, "config"):
        base_policy.model.config.num_steps = num_steps


def get_libero_dummy_action():
    # LeRobot LIBERO warmup no-op.
    return [0, 0, 0, 0, 0, 0, -1]


def build_policy_state(obs):
    """Construct SmolVLA-compatible 8D state from LIBERO observations."""
    # Preferred layout used by SmolVLA eval: eef_pos (3) + eef_axis_angle (3) + gripper_qpos (2)
    if all(k in obs for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")):
        eef_pos = obs["robot0_eef_pos"]
        eef_axis_angle = quat2axisangle(obs["robot0_eef_quat"])
        gripper_qpos = obs["robot0_gripper_qpos"]
        return np.concatenate([eef_pos, eef_axis_angle, gripper_qpos]).astype(np.float32)

    # Fallback path if the env exposes only joint states.
    joint_pos = obs["robot0_joint_pos"]
    gripper_pos = obs["robot0_gripper_qpos"][0:1]
    return np.concatenate([joint_pos, gripper_pos]).astype(np.float32)


def preprocess_policy_images(obs, device, flip_mode="vertical"):
    """Prepare agent and wrist images with a configurable flip convention."""
    img_agent = obs["agentview_image"].copy()
    img_wrist = obs["robot0_eye_in_hand_image"].copy()

    if flip_mode == "vertical":
        img_agent = img_agent[::-1, :, :].copy()
        img_wrist = img_wrist[::-1, :, :].copy()
    elif flip_mode == "vertical_horizontal":
        img_agent = img_agent[::-1, ::-1, :].copy()
        img_wrist = img_wrist[::-1, ::-1, :].copy()
    elif flip_mode != "none":
        raise ValueError(f"Unknown flip_mode: {flip_mode}")

    img_tensor_agent = torch.from_numpy(img_agent).to(torch.float32)
    if img_tensor_agent.max() > 1.0:
        img_tensor_agent /= 255.0
    img_tensor_agent = img_tensor_agent.permute(2, 0, 1).unsqueeze(0).to(device)

    img_tensor_wrist = torch.from_numpy(img_wrist).to(torch.float32)
    if img_tensor_wrist.max() > 1.0:
        img_tensor_wrist /= 255.0
    img_tensor_wrist = img_tensor_wrist.permute(2, 0, 1).unsqueeze(0).to(device)

    return img_tensor_agent, img_tensor_wrist


def get_policy_obs_from_env_obs(obs, flip_mode="vertical"):
    """Build raw policy observation dict matching LeRobot eval conventions."""
    img_agent = obs["agentview_image"].copy()
    img_wrist = obs["robot0_eye_in_hand_image"].copy()

    if flip_mode == "vertical":
        img_agent = img_agent[::-1, :, :].copy()
        img_wrist = img_wrist[::-1, :, :].copy()
    elif flip_mode == "vertical_horizontal":
        img_agent = img_agent[::-1, ::-1, :].copy()
        img_wrist = img_wrist[::-1, ::-1, :].copy()
    elif flip_mode != "none":
        raise ValueError(f"Unknown flip_mode: {flip_mode}")

    state_np = build_policy_state(obs)
    return {
        "pixels": {
            "image": img_agent,
            "image2": img_wrist,
        },
        "agent_pos": state_np.astype(np.float32),
    }


def parse_task_id_tokens(task_id_tokens):
    """Parse task IDs from flexible CLI formats like `1 3 5`, `1,3,5`, or `[1, 3, 5]`."""
    if task_id_tokens is None:
        return [0, 1, 2]

    merged = " ".join(task_id_tokens)
    ids = [int(x) for x in re.findall(r"-?\d+", merged)]
    if not ids:
        raise argparse.ArgumentTypeError(
            "--eval_task_ids must include at least one integer task id (e.g. '1 3 5' or '[1, 3, 5]')."
        )

    # Remove duplicates while preserving order.
    seen = set()
    unique_ids = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            unique_ids.append(tid)
    return unique_ids

def create_libero_env(task_id, benchmark, benchmark_root):
    task = benchmark.get_task(task_id)
    bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
    )
    for robot in env.robots:
        if hasattr(robot, "controller") and robot.controller is not None:
            robot.controller.use_delta = True
    return env




def evaluate_in_environment(
    model,
    device,
    epoch,
    tasks: Iterable[int] = range(10),
    num_episodes=2,
    max_steps=520,
    global_step=None,
    image_flip_mode="vertical_horizontal",
    preprocessor=None,
    postprocessor=None,
    policy_mode="hybrid",
    hybrid_mix=0.0,
    residual_alpha=0.1,
    diffusion_steps=10,
    eval_replan_each_step=False,
    eval_log_action_stats_every=0,
    eval_action_clip=1.0,
    eval_output_dir=None,
):
    """Run an evaluation loop in the LIBERO environment to measure true success rate."""
    if not has_libero:
        print("[evaluate_in_environment] Skipped because LIBERO is unavailable.")
        return None

    benchmark = get_benchmark_dict()["libero_10"]()
    benchmark_root = os.path.dirname(libero_pkg.__file__)

    model.eval()
    if hasattr(model, "base_policy"):
        set_base_policy_num_steps(model.base_policy, diffusion_steps)

    all_tasks_success = []
    all_episode_details = []

    # Check if LeRobot make_env is available
    use_lerobot_env = False
    if has_lerobot_processors:
        try:
            from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
            use_lerobot_env = True
        except ImportError:
            use_lerobot_env = False

    for task_id in tasks:
        task = benchmark.get_task(task_id)
        success_count = 0
        total_reward = 0.0

        print(f"\n--- Running Evaluation on Task {task_id}: {task.language} ---", flush=True)

        if use_lerobot_env:
            env_cfg = make_env_config(
                "libero",
                task="libero_10",
                task_ids=[task_id],
                episode_length=max_steps,
                observation_height=256,
                observation_width=256,
                obs_type="pixels_agent_pos",
            )
            policy_cfg = getattr(getattr(model, "base_policy", model), "config", None)
            task_env_pre, task_env_post = make_env_pre_post_processors(env_cfg, policy_cfg)
            env_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
            vec_env = env_dict["libero_10"][task_id]
            task_desc = list(vec_env.call("task_description"))[0]

            try:
                for ep in range(num_episodes):
                    print(f"Starting Episode {ep + 1}/{num_episodes} (Task {task_id}, seed={ep})...", flush=True)
                    if hasattr(model, "base_policy") and hasattr(model.base_policy, "reset"):
                        model.base_policy.reset()
                    elif hasattr(model, "reset"):
                        model.reset()

                    obs_dict, info = vec_env.reset(seed=[ep])
                    step = 0
                    done = False
                    ep_reward = 0.0
                    ep_success = False
                    pending_chunk = None
                    chunk_idx = 0

                    while not done and step < max_steps:
                        obs = preprocess_observation(obs_dict)
                        obs["task"] = [task_desc]
                        obs = task_env_pre(obs)
                        batch = preprocessor(obs) if preprocessor is not None else obs

                        with torch.inference_mode():
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                if policy_mode == "base":
                                    base_action = model.base_policy.select_action(batch)
                                    if eval_log_action_stats_every > 0 and (step % eval_log_action_stats_every == 0):
                                        b = base_action.detach().to(torch.float32)
                                        print(
                                            f"[Eval Action Stats][task={task_id} ep={ep+1} step={step}] "
                                            f"base(min={b.min().item():.3f}, max={b.max().item():.3f}, mean={b.mean().item():.3f}, std={b.std(unbiased=False).item():.3f})"
                                        )
                                    chosen_action = base_action
                                else:
                                    if eval_replan_each_step:
                                        pending_chunk = None
                                        chunk_idx = 0

                                    if pending_chunk is None or chunk_idx >= pending_chunk.shape[1]:
                                        pending_chunk = model.select_action(
                                            batch,
                                            steps=diffusion_steps,
                                            return_intermediates=False,
                                        )
                                        chunk_idx = 0

                                    hybrid_action = pending_chunk[:, chunk_idx, :]
                                    chunk_idx += 1
                                    hybrid_action = torch.clamp(hybrid_action, -eval_action_clip, eval_action_clip)

                                    if policy_mode == "mixed":
                                        base_action = model.base_policy.select_action(batch)
                                        chosen_action = (1.0 - hybrid_mix) * base_action + hybrid_mix * hybrid_action
                                        chosen_action = torch.clamp(chosen_action, -eval_action_clip, eval_action_clip)
                                    elif policy_mode == "residual":
                                        base_action = model.base_policy.select_action(batch)
                                        chosen_action = base_action + residual_alpha * hybrid_action
                                        chosen_action = torch.clamp(chosen_action, -eval_action_clip, eval_action_clip)
                                    else:
                                        chosen_action = hybrid_action

                                if postprocessor is not None:
                                    env_action = postprocessor(chosen_action)
                                else:
                                    env_action = chosen_action

                                if task_env_post is not None:
                                    action_transition = task_env_post({"action": env_action})
                                    action_to_step = action_transition["action"]
                                else:
                                    action_to_step = env_action

                                action_np = action_to_step.detach().cpu().to(torch.float32).numpy()
                                if action_np.ndim == 1:
                                    action_np = np.expand_dims(action_np, 0)

                        obs_dict, reward, terminated, truncated, info = vec_env.step(action_np)

                        step_success = False
                        if "final_info" in info:
                            fi = info["final_info"]
                            if isinstance(fi, dict) and "is_success" in fi:
                                s = fi["is_success"]
                                step_success = bool(s[0] if hasattr(s, "__getitem__") else s)
                            elif isinstance(fi, (list, tuple, np.ndarray)) and len(fi) > 0:
                                item = fi[0]
                                if isinstance(item, dict) and "is_success" in item:
                                    step_success = bool(item["is_success"])
                        elif "is_success" in info:
                            s = info["is_success"]
                            step_success = bool(s[0] if hasattr(s, "__getitem__") else s)

                        is_term = bool(terminated[0] if hasattr(terminated, "__getitem__") else terminated)
                        is_trunc = bool(truncated[0] if hasattr(truncated, "__getitem__") else truncated)
                        done = bool(is_term or is_trunc or step_success)
                        ep_success = ep_success or step_success
                        ep_reward += float(reward[0] if hasattr(reward, "__getitem__") else reward)
                        step += 1

                    clear_preprocessor_state(preprocessor)
                    success_count += int(ep_success)
                    total_reward += ep_reward
                    print(f"Eval Ep {ep+1}/{num_episodes} | Success: {ep_success} | Reward: {ep_reward:.2f} | Steps: {step}", flush=True)

                    all_episode_details.append({
                        "task_id": int(task_id),
                        "task_name": getattr(task, "name", f"task_{task_id}"),
                        "language": getattr(task, "language", ""),
                        "episode": ep + 1,
                        "success": bool(ep_success),
                        "reward": float(ep_reward),
                        "steps": int(step),
                    })
            finally:
                try:
                    vec_env.close()
                except Exception:
                    pass
                del vec_env
                clear_preprocessor_state(preprocessor)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            # Fallback legacy path if make_env is not available
            init_states = benchmark.get_task_init_states(task_id)
            for ep in range(num_episodes):
                print(f"Starting Episode {ep + 1}/{num_episodes} (Task {task_id})...", flush=True)
                env = create_libero_env(task_id, benchmark, benchmark_root)
                try:
                    obs = env.reset()
                    if init_states is not None and len(init_states) > 0:
                        init_state = init_states[ep % len(init_states)]
                        env.set_init_state(init_state)
                        if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
                            obs = env.env._get_observations()
                        elif hasattr(env, "_get_observations"):
                            obs = env._get_observations()
                    else:
                        obs = env.reset()

                    for _ in range(10):
                        obs, _, _, _ = env.step(get_libero_dummy_action())

                    for robot in env.robots:
                        if hasattr(robot, "controller") and robot.controller is not None:
                            robot.controller.use_delta = True

                    if hasattr(model, "base_policy") and hasattr(model.base_policy, "reset"):
                        model.base_policy.reset()
                    elif hasattr(model, "reset"):
                        model.reset()

                    step = 0
                    done = False
                    ep_reward = 0.0
                    ep_success = False
                    pending_chunk = None
                    chunk_idx = 0

                    while not done and step < max_steps:
                        current_instruction = task.language
                        raw_obs = get_policy_obs_from_env_obs(obs, flip_mode=image_flip_mode)

                        if preprocessor is not None and postprocessor is not None:
                            policy_obs = preprocess_observation(raw_obs)
                            policy_obs["task"] = [current_instruction]
                            batch = preprocessor(policy_obs)
                        else:
                            img_tensor_agent, img_tensor_wrist = preprocess_policy_images(
                                obs,
                                device,
                                flip_mode=image_flip_mode,
                            )
                            state_np = build_policy_state(obs)
                            state_tensor = torch.from_numpy(state_np).to(torch.float32).unsqueeze(0).to(device)

                            processor = model.base_policy.model.vlm_with_expert.processor
                            text_out = processor(text=current_instruction, return_tensors='pt')

                            batch = {
                                'observation.images.image': img_tensor_agent,
                                'observation.images.image2': img_tensor_wrist,
                                'observation.state': state_tensor,
                                'observation.language.tokens': text_out['input_ids'].to(device),
                                'observation.language.attention_mask': text_out['attention_mask'].to(device).bool(),
                                'language_instruction': [current_instruction]
                            }

                        with torch.inference_mode():
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                if policy_mode == "base":
                                    base_action = model.base_policy.select_action(batch)
                                    chosen_action = base_action
                                else:
                                    if eval_replan_each_step:
                                        pending_chunk = None
                                        chunk_idx = 0

                                    if pending_chunk is None or chunk_idx >= pending_chunk.shape[1]:
                                        pending_chunk = model.select_action(
                                            batch,
                                            steps=diffusion_steps,
                                            return_intermediates=False,
                                        )
                                        chunk_idx = 0

                                    hybrid_action = pending_chunk[:, chunk_idx, :]
                                    chunk_idx += 1
                                    hybrid_action = torch.clamp(hybrid_action, -eval_action_clip, eval_action_clip)

                                    if policy_mode == "mixed":
                                        base_action = model.base_policy.select_action(batch)
                                        chosen_action = (1.0 - hybrid_mix) * base_action + hybrid_mix * hybrid_action
                                        chosen_action = torch.clamp(chosen_action, -eval_action_clip, eval_action_clip)
                                    elif policy_mode == "residual":
                                        base_action = model.base_policy.select_action(batch)
                                        chosen_action = base_action + residual_alpha * hybrid_action
                                        chosen_action = torch.clamp(chosen_action, -eval_action_clip, eval_action_clip)
                                    else:
                                        chosen_action = hybrid_action

                                if postprocessor is not None:
                                    env_action = postprocessor(chosen_action)
                                    action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]
                                else:
                                    action_np = chosen_action[0].detach().cpu().to(torch.float32).numpy()
                                    action_np = np.clip(action_np, -1.0, 1.0)

                        next_obs, reward, done, info = env.step(action_np)
                        step_success = extract_success(info, env)
                        done = bool(done or step_success)
                        ep_success = ep_success or step_success
                        ep_reward += float(reward)
                        obs = next_obs
                        step += 1

                    success_count += int(ep_success)
                    total_reward += ep_reward
                    print(f"Eval Ep {ep+1}/{num_episodes} | Success: {ep_success} | Reward: {ep_reward:.2f} | Steps: {step}", flush=True)

                    all_episode_details.append({
                        "task_id": int(task_id),
                        "task_name": getattr(task, "name", f"task_{task_id}"),
                        "language": getattr(task, "language", ""),
                        "episode": ep + 1,
                        "success": bool(ep_success),
                        "reward": float(ep_reward),
                        "steps": int(step),
                    })
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass
                    del env
                    clear_preprocessor_state(preprocessor)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        success_rate = success_count / num_episodes
        avg_ep_reward = total_reward / num_episodes
        all_tasks_success.append(success_rate)

        if wandb.run is not None:
            eval_log = {
                f"eval_task/task_{task_id}_success_rate": success_rate,
                f"eval_task/task_{task_id}_reward": avg_ep_reward,
                "epoch": epoch
            }
            if global_step is not None:
                wandb.log(eval_log, step=global_step)
            else:
                wandb.log(eval_log)

    overall_success = sum(all_tasks_success) / len(all_tasks_success) if all_tasks_success else 0.0
    print(f"\nEval Complete | Overall Success Rate: {overall_success:.2f}\n")

    if eval_output_dir is not None:
        os.makedirs(eval_output_dir, exist_ok=True)
        results_payload = {
            "overall_success_rate": float(overall_success),
            "epoch": epoch,
            "policy_mode": policy_mode,
            "residual_alpha": float(residual_alpha),
            "diffusion_steps": int(diffusion_steps),
            "eval_replan_each_step": bool(eval_replan_each_step),
            "tasks": {
                str(tid): float(sr) for tid, sr in zip(tasks, all_tasks_success)
            },
            "episodes": all_episode_details,
        }
        results_json_path = os.path.join(eval_output_dir, "eval_results.json")
        try:
            with open(results_json_path, "w") as f:
                json.dump(results_payload, f, indent=2)
            print(f"[Eval Results] Saved JSON results to: {results_json_path}")
        except Exception as e:
            print(f"Warning: Failed to save eval_results.json: {e}")

        summary_csv_path = os.path.join(eval_output_dir, "eval_summary.csv")
        try:
            with open(summary_csv_path, "w") as f:
                f.write("task_id,task_name,success_rate,avg_reward,num_episodes\n")
                for tid, sr in zip(tasks, all_tasks_success):
                    task_eps = [e for e in all_episode_details if e["task_id"] == tid]
                    t_rew = sum(e["reward"] for e in task_eps) / len(task_eps) if task_eps else 0.0
                    t_name = task_eps[0]["task_name"] if task_eps else f"task_{tid}"
                    f.write(f"{tid},{t_name},{sr:.4f},{t_rew:.4f},{len(task_eps)}\n")
            print(f"[Eval Results] Saved CSV summary to: {summary_csv_path}")
        except Exception as e:
            print(f"Warning: Failed to save eval_summary.csv: {e}")

    if wandb.run is not None:
        overall_log = {
            "eval/success rate": overall_success,
            "epoch": epoch
        }
        if global_step is not None:
            wandb.log(overall_log, step=global_step)
        else:
            wandb.log(overall_log)

    clear_video_decoder_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()

    return {
        "overall_success": float(overall_success),
        "episodes": int(num_episodes),
        "task_metrics": [
            {
                "task_id": int(tid),
                "success_rate": float(sr),
                "avg_reward": float(
                    sum(e["reward"] for e in all_episode_details if e["task_id"] == tid)
                    / max(1, len([e for e in all_episode_details if e["task_id"] == tid]))
                ),
            }
            for tid, sr in zip(tasks, all_tasks_success)
        ],
        "episode_details": all_episode_details,
    }


def run_preflight_baseline_check(
    model,
    device,
    tasks: Iterable[int],
    num_episodes=3,
    max_steps=520,
    image_flip_mode="vertical_horizontal",
    preprocessor=None,
    postprocessor=None,
):
    """Evaluate frozen base SmolVLA before training to establish a task-wise baseline."""
    if not has_libero:
        print("[Preflight] Skipped baseline check because LIBERO is unavailable.")
        return None

    print("\n[Preflight] Running frozen base-policy baseline check...")
    results = evaluate_in_environment(
        model=model,
        device=device,
        epoch=0,
        tasks=tasks,
        num_episodes=num_episodes,
        max_steps=max_steps,
        image_flip_mode=image_flip_mode,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        policy_mode="base",
        diffusion_steps=10,
    )
    return results


def log_preflight_metrics(preflight_result, step=0):
    """Log preflight metrics to W&B after run initialization."""
    if preflight_result is None:
        return

    for item in preflight_result["task_metrics"]:
        task_id = item["task_id"]
        wandb.log({
            f"preflight/task_{task_id}_success_rate": item["success_rate"],
            f"preflight/task_{task_id}_reward": item["avg_reward"],
            "preflight/episodes": preflight_result["episodes"],
        }, step=step)

    wandb.log({
        "preflight/overall_success_rate": preflight_result["overall_success"],
        "preflight/episodes": preflight_result["episodes"],
    }, step=step)


def plot_diffusion_trajectories(model, batch, device, epoch, num_samples=4, global_step=None):
    """
    Generate plots of the diffusion interpolation steps mapped against the ground truth.
    Logs the images sequentially to Weights & Biases.
    """
    model.eval()
    with torch.inference_mode():
        gt_actions = batch["action"].to(device)
        if gt_actions.ndim == 2:
            gt_actions = gt_actions.unsqueeze(1)
        
        # We only visualize the first num_samples to keep it fast
        vis_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v_sub = v[:num_samples]
                if v_sub.is_floating_point():
                    vis_batch[k] = v_sub.to(dtype=torch.float32, device=device)
                else:
                    vis_batch[k] = v_sub.to(device=device)
            elif isinstance(v, list):
                vis_batch[k] = v[:num_samples]
            else:
                vis_batch[k] = v
                
        gt_actions = gt_actions[:num_samples]
        
        # Run inference tracking intermediate noise steps
        final_action, intermediates = model.select_action(vis_batch, steps=10, return_intermediates=True)
            
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            # [chunk_size, action_dim]
            gt = gt_actions[i].cpu().numpy()
            
            # Plot dimension 0 vs 1 as a 2D physical arm trajectory outline
            ax.plot(gt[:, 0], gt[:, 1], 'g-*', label='Ground Truth', linewidth=2)
            
            # Plot intermediates shifting from pure noise to predicted manifold
            colors = plt.cm.Blues(np.linspace(0.3, 1, len(intermediates)))
            for step_idx, interm in enumerate(intermediates):
                interm_cpu = interm[i].cpu().numpy()
                alpha = 0.3 if step_idx < len(intermediates)-1 else 1.0
                ax.plot(
                    interm_cpu[:, 0], interm_cpu[:, 1], 
                    color=colors[step_idx], alpha=alpha, 
                    label='Diffusion Steps' if step_idx == len(intermediates)-1 else ""
                )
            
            ax.set_title(f"Sample {i+1} Trajectory (Action Dim 0 vs 1)")
            ax.legend()
        
        plt.tight_layout()
        vis_log = {"eval/diffusion_trajectories": wandb.Image(fig), "epoch": epoch}
        if global_step is not None:
            wandb.log(vis_log, step=global_step)
        else:
            wandb.log(vis_log)
        plt.close(fig)
    model.train()


def train():
    parser = argparse.ArgumentParser(description="Train Hybrid Frozen Brain Diffusion Hands on LIBERO Dataset.")
    parser.add_argument("--base_policy_path", type=str, required=True, help="Path to the frozen base SmolVLA policy (e.g. huggingface repo or local path)")
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/libero_10", help="HuggingFace repo ID for the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers. Set to 0 to prevent video decoding segfaults.")
    parser.add_argument("--video_backend", type=str, default=None, help="Video backend for LeRobotDataset (e.g. torchcodec, pyav, video_reader). Defaults to None (torchcodec).")
    parser.add_argument("--decoder_cache_clear_freq", type=int, default=500, help="Step frequency to clear video decoder cache and free C++ decoder handles.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the diffusion head")
    parser.add_argument("--chunk_size", type=int, default=16, help="Action chunk size")
    parser.add_argument("--action_dim", type=int, default=7, help="Dimension of the action space")
    parser.add_argument("--cond_dim", type=int, default=960, help="Dimension of the frozen backbone semantic features")
    parser.add_argument("--diff_hidden_dim", type=int, default=256, help="Hidden dimension of the diffusion head")
    parser.add_argument("--diff_layers", type=int, default=5, help="Number of residual blocks in the diffusion head")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--out_dir", type=str, default="checkpoints_hybrid_diffusion", help="Output directory for checkpoints")
    parser.add_argument("--wandb_project", type=str, default="hybrid_diffusion_vla", help="W&B project name")
    parser.add_argument("--vis_freq", type=int, default=5, help="Epoch frequency to visualize diffusion trajectories")
    parser.add_argument("--eval_episodes", type=int, default=2, help="Number of evaluation episodes per task")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and run evaluation only (base/hybrid/mixed) using an existing checkpoint.",
    )
    parser.add_argument(
        "--eval_checkpoint_path",
        type=str,
        default=None,
        help="Optional checkpoint path for eval-only or resume. Defaults to out_dir/latest_checkpoint.pt when available.",
    )
    parser.add_argument(
        "--eval_wandb_run_id",
        type=str,
        default=None,
        help="Optional WandB run id used in eval-only mode. If set, eval-only logs resume into the same run id.",
    )
    parser.add_argument(
        "--eval_wandb_resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto"],
        help="WandB resume policy used with --eval_wandb_run_id in eval-only mode.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Optional WandB run id for training mode. Useful for external resume orchestration.",
    )
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto"],
        help="WandB resume policy used with --wandb_run_id in training mode.",
    )
    parser.add_argument(
        "--wandb_run_id_file",
        type=str,
        default=None,
        help="If set, writes the actual WandB run id used in training mode to this file.",
    )
    parser.add_argument("--preflight_baseline_check", action="store_true", help="Run frozen-base policy baseline evaluation before training")
    parser.add_argument("--preflight_episodes", type=int, default=3, help="Episodes per task for preflight baseline evaluation")
    parser.add_argument(
        "--min_baseline_success",
        type=float,
        default=None,
        help="If set, start wandb/training only when preflight overall baseline success >= this threshold (0-1).",
    )
    parser.add_argument(
        "--preflight_task_ids",
        nargs="+",
        default=None,
        help="Task IDs for preflight baseline check. Defaults to eval_task_ids if omitted.",
    )
    parser.add_argument(
        "--image_flip_mode",
        type=str,
        default="vertical_horizontal",
        choices=["vertical", "vertical_horizontal", "none"],
        help="Image flip convention for preflight/eval observation preprocessing.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 2 training batches and 1 eval episode to verify execution sanity.",
    )
    parser.add_argument(
        "--train_task_ids",
        nargs="+",
        default=None,
        help="Optional task IDs to train on (e.g. 0 or 0 1 2). If omitted, trains on all 10 tasks.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Convenience alias to train and evaluate on a single task (0-9). Overrides both --train_task_ids and --eval_task_ids.",
    )
    parser.add_argument(
        "--eval_task_ids",
        nargs="+",
        default=["0", "1", "2"],
        help="Task IDs to evaluate. Accepts formats like: 1 3 5, 1,3,5, or [1, 3, 5]",
    )
    parser.add_argument(
        "--eval_policy_mode",
        type=str,
        default="hybrid",
        choices=["base", "hybrid", "mixed", "residual"],
        help="Policy used during eval: base SmolVLA, hybrid diffusion-only, blended mixed mode, or residual mode.",
    )
    parser.add_argument(
        "--eval_hybrid_mix",
        type=float,
        default=0.3,
        help="When --eval_policy_mode=mixed, fraction for hybrid action in blended action.",
    )
    parser.add_argument(
        "--eval_diffusion_steps",
        type=int,
        default=10,
        help="Number of diffusion integration steps used during eval chunk generation.",
    )
    parser.add_argument(
        "--eval_residual_alpha",
        type=float,
        default=0.1,
        help="When --eval_policy_mode=residual, scale factor for hybrid residual added to base action.",
    )
    parser.add_argument(
        "--eval_replan_each_step",
        action="store_true",
        help="If set, regenerate a fresh diffusion action chunk every env step during eval.",
    )
    parser.add_argument(
        "--eval_max_steps",
        type=int,
        default=520,
        help="Maximum rollout horizon per episode during preflight and eval (default: 520).",
    )
    parser.add_argument(
        "--eval_log_action_stats_every",
        type=int,
        default=0,
        help="If >0, print action min/max/mean/std every N eval steps (0 disables).",
    )
    parser.add_argument(
        "--eval_action_clip",
        type=float,
        default=1.0,
        help="Clamp hybrid/mixed normalized actions to [-clip, clip] before postprocessor.",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default=None,
        help="Directory to store evaluation results (eval_results.json, eval_summary.csv, eval_config.json).",
    )
    parser.add_argument(
        "--residual_target",
        action="store_true",
        help="Train diffusion head on residual deltas (dataset action minus frozen base action).",
    )
    parser.add_argument(
        "--delta_l2_weight",
        type=float,
        default=0.0,
        help="Optional L2 regularization weight on predicted residual magnitude during residual-target training.",
    )
    
    args = parser.parse_args()
    if args.task_id is not None:
        args.train_task_ids = [args.task_id]
        args.eval_task_ids = [str(args.task_id)]
        args.preflight_task_ids = [str(args.task_id)]
    elif args.train_task_ids is not None:
        args.train_task_ids = parse_task_id_tokens(args.train_task_ids)

    args.eval_task_ids = parse_task_id_tokens(args.eval_task_ids)
    args.preflight_task_ids = parse_task_id_tokens(args.preflight_task_ids) if args.preflight_task_ids is not None else list(args.eval_task_ids)
    if args.min_baseline_success is not None and not (0.0 <= args.min_baseline_success <= 1.0):
        parser.error("--min_baseline_success must be in [0, 1].")
    if not (0.0 <= args.eval_hybrid_mix <= 1.0):
        parser.error("--eval_hybrid_mix must be in [0, 1].")
    if args.eval_diffusion_steps < 1:
        parser.error("--eval_diffusion_steps must be >= 1.")
    if args.eval_residual_alpha < 0.0:
        parser.error("--eval_residual_alpha must be >= 0.")
    if args.eval_log_action_stats_every < 0:
        parser.error("--eval_log_action_stats_every must be >= 0.")
    if args.eval_action_clip <= 0.0:
        parser.error("--eval_action_clip must be > 0.")
    if args.delta_l2_weight < 0.0:
        parser.error("--delta_l2_weight must be >= 0.")
    
    if args.dry_run:
        args.epochs = 1
        args.eval_episodes = 1
        args.eval_task_ids = [args.eval_task_ids[0]] if args.eval_task_ids else [0]
        args.preflight_episodes = 1
        args.preflight_task_ids = [args.eval_task_ids[0]]
        print(f"[Dry Run] dry_run mode enabled: restricting to 1 epoch, 2 batches, 1 eval episode on task {args.eval_task_ids[0]}.")

    os.makedirs(args.out_dir, exist_ok=True)
    
    import uuid
    if not args.eval_only and args.wandb_run_id is not None:
        run_id = args.wandb_run_id
    else:
        try:
            run_id = wandb.util.generate_id()
        except Exception:
            try:
                from wandb.sdk.lib.runid import generate_id
                run_id = generate_id()
            except Exception:
                run_id = uuid.uuid4().hex[:8]

    start_epoch = 0
    checkpoint = None
    latest_ckpt_path = os.path.join(args.out_dir, "latest_checkpoint.pt")
    
    checkpoint_path = args.eval_checkpoint_path or latest_ckpt_path
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        start_epoch = checkpoint.get('epoch', 0)
        if not args.eval_only:
            run_id = checkpoint.get('wandb_run_id', run_id)
            print(f"Found checkpoint! Resuming run {run_id} from epoch {start_epoch}...")
        else:
            print(f"Found checkpoint for eval-only: {checkpoint_path} (epoch {start_epoch}).")
    elif args.eval_checkpoint_path is not None:
        parser.error(f"--eval_checkpoint_path was provided but file does not exist: {args.eval_checkpoint_path}")

    if not args.eval_only and args.wandb_run_id is not None:
        run_id = args.wandb_run_id

    print("Initializing Hybrid Diffusion Agent...")
    model = HybridFrozenBrainDiffusionHands(
        base_policy_path=args.base_policy_path,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        cond_dim=args.cond_dim,
        diff_hidden_dim=args.diff_hidden_dim,
        diff_layers=args.diff_layers,
        device=args.device
    )
    
    if checkpoint is not None and "model_state_dict" in checkpoint:
        model.load_trainable_state_dict(checkpoint['model_state_dict'])
    elif args.eval_only and args.eval_policy_mode in ("hybrid", "mixed", "residual"):
        parser.error("--eval_only with --eval_policy_mode hybrid/mixed/residual requires a checkpoint with model_state_dict.")

    preprocessor = None
    postprocessor = None
    if has_lerobot_processors:
        preprocessor_overrides = {
            "device_processor": {"device": str(args.device)},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=model.base_policy.config,
            pretrained_path=args.base_policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        print("[Eval Stack] Using LeRobot preprocessor/postprocessor for preflight/eval.")
    else:
        print("[Eval Stack] LeRobot processors unavailable; using fallback direct formatting.")

    preflight_required = args.preflight_baseline_check or (args.min_baseline_success is not None)
    preflight_result = None
    if preflight_required:
        preflight_result = run_preflight_baseline_check(
            model,
            args.device,
            tasks=args.preflight_task_ids,
            num_episodes=args.preflight_episodes,
            max_steps=args.eval_max_steps,
            image_flip_mode=args.image_flip_mode,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        if preflight_result is None:
            print("[Preflight] Unable to run baseline check. Aborting due to preflight requirement.")
            return
        if args.min_baseline_success is not None:
            observed = preflight_result["overall_success"]
            if observed < args.min_baseline_success:
                print(
                    f"[Preflight Gate] Baseline success {observed:.3f} < required {args.min_baseline_success:.3f}. "
                    "W&B and training will not start."
                )
                return

    if args.eval_only:
        eval_wandb_kwargs = {
            "project": args.wandb_project,
            "config": vars(args),
            "job_type": "eval_only",
        }
        if args.eval_wandb_run_id is not None:
            eval_wandb_kwargs["id"] = args.eval_wandb_run_id
            eval_wandb_kwargs["resume"] = args.eval_wandb_resume
        wandb.init(**eval_wandb_kwargs)
        eval_step = int(getattr(wandb.run, "step", 0) or 0)
        if preflight_result is not None:
            log_preflight_metrics(preflight_result, step=eval_step)

        if args.eval_output_dir is not None:
            os.makedirs(args.eval_output_dir, exist_ok=True)
            eval_cfg_path = os.path.join(args.eval_output_dir, "eval_config.json")
            try:
                with open(eval_cfg_path, "w") as f:
                    json.dump(vars(args), f, indent=2)
                print(f"[Eval Config] Saved evaluation config to: {eval_cfg_path}")
            except Exception as e:
                print(f"Warning: Failed to save eval_config.json: {e}")

        evaluate_in_environment(
            model,
            args.device,
            start_epoch,
            tasks=args.eval_task_ids,
            num_episodes=args.eval_episodes,
            max_steps=args.eval_max_steps,
            global_step=eval_step,
            image_flip_mode=args.image_flip_mode,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_mode=args.eval_policy_mode,
            hybrid_mix=args.eval_hybrid_mix,
            residual_alpha=args.eval_residual_alpha,
            diffusion_steps=args.eval_diffusion_steps,
            eval_replan_each_step=args.eval_replan_each_step,
            eval_log_action_stats_every=args.eval_log_action_stats_every,
            eval_action_clip=args.eval_action_clip,
            eval_output_dir=args.eval_output_dir,
        )
        wandb.finish()
        print("Eval-only complete.")
        return

    wandb.init(project=args.wandb_project, id=run_id, resume=args.wandb_resume, config=vars(args))
    active_run_id = str(getattr(wandb.run, "id", run_id))
    run_id = active_run_id
    if args.wandb_run_id_file is not None:
        run_id_path = Path(args.wandb_run_id_file)
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(f"{active_run_id}\n", encoding="utf-8")
    wandb_resume_step = int(getattr(wandb.run, "step", 0) or 0)
    if preflight_result is not None:
        log_preflight_metrics(preflight_result, step=wandb_resume_step)

    is_hdf5_dir = False
    if os.path.isdir(args.dataset_repo_id):
        h5_found = list(Path(args.dataset_repo_id).rglob("*.h5")) + list(Path(args.dataset_repo_id).rglob("*.hdf5"))
        if len(h5_found) > 0:
            is_hdf5_dir = True

    if is_hdf5_dir:
        print(f"Loading local HDF5 Dataset from: {args.dataset_repo_id} ...")
        dataset = HDF5LiberoDataset(
            args.dataset_repo_id,
            chunk_size=args.chunk_size,
            train_task_ids=args.train_task_ids,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=hdf5_collate_fn,
        )
    else:
        print(f"Loading Dataset: {args.dataset_repo_id} ...")
        delta_timestamps = {'action': [i / 10.0 for i in range(args.chunk_size)]}
        dataset_kwargs = {"delta_timestamps": delta_timestamps}
        if args.video_backend:
            dataset_kwargs["video_backend"] = args.video_backend

        dataset = LeRobotDataset(
            args.dataset_repo_id,
            **dataset_kwargs
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    # Grab a single fixed batch early for consistent visualizations across epochs
    print("Caching fixed batch for evaluation interpolation visualizations...")
    raw_vis_batch = next(iter(dataloader))
    vis_batch = {}
    for k, v in raw_vis_batch.items():
        if isinstance(v, torch.Tensor):
            vis_batch[k] = v.detach().clone()
        elif isinstance(v, list):
            vis_batch[k] = list(v)
        else:
            vis_batch[k] = v
    del raw_vis_batch

    # Optimizer specifically targeting ONLY the diffusion head
    optimizer = optim.AdamW(model.diffusion_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
        checkpoint_step = start_epoch * len(dataloader)
        global_step = max(checkpoint_step, wandb_resume_step)
        if global_step != checkpoint_step:
            print(f"[W&B] Resuming from run step {wandb_resume_step} (checkpoint-derived step was {checkpoint_step}).")
    else:
        print(f"[W&B] No checkpoint file loaded. Starting training from epoch 1 (global_step={wandb_resume_step}).")
        global_step = wandb_resume_step
    print("Starting Training Loop...")
    if args.residual_target:
        print(f"[Train] Residual-target mode enabled (delta_l2_weight={args.delta_l2_weight}).")
        if hasattr(model, "base_policy"):
            set_base_policy_num_steps(model.base_policy, 1)
    for epoch in range(start_epoch, args.epochs):
        model.diffusion_head.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if args.dry_run and batch_idx >= 2:
                print(f"[Dry Run] Completed {batch_idx} batches. Breaking epoch for fast sanity check.")
                break
            # Normalize state and action to policy space [-1, 1] using preprocessor if available
            if preprocessor is not None:
                if "observation.language.tokens" not in batch:
                    batch = preprocessor(batch)
                    clear_preprocessor_state(preprocessor)

            # Move tensors in batch to the specified compute device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if k == "action":
                        batch[k] = v.to(dtype=torch.float32, device=args.device)
                    elif v.is_floating_point():
                        batch[k] = v.to(dtype=torch.float32, device=args.device)
                    else:
                        batch[k] = v.to(device=args.device)
            
            # Extract Ground truth chunked actions [B, chunk_size, action_dim]
            gt_actions = batch["action"]
            if gt_actions.ndim == 2:
                gt_actions = gt_actions.unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass: Extract semantic features from the frozen VLA, 
            # then compute flow matching distance vs target velocity
            loss, loss_metrics = model.compute_loss(
                batch,
                gt_actions,
                residual_target=args.residual_target,
                delta_l2_weight=args.delta_l2_weight,
                return_metrics=True,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.diffusion_head.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1

            
            wandb_payload = {
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/flow_loss": loss_metrics["flow_loss"],
                "train/target_action_abs_mean": loss_metrics["target_action_abs_mean"],
            }
            if args.residual_target:
                wandb_payload["train/delta_l2_loss"] = loss_metrics["delta_l2_loss"]
                if "base_action_abs_mean" in loss_metrics:
                    wandb_payload["train/base_action_abs_mean"] = loss_metrics["base_action_abs_mean"]
            wandb.log(wandb_payload, step=global_step)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Explicitly delete variables that hold large compute graphs or memory
            del loss, loss_metrics, batch, gt_actions

            if args.decoder_cache_clear_freq > 0 and (global_step % args.decoder_cache_clear_freq == 0):
                clear_video_decoder_cache()
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch+1}, step=global_step)
        
        # Always save latest checkpoint every epoch for seamless resumption
        ckpt_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.get_trainable_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'wandb_run_id': run_id,
        }
        torch.save(ckpt_state, latest_ckpt_path)

        # Save periodic numbered checkpoint according to vis_freq
        if (epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"hybrid_diff_epoch_{epoch+1:03d}.pt")
            torch.save(ckpt_state, ckpt_path)
            print(f"Checkpoints saved to {args.out_dir}")
            
        # Periodic visualization and real environment evaluation
        if (epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epochs:
            plot_diffusion_trajectories(model, vis_batch, args.device, epoch+1, global_step=global_step)
            # Evaluate across multiple tasks in the benchmark
            evaluate_in_environment(
                model,
                args.device,
                epoch+1,
                tasks=args.eval_task_ids,
                num_episodes=args.eval_episodes,
                global_step=global_step,
                image_flip_mode=args.image_flip_mode,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                policy_mode=args.eval_policy_mode,
                hybrid_mix=args.eval_hybrid_mix,
                residual_alpha=args.eval_residual_alpha,
                diffusion_steps=args.eval_diffusion_steps,
                eval_replan_each_step=args.eval_replan_each_step,
                eval_log_action_stats_every=args.eval_log_action_stats_every,
                eval_action_clip=args.eval_action_clip,
                max_steps=args.eval_max_steps,
            )
        
        clear_video_decoder_cache()
        torch.cuda.empty_cache()
            
    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    try:
        with LinuxInhibit(reason="Training Hybrid Diffusion Model"):
            train()
    except Exception as _inhibit_err:
        train()
