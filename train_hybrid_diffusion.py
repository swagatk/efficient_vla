import argparse
import os
import torch

# Fix for PyTorch 2.6+ weights_only=True default when loading libero init states
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Iterable
import re

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

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Warning: Could not import LeRobotDataset from lerobot.")

try:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    has_lerobot_processors = True
except ImportError:
    has_lerobot_processors = False
    print("Warning: Could not import LeRobot pre/post processors.")

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

def evaluate_in_environment(
    model,
    device,
    epoch,
    tasks: Iterable[int] = range(10),
    num_episodes=2,
    max_steps=500,
    global_step=None,
    image_flip_mode="vertical",
    preprocessor=None,
    postprocessor=None,
    policy_mode="hybrid",
    hybrid_mix=0.0,
    diffusion_steps=10,
):
    """Run an evaluation loop in the LIBERO environment to measure true success rate."""
    if not has_libero:
        return

    benchmark = get_benchmark_dict()["libero_10"]()
    benchmark_root = os.path.dirname(libero_pkg.__file__)
    
    model.eval()
    
    all_tasks_success = []
    
    for task_id in tasks:
        task = benchmark.get_task(task_id)
        bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)
        
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_path,
            camera_heights=256,
            camera_widths=256,
        )
        
        success_count = 0
        total_reward = 0.0
    
        print(f"\n--- Running Evaluation on Task {task_id}: {task.language} ---")
        for ep in range(num_episodes):
            env.reset()
            init_states = benchmark.get_task_init_states(task_id)
            env.set_init_state(random.choice(init_states))
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())

            if hasattr(model.base_policy, "reset"):
                model.base_policy.reset()
            
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
                
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if policy_mode == "base":
                            base_action = model.base_policy.select_action(batch)
                            if postprocessor is not None:
                                env_action = postprocessor(base_action)
                                action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]
                            else:
                                action_np = base_action[0].detach().cpu().to(torch.float32).numpy()
                                action_np = np.clip(action_np, -1.0, 1.0)
                        else:
                            if pending_chunk is None or chunk_idx >= pending_chunk.shape[1]:
                                pending_chunk = model.select_action(
                                    batch,
                                    steps=diffusion_steps,
                                    return_intermediates=False,
                                )
                                chunk_idx = 0

                            hybrid_action = pending_chunk[:, chunk_idx, :]
                            chunk_idx += 1

                            if policy_mode == "mixed":
                                base_action = model.base_policy.select_action(batch)
                                blended_action = (1.0 - hybrid_mix) * base_action + hybrid_mix * hybrid_action
                                if postprocessor is not None:
                                    env_action = postprocessor(blended_action)
                                    action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]
                                else:
                                    action_np = blended_action[0].detach().cpu().to(torch.float32).numpy()
                                    action_np = np.clip(action_np, -1.0, 1.0)
                            else:
                                if postprocessor is not None:
                                    env_action = postprocessor(hybrid_action)
                                    action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]
                                else:
                                    action_np = hybrid_action[0].detach().cpu().to(torch.float32).numpy()
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
            print(f"Eval Ep {ep+1}/{num_episodes} | Success: {ep_success} | Reward: {ep_reward:.2f} | Steps: {step}")
            
        success_rate = success_count / num_episodes
        avg_ep_reward = total_reward / num_episodes
        all_tasks_success.append(success_rate)
        
        eval_log = {
            f"eval_task/task_{task_id}_success_rate": success_rate,
            f"eval_task/task_{task_id}_reward": avg_ep_reward,
            "epoch": epoch
        }
        if global_step is not None:
            wandb.log(eval_log, step=global_step)
        else:
            wandb.log(eval_log)
        
        # Clean up env for next task to prevent memory leaks from MuJoCo
        env.close()
        del env

    overall_success = sum(all_tasks_success) / len(all_tasks_success)
    print(f"\nEval Complete | Overall Success Rate: {overall_success:.2f}\n")
    
    overall_log = {
        "eval/success rate": overall_success,
        "epoch": epoch
    }
    if global_step is not None:
        wandb.log(overall_log, step=global_step)
    else:
        wandb.log(overall_log)
    model.train()


def run_preflight_baseline_check(
    model,
    device,
    tasks: Iterable[int],
    num_episodes=3,
    max_steps=500,
    image_flip_mode="vertical",
    preprocessor=None,
    postprocessor=None,
):
    """Evaluate frozen base SmolVLA before training to establish a task-wise baseline."""
    if not has_libero:
        print("[Preflight] Skipped baseline check because LIBERO is unavailable.")
        return None

    benchmark = get_benchmark_dict()["libero_10"]()
    benchmark_root = os.path.dirname(libero_pkg.__file__)
    all_tasks_success = []
    task_metrics = []

    model.base_policy.eval()
    print("\n[Preflight] Running frozen base-policy baseline check...")

    for task_id in tasks:
        task = benchmark.get_task(task_id)
        bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_path,
            camera_heights=256,
            camera_widths=256,
        )

        success_count = 0
        total_reward = 0.0

        print(f"\n[Preflight] Task {task_id}: {task.language}")
        for ep in range(num_episodes):
            env.reset()
            init_states = benchmark.get_task_init_states(task_id)
            env.set_init_state(random.choice(init_states))
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())

            if hasattr(model.base_policy, "reset"):
                model.base_policy.reset()

            step = 0
            done = False
            ep_reward = 0.0
            ep_success = False

            while not done and step < max_steps:
                if preprocessor is not None and postprocessor is not None:
                    raw_obs = get_policy_obs_from_env_obs(obs, flip_mode=image_flip_mode)
                    policy_obs = preprocess_observation(raw_obs)
                    policy_obs["task"] = [task.language]
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
                    text_out = processor(text=task.language, return_tensors='pt')

                    batch = {
                        'observation.images.image': img_tensor_agent,
                        'observation.images.image2': img_tensor_wrist,
                        'observation.state': state_tensor,
                        'observation.language.tokens': text_out['input_ids'].to(device),
                        'observation.language.attention_mask': text_out['attention_mask'].to(device).bool(),
                        'language_instruction': [task.language]
                    }

                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        base_action = model.base_policy.select_action(batch)

                if postprocessor is not None:
                    env_action = postprocessor(base_action)
                    action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]
                else:
                    action_np = base_action[0].detach().cpu().to(torch.float32).numpy()
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
            print(
                f"[Preflight] Ep {ep+1}/{num_episodes} | Success: {ep_success} "
                f"| Reward: {ep_reward:.2f} | Steps: {step}"
            )

        success_rate = success_count / num_episodes
        avg_ep_reward = total_reward / num_episodes
        all_tasks_success.append(success_rate)
        task_metrics.append({
            "task_id": int(task_id),
            "success_rate": float(success_rate),
            "avg_reward": float(avg_ep_reward),
        })
        env.close()

    overall_success = sum(all_tasks_success) / len(all_tasks_success)
    print(f"\n[Preflight] Overall baseline success rate: {overall_success:.2f}\n")
    return {
        "overall_success": float(overall_success),
        "episodes": int(num_episodes),
        "task_metrics": task_metrics,
    }


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
    with torch.no_grad():
        gt_actions = batch["action"].to(device)
        if gt_actions.ndim == 2:
            gt_actions = gt_actions.unsqueeze(1)
        
        # We only visualize the first num_samples to keep it fast
        vis_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v_sub = v[:num_samples]
                if v_sub.is_floating_point():
                    vis_batch[k] = v_sub.to(dtype=torch.bfloat16, device=device)
                else:
                    vis_batch[k] = v_sub.to(device=device)
            elif isinstance(v, list):
                vis_batch[k] = v[:num_samples]
            else:
                vis_batch[k] = v
                
        gt_actions = gt_actions[:num_samples]
        
        # DEBUG
        print("DEBUG VIS_BATCH DEVICES:")
        for k,v in vis_batch.items():
            if hasattr(v, 'device'):
                print(f"  {k}: {v.device}")
        
        # Run inference tracking intermediate noise steps
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        default="vertical",
        choices=["vertical", "vertical_horizontal", "none"],
        help="Image flip convention for preflight/eval observation preprocessing.",
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
        choices=["base", "hybrid", "mixed"],
        help="Policy used during eval: base SmolVLA, hybrid diffusion-only, or blended mixed mode.",
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
    
    args = parser.parse_args()
    args.eval_task_ids = parse_task_id_tokens(args.eval_task_ids)
    args.preflight_task_ids = parse_task_id_tokens(args.preflight_task_ids) if args.preflight_task_ids is not None else list(args.eval_task_ids)
    if args.min_baseline_success is not None and not (0.0 <= args.min_baseline_success <= 1.0):
        parser.error("--min_baseline_success must be in [0, 1].")
    if not (0.0 <= args.eval_hybrid_mix <= 1.0):
        parser.error("--eval_hybrid_mix must be in [0, 1].")
    if args.eval_diffusion_steps < 1:
        parser.error("--eval_diffusion_steps must be >= 1.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    run_id = wandb.util.generate_id()
    start_epoch = 0
    checkpoint = None
    latest_ckpt_path = os.path.join(args.out_dir, "latest_checkpoint.pt")
    
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=args.device)
        run_id = checkpoint.get('wandb_run_id', run_id)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Found checkpoint! Resuming run {run_id} from epoch {start_epoch}...")

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
    
    if checkpoint is not None:
        model.load_trainable_state_dict(checkpoint['model_state_dict'])

    preprocessor = None
    postprocessor = None
    if has_lerobot_processors:
        preprocessor_overrides = {
            "device_processor": {"device": str(args.device)},
            "rename_observations_processor": {"rename_map": {}},
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
            max_steps=500,
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

    wandb.init(project=args.wandb_project, id=run_id, resume="allow", config=vars(args))
    wandb_resume_step = int(getattr(wandb.run, "step", 0) or 0)
    if preflight_result is not None:
        log_preflight_metrics(preflight_result, step=wandb_resume_step)

    print(f"Loading Dataset: {args.dataset_repo_id} ...")
    # For a chunk size, we need up to chunk_size future actions
    delta_timestamps = {'action': [i / 10.0 for i in range(args.chunk_size)]} # assuming 10Hz, modify dt to match your dataset 
    
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        delta_timestamps=delta_timestamps
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Grab a single fixed batch early for consistent visualizations across epochs
    print("Caching fixed batch for evaluation interpolation visualizations...")
    vis_batch = next(iter(dataloader))

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
    print("Starting Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        model.diffusion_head.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Move tensors in batch to the specified compute device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    # Keep action targets in fp32 for stable diffusion loss.
                    if k == "action":
                        batch[k] = v.to(dtype=torch.float32, device=args.device)
                    # Cast remaining floats to bfloat16 to match the frozen backbone.
                    elif v.is_floating_point():
                        batch[k] = v.to(dtype=torch.bfloat16, device=args.device)
                    else:
                        batch[k] = v.to(device=args.device)
            
            # Extract Ground truth chunked actions [B, chunk_size, action_dim]
            gt_actions = batch["action"]
            if gt_actions.ndim == 2:
                # Fallback if chunking wasn't handled by delta_timestamps
                gt_actions = gt_actions.unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass: Extract semantic features from the frozen VLA, 
            # then compute flow matching distance vs target velocity
            loss = model.compute_loss(batch, gt_actions)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.diffusion_head.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch+1}, step=global_step)
        
        # Save checkpoints BEFORE evaluation to prevent progress loss on eval crashes
        if (epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.get_trainable_state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'wandb_run_id': run_id,
            }
            
            # Save periodic numbered checkpoint
            ckpt_path = os.path.join(args.out_dir, f"hybrid_diff_epoch_{epoch+1:03d}.pt")
            torch.save(ckpt_state, ckpt_path)
            
            # Save latest checkpoint for easy resumption
            torch.save(ckpt_state, latest_ckpt_path)
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
                diffusion_steps=args.eval_diffusion_steps,
            )
            
    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    with LinuxInhibit(reason="Training Hybrid Diffusion Model"):
        train()
