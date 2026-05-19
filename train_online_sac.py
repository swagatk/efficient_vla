#!/usr/bin/env python3
import os
import argparse
import re
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import random
from collections import deque
from copy import deepcopy

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import libero.libero as libero_pkg
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

from sac_residual_agent import SACResidualVLAPolicy
from visual_prompt_wrapper import VisualPromptingWrapper
from linux_inhibit import LinuxInhibit
from robosuite.utils.transform_utils import quat2axisangle
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation

class ReplayBuffer:
    def __init__(self, capacity, success_capacity=5000):
        self.buffer = deque(maxlen=capacity)
        self.success_buffer = deque(maxlen=success_capacity)

    @staticmethod
    def _to_cpu_item(item):
        if torch.is_tensor(item):
            return item.detach().cpu()
        if isinstance(item, dict):
            return {k: ReplayBuffer._to_cpu_item(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            converted = [ReplayBuffer._to_cpu_item(v) for v in item]
            return type(item)(converted)
        return item

    @classmethod
    def _normalize_transition(cls, transition):
        if not isinstance(transition, (list, tuple)) or len(transition) != 5:
            return transition
        state_dict, action, reward, next_state_dict, done = transition
        return (
            cls._to_cpu_item(state_dict),
            cls._to_cpu_item(action),
            reward,
            cls._to_cpu_item(next_state_dict),
            done,
        )

    def push(self, state_dict, action, reward, next_state_dict, done, success=False):
        transition = self._normalize_transition((state_dict, action, reward, next_state_dict, done))
        self.buffer.append(transition)
        if success:
            self.success_buffer.append(transition)

    def sample(self, batch_size, success_fraction=0.0):
        success_fraction = float(np.clip(success_fraction, 0.0, 1.0))
        n_success = min(len(self.success_buffer), int(batch_size * success_fraction))
        n_regular = batch_size - n_success

        if n_regular > len(self.buffer):
            n_regular = len(self.buffer)
            n_success = min(batch_size - n_regular, len(self.success_buffer))

        batch = []
        if n_regular > 0:
            batch.extend(random.sample(self.buffer, n_regular))
        if n_success > 0:
            batch.extend(random.sample(self.success_buffer, n_success))
        random.shuffle(batch)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # Merge dicts
        states = {
            k: torch.cat([self._to_cpu_item(s[k]) for s in state_batch], dim=0)
            for k in state_batch[0].keys()
        }
        next_states = {
            k: torch.cat([self._to_cpu_item(s[k]) for s in next_state_batch], dim=0)
            for k in next_state_batch[0].keys()
        }
        
        actions = torch.stack([self._to_cpu_item(a) for a in action_batch])
        rewards = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()
        self.success_buffer.clear()

    def state_dict(self, max_buffer_items=None, max_success_items=None):
        if max_buffer_items is None:
            buffer_items = list(self.buffer)
        else:
            buffer_items = list(self.buffer)[-int(max_buffer_items):]

        if max_success_items is None:
            success_items = list(self.success_buffer)
        else:
            success_items = list(self.success_buffer)[-int(max_success_items):]

        return {
            "capacity": int(self.buffer.maxlen) if self.buffer.maxlen is not None else None,
            "success_capacity": int(self.success_buffer.maxlen) if self.success_buffer.maxlen is not None else None,
            "buffer": buffer_items,
            "success_buffer": success_items,
        }

    def load_state_dict(self, payload):
        if not isinstance(payload, dict):
            return

        capacity = payload.get("capacity", self.buffer.maxlen)
        success_capacity = payload.get("success_capacity", self.success_buffer.maxlen)
        restored_buffer = [self._normalize_transition(t) for t in payload.get("buffer", [])]
        restored_success = [self._normalize_transition(t) for t in payload.get("success_buffer", [])]
        self.buffer = deque(restored_buffer, maxlen=capacity)
        self.success_buffer = deque(restored_success, maxlen=success_capacity)

    def __len__(self):
        return len(self.buffer)

def extract_success(info, env=None):
    # LeRobot's LIBERO wrapper computes success via env.check_success().
    if env is not None and hasattr(env, "check_success"):
        try:
            return bool(env.check_success())
        except Exception:
            pass
    if not isinstance(info, dict): return False
    for key in ("success", "is_success", "task_success", "episode_success"):
        if key in info:
            val = info[key]
            return bool(val.item() if hasattr(val, "item") else val)
    return False


def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def load_trainable_checkpoint(agent, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trainable = checkpoint.get("agent_trainable_state_dict", {})

    for name in ("vision_encoder", "state_encoder", "actor_net", "actor_mean", "actor_log_std", "q1", "q2"):
        if name in trainable and hasattr(agent, name):
            getattr(agent, name).load_state_dict(trainable[name])

    return checkpoint


def save_trainable_checkpoint(
    checkpoint_path,
    agent,
    episode,
    global_step,
    actor_optimizer=None,
    critic_optimizer=None,
    wandb_run_id=None,
    replay_state_dict=None,
    target_state_dict=None,
    alpha_state_dict=None,
    success_history=None,
):
    payload = {
        "episode": int(episode),
        "global_step": int(global_step),
        "agent_trainable_state_dict": agent.get_trainable_state_dict(),
    }
    if actor_optimizer is not None:
        payload["actor_optimizer_state_dict"] = actor_optimizer.state_dict()
    if critic_optimizer is not None:
        payload["critic_optimizer_state_dict"] = critic_optimizer.state_dict()
    if wandb_run_id is not None:
        payload["wandb_run_id"] = str(wandb_run_id)
    if replay_state_dict is not None:
        payload["replay_state_dict"] = replay_state_dict
    if target_state_dict is not None:
        payload["target_state_dict"] = target_state_dict
    if alpha_state_dict is not None:
        payload["alpha_state_dict"] = alpha_state_dict
    if success_history is not None:
        payload["success_history"] = success_history

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(payload, checkpoint_path)


def save_checkpoint_bundle(
    checkpoint_dir,
    agent,
    episode,
    global_step,
    actor_optimizer=None,
    critic_optimizer=None,
    wandb_run_id=None,
    replay_state_dict=None,
    target_state_dict=None,
    alpha_state_dict=None,
    success_history=None,
):
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    ep_path = os.path.join(checkpoint_dir, f"checkpoint_ep_{int(episode):04d}.pt")

    save_trainable_checkpoint(
        checkpoint_path=latest_path,
        agent=agent,
        episode=episode,
        global_step=global_step,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        wandb_run_id=wandb_run_id,
        replay_state_dict=replay_state_dict,
        target_state_dict=target_state_dict,
        alpha_state_dict=alpha_state_dict,
        success_history=success_history,
    )
    save_trainable_checkpoint(
        checkpoint_path=ep_path,
        agent=agent,
        episode=episode,
        global_step=global_step,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        wandb_run_id=wandb_run_id,
        replay_state_dict=replay_state_dict,
        target_state_dict=target_state_dict,
        alpha_state_dict=alpha_state_dict,
        success_history=success_history,
    )
    print(f"[ckpt] saved: {latest_path}")
    print(f"[ckpt] saved: {ep_path}")


def build_policy_batch_from_obs(obs, instruction, device, prompter=None, step=0):
    img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
    if prompter is not None:
        prompt_obs = prompter.apply_prompts(
            {"image": img_agent, "instruction": instruction},
            update_grounding=(step % 5 == 0),
        )
        img_agent = prompt_obs["image"]

    img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()
    state_np = np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    )

    state_tensor = torch.from_numpy(state_np).to(torch.float32).unsqueeze(0).to(device)

    # Match lerobot env output structure so preprocess_observation can convert HWC uint8 to BCHW float.
    batch_obs = {
        "pixels": {
            "image": img_agent,
            "image2": img_wrist,
        },
        "agent_pos": state_np.astype(np.float32),
    }
    return batch_obs, img_agent, state_tensor


def extract_features_with_encoders(batch, vision_encoder, state_encoder):
    img = batch.get("observation.images.image", batch.get("observation.image"))
    img = img.float() if img.dtype != torch.float32 else img
    if img.max() > 1.0:
        img = img / 255.0
        
    import torchvision.transforms as T
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    
    state = batch.get("observation.state").float()
    features = vision_encoder(img)
    state_features = state_encoder(state)
    return torch.cat([features, state_features], dim=-1)


def apply_dense_shaping(env, next_obs, current_instruction, reward, prev_dist):
    """Optional dense shaping adapted from PPO script for hard tasks."""
    try:
        eef_site_id = env.env.sim.model.site_name2id("gripper0_grip_site")
        eef_pos = env.env.sim.data.site_xpos[eef_site_id]

        target_distances = []
        for obj in env.env.objects:
            obj_id = env.env.sim.model.body_name2id(obj.root_body)
            obj_pos = env.env.sim.data.body_xpos[obj_id]
            dist = np.linalg.norm(eef_pos - obj_pos)

            obj_name = obj.name.lower().replace("_", " ")
            obj_name = "".join(c for c in obj_name if not c.isdigit()).strip()
            if obj_name and obj_name in current_instruction.lower():
                target_distances.append(dist)

        if target_distances:
            current_min_dist = min(target_distances)
            if prev_dist is not None:
                dist_improvement = prev_dist - current_min_dist
                reward += 5.0 * dist_improvement
            prev_dist = current_min_dist

            gripper_qpos = next_obs.get("robot0_gripper_qpos", [0.0])
            if sum(gripper_qpos) > 0.01 and current_min_dist < 0.05:
                reward += 0.02
    except Exception:
        pass

    return reward, prev_dist


def infer_wandb_run_id_from_latest_file(latest_run_path="wandb/latest-run"):
    if not os.path.exists(latest_run_path):
        return None

    latest_value = None
    if os.path.islink(latest_run_path):
        try:
            latest_value = os.readlink(latest_run_path).strip()
        except OSError:
            latest_value = None
    else:
        try:
            with open(latest_run_path, "r", encoding="utf-8") as f:
                latest_value = f.read().strip()
        except OSError:
            latest_value = None

    if not latest_value:
        return None

    latest_value = os.path.basename(latest_value)

    # Typical format: run-YYYYMMDD_HHMMSS-<run_id>
    m = re.search(r"run-[^-]+-([A-Za-z0-9]+)$", latest_value)
    if m:
        return m.group(1)

    return None


def get_wandb_run_id_file_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, "wandb_run_id.txt")


def load_wandb_run_id_from_checkpoint_dir(checkpoint_dir):
    run_id_path = get_wandb_run_id_file_path(checkpoint_dir)
    if not os.path.exists(run_id_path):
        return None
    try:
        with open(run_id_path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        return value or None
    except OSError:
        return None


def save_wandb_run_id_to_checkpoint_dir(checkpoint_dir, run_id):
    if not run_id:
        return
    run_id_path = get_wandb_run_id_file_path(checkpoint_dir)
    try:
        with open(run_id_path, "w", encoding="utf-8") as f:
            f.write(str(run_id).strip() + "\n")
    except OSError as exc:
        print(f"[wandb] warning: failed to persist run id to {run_id_path}: {exc}")


def format_progress_bar(current, total, width=24):
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    filled = int(round(width * (current / total)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def run_preflight_base_success(
    env,
    benchmark,
    task,
    task_id,
    agent,
    device,
    prompter,
    preprocessor,
    postprocessor,
    episodes,
    max_steps,
):
    successes = 0

    for ep in range(episodes):
        env.reset()
        env.set_init_state(random.choice(benchmark.get_task_init_states(task_id)))
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        agent.base_policy.reset()

        done = False
        step = 0
        episode_success = False

        while not done and step < max_steps:
            raw_obs, _, _ = build_policy_batch_from_obs(
                obs,
                task.language,
                device,
                prompter=prompter,
                step=step,
            )
            policy_obs = preprocess_observation(raw_obs)
            policy_obs["task"] = [task.language]
            batch_obs = preprocessor(policy_obs)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base_action = agent.base_policy.select_action(batch_obs)
            env_action = postprocessor(base_action)

            next_obs, _, env_done, info = env.step(env_action.detach().cpu().numpy()[0])
            step_success = extract_success(info, env)
            episode_success = episode_success or step_success
            done = bool(env_done or step_success)
            obs = next_obs
            step += 1

        successes += int(episode_success)

    return successes / max(episodes, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=3, help="LIBERO task index to train on")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Python, NumPy, and PyTorch.",
    )
    parser.add_argument("--use_visual_prompting", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only using a checkpoint for the given task_id.")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Number of episodes to run in --eval mode.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint file. Defaults to checkpoints_task_<task_id>/latest_checkpoint.pt",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory where checkpoints are written. Defaults to checkpoints_task_<task_id>.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5,
        help="Save latest checkpoint every N training episodes.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Optional explicit Weights & Biases run ID to use (useful for forced resume).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="efficient_vla_rl",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name. Defaults to task/seed-based name.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional Weights & Biases run group. Defaults to task-based group.",
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume SAC training from checkpoint if it exists.",
    )
    parser.add_argument(
        "--preflight_episodes",
        type=int,
        default=5,
        help="Number of base-only episodes to run before SAC updates.",
    )
    parser.add_argument(
        "--preflight_min_success",
        type=float,
        default=0.5,
        help="Abort if base-policy success in preflight falls below this threshold.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Actor/Critic learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Entropy temperature (lower for harder sparse tasks).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="SAC minibatch size.")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer capacity.")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Gradient updates per env step.")
    parser.add_argument(
        "--start_steps",
        type=int,
        default=0,
        help="Warmup steps with random residuals before SAC updates.",
    )
    parser.add_argument("--num_episodes", type=int, default=200, help="Total training episodes.")
    parser.add_argument("--max_steps", type=int, default=520, help="Episode horizon.")
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=0.01,
        help="Scale factor for residual action added to base policy action.",
    )
    parser.add_argument(
        "--success_bonus",
        type=float,
        default=20.0,
        help="Terminal reward bonus applied on success.",
    )
    parser.add_argument(
        "--step_penalty",
        type=float,
        default=0.01,
        help="Per-step penalty when episode has not succeeded.",
    )
    parser.add_argument(
        "--action_l2_penalty_coef",
        type=float,
        default=0.01,
        help="L2 penalty coefficient on actor output action magnitude.",
    )
    parser.add_argument(
        "--use_dense_shaping",
        action="store_true",
        help="Enable PPO-style dense shaping from object distance and gripper proximity.",
    )
    parser.add_argument(
        "--hard_task_preset",
        action="store_true",
        help="Apply a recommended preset for difficult tasks (e.g., task 0).",
    )
    parser.add_argument(
        "--auto_alpha",
        action="store_true",
        help="Enable automatic entropy temperature tuning.",
    )
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=None,
        help="Target entropy for auto-alpha. Defaults to -action_dim.",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=1e-4,
        help="Learning rate for entropy temperature when --auto_alpha is enabled.",
    )
    parser.add_argument(
        "--min_alpha",
        type=float,
        default=1e-5,
        help="Lower clamp bound for entropy temperature with auto-alpha.",
    )
    parser.add_argument(
        "--max_alpha",
        type=float,
        default=0.2,
        help="Upper clamp bound for entropy temperature with auto-alpha.",
    )
    parser.add_argument(
        "--success_replay_ratio",
        type=float,
        default=0.2,
        help="Fraction of each minibatch drawn from successful transitions.",
    )
    parser.add_argument(
        "--checkpoint_replay_size",
        type=int,
        default=256,
        help="How many recent replay transitions to persist in each checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_success_replay_size",
        type=int,
        default=5000,
        help="How many recent successful replay transitions to persist in each checkpoint.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stop_requested = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    previous_sigttou_handler = None
    previous_sigttin_handler = None

    def _request_stop(sig, frame):
        nonlocal stop_requested
        sig_name = signal.Signals(sig).name if sig in signal.Signals.__members__.values() else str(sig)
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print(f"[signal] {sig_name} received, stopping at next safe checkpoint (press Ctrl+C again to force)")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGTTOU"):
        previous_sigttou_handler = signal.getsignal(signal.SIGTTOU)
        signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    if hasattr(signal, "SIGTTIN"):
        previous_sigttin_handler = signal.getsignal(signal.SIGTTIN)
        signal.signal(signal.SIGTTIN, signal.SIG_IGN)

    if args.hard_task_preset:
        args.start_steps = 0
        args.residual_scale = 0.02
        args.alpha = 0.005
        args.action_l2_penalty_coef = 0.01
        args.success_bonus = 20.0
        args.use_dense_shaping = True
        args.auto_alpha = True
        args.success_replay_ratio = 0.25
        args.preflight_min_success = min(args.preflight_min_success, 0.1)
        print("[preset] Applied hard-task SAC preset.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "HuggingFaceVLA/smolvla_libero"
    task_name = "libero_10"
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else f"checkpoints_task_{args.task_id}"
    default_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else default_checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # SAC Hyperparameters
    lr = args.learning_rate
    gamma = args.gamma
    tau = args.tau
    alpha = float(args.alpha)
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    updates_per_step = args.updates_per_step
    start_steps = args.start_steps
    num_episodes = args.num_episodes
    max_steps = args.max_steps

    # Lowered residual scale to prevent destroying base policy prior entirely
    agent = SACResidualVLAPolicy(base_model_id, device=device, residual_scale=args.residual_scale)

    for param in agent.base_policy.parameters():
        param.requires_grad = False

    benchmark = get_benchmark_dict()[task_name]()
    task = benchmark.get_task(args.task_id)
    bddl_file_path = os.path.join(os.path.dirname(libero_pkg.__file__), "bddl_files", task.problem_folder, task.bddl_file)
    
    env = OffScreenRenderEnv(bddl_file_name=bddl_file_path, camera_heights=256, camera_widths=256)
    
    # Disabled by default; enable only when explicitly requested.
    prompter = VisualPromptingWrapper(use_image_box=True, use_text_hint=True, device=device) if args.use_visual_prompting else None

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=agent.base_policy.config,
        pretrained_path=base_model_id,
    )

    if args.eval:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = load_trainable_checkpoint(agent, checkpoint_path, device)
        print(f"[eval] loaded checkpoint: {checkpoint_path}")
        if "episode" in ckpt:
            print(f"[eval] checkpoint episode: {ckpt['episode']}")

        eval_successes = 0
        eval_rewards = []
        eval_steps = []

        try:
            for ep in range(args.eval_episodes):
                env.reset()
                env.set_init_state(random.choice(benchmark.get_task_init_states(args.task_id)))
                obs = env.reset()
                for _ in range(10):
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                agent.base_policy.reset()

                done = False
                step = 0
                episode_success = False
                episode_reward = 0.0

                while not done and step < max_steps:
                    current_instruction = task.language
                    raw_obs, img_agent, state_tensor = build_policy_batch_from_obs(
                        obs,
                        current_instruction,
                        device,
                        prompter=prompter,
                        step=step,
                    )
                    policy_obs = preprocess_observation(raw_obs)
                    policy_obs["task"] = [current_instruction]
                    batch_obs = preprocessor(policy_obs)

                    img_uint8 = torch.from_numpy(img_agent.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
                    residual_batch = {
                        "observation.images.image": img_uint8,
                        "observation.state": state_tensor,
                    }

                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            final_action, _, _, _, _ = agent(batch_obs, residual_batch=residual_batch, deterministic=True)

                    env_action = postprocessor(final_action)
                    next_obs, reward, env_done, info = env.step(env_action.detach().cpu().numpy()[0])

                    step_success = extract_success(info, env)
                    episode_success = episode_success or step_success
                    done = bool(env_done or step_success)
                    episode_reward += float(reward)
                    obs = next_obs
                    step += 1

                eval_successes += int(episode_success)
                eval_rewards.append(episode_reward)
                eval_steps.append(step)
                print(f"[eval] Ep {ep}: success={int(episode_success)} reward={episode_reward:.2f} steps={step}")

            success_rate = eval_successes / max(1, args.eval_episodes)
            print(
                f"[eval] success_rate={success_rate:.4f} ({eval_successes}/{args.eval_episodes}) "
                f"mean_reward={np.mean(eval_rewards):.3f} mean_steps={np.mean(eval_steps):.1f}"
            )
        finally:
            try:
                env.close()
            except Exception as exc:
                print(f"[warn] env.close() failed: {exc}")
        return

    # Lightweight target networks (no duplicated base SmolVLA).
    target_vision_encoder = deepcopy(agent.vision_encoder).to(device)
    target_state_encoder = deepcopy(agent.state_encoder).to(device)
    target_q1 = deepcopy(agent.q1).to(device)
    target_q2 = deepcopy(agent.q2).to(device)
    for module in (target_vision_encoder, target_state_encoder, target_q1, target_q2):
        for param in module.parameters():
            param.requires_grad = False

    # Optimizers
    actor_params = list(agent.actor_net.parameters()) + list(agent.actor_mean.parameters()) + list(agent.actor_log_std.parameters())
    critic_params = list(agent.vision_encoder.parameters()) + list(agent.state_encoder.parameters()) + list(agent.q1.parameters()) + list(agent.q2.parameters())
    
    actor_optimizer = optim.Adam(actor_params, lr=lr)
    critic_optimizer = optim.Adam(critic_params, lr=lr)

    replay_buffer = ReplayBuffer(buffer_size)
    start_episode = 0
    global_step = 0
    wandb_run_id = args.wandb_run_id
    resumed_from_checkpoint = False
    initial_success_history = []

    action_dim = int(agent.action_dim)
    target_entropy = args.target_entropy if args.target_entropy is not None else -float(action_dim)
    log_alpha = None
    alpha_optimizer = None
    if args.auto_alpha:
        log_alpha = torch.tensor(np.log(max(alpha, 1e-8)), device=device, dtype=torch.float32, requires_grad=True)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
        print(f"[alpha] auto tuning enabled, target_entropy={target_entropy:.3f}")

    if args.resume_training and os.path.exists(checkpoint_path):
        resumed_from_checkpoint = True
        ckpt = load_trainable_checkpoint(agent, checkpoint_path, device)
        print(f"[train] loaded checkpoint: {checkpoint_path}")

        if "actor_optimizer_state_dict" in ckpt:
            actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in ckpt:
            critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])

        start_episode = int(ckpt.get("episode", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if wandb_run_id is None:
            wandb_run_id = ckpt.get("wandb_run_id", None)

        if "replay_state_dict" in ckpt:
            replay_buffer.load_state_dict(ckpt["replay_state_dict"])
            print(
                f"[train] restored replay: buffer={len(replay_buffer.buffer)} "
                f"success_buffer={len(replay_buffer.success_buffer)}"
            )
            
        initial_success_history = ckpt.get("success_history", [])

        target_state = ckpt.get("target_state_dict", None)
        if isinstance(target_state, dict):
            if "vision_encoder" in target_state:
                target_vision_encoder.load_state_dict(target_state["vision_encoder"])
            if "state_encoder" in target_state:
                target_state_encoder.load_state_dict(target_state["state_encoder"])
            if "q1" in target_state:
                target_q1.load_state_dict(target_state["q1"])
            if "q2" in target_state:
                target_q2.load_state_dict(target_state["q2"])
            print("[train] resume target_state_dict: restored from checkpoint")
        else:
            # Backward compatibility for old checkpoints: sync targets to resumed online nets.
            target_vision_encoder.load_state_dict(agent.vision_encoder.state_dict())
            target_state_encoder.load_state_dict(agent.state_encoder.state_dict())
            target_q1.load_state_dict(agent.q1.state_dict())
            target_q2.load_state_dict(agent.q2.state_dict())
            print("[train] resume target_state_dict: missing, fallback to online->target hard sync")

        alpha_state = ckpt.get("alpha_state_dict", None)
        if isinstance(alpha_state, dict):
            if "alpha" in alpha_state:
                alpha = float(alpha_state["alpha"])
            if log_alpha is not None and "log_alpha" in alpha_state:
                with torch.no_grad():
                    log_alpha.copy_(
                        torch.tensor(
                            float(alpha_state["log_alpha"]),
                            device=device,
                            dtype=torch.float32,
                        )
                    )
            if alpha_optimizer is not None and "alpha_optimizer_state_dict" in alpha_state:
                alpha_optimizer.load_state_dict(alpha_state["alpha_optimizer_state_dict"])
            print("[train] resume alpha_state_dict: restored from checkpoint")
        else:
            print("[train] resume alpha_state_dict: missing, fallback to CLI/init values")

        if start_episode >= num_episodes:
            raise ValueError(
                f"Checkpoint episode is {start_episode - 1}. Set --num_episodes > {start_episode - 1} "
                "to continue training."
            )

    if wandb_run_id is None:
        # Per-seed/per-checkpoint-dir run id persistence prevents cross-seed reuse.
        wandb_run_id = load_wandb_run_id_from_checkpoint_dir(checkpoint_dir)

    try:
        if args.resume_training:
            print("[preflight] skipped because --resume_training is enabled")
        else:
            preflight_success = run_preflight_base_success(
                env=env,
                benchmark=benchmark,
                task=task,
                task_id=args.task_id,
                agent=agent,
                device=device,
                prompter=prompter,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                episodes=args.preflight_episodes,
                max_steps=max_steps,
            )
            print(
                f"[preflight] base success={preflight_success:.3f} over {args.preflight_episodes} episodes"
            )
            if preflight_success < args.preflight_min_success:
                raise RuntimeError(
                    f"Preflight failed: base success {preflight_success:.3f} < {args.preflight_min_success:.3f}. "
                    "Training aborted to avoid wasting compute on a mismatched rollout stack."
                )

        if args.resume_training and resumed_from_checkpoint:
            if wandb_run_id is None:
                # Backward-compat fallback for old checkpoints without stored run id.
                wandb_run_id = infer_wandb_run_id_from_latest_file()
            if wandb_run_id is None:
                wandb_run_id = wandb.util.generate_id()
        elif args.resume_training:
            # Fresh seed with --resume_training enabled should still get a new run id.
            if wandb_run_id is None:
                wandb_run_id = wandb.util.generate_id()
        else:
            wandb_run_id = wandb.util.generate_id()

        print(f"[wandb] using run_id={wandb_run_id}")

        wandb_name = args.wandb_name if args.wandb_name else f"sac_residual_task{args.task_id}_seed{args.seed}"
        wandb_group = args.wandb_group if args.wandb_group else f"task_{args.task_id}"
        wandb_tags = ["sac", f"task_{args.task_id}", f"seed_{args.seed}"]
        print(f"[wandb] project={args.wandb_project} group={wandb_group} name={wandb_name}")

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            group=wandb_group,
            tags=wandb_tags,
            id=wandb_run_id,
            resume="allow",
            config=vars(args),
        )
        if getattr(wandb, "run", None) is not None:
            wandb_run_id = wandb.run.id
            save_wandb_run_id_to_checkpoint_dir(checkpoint_dir, wandb_run_id)

        last_episode_seen = start_episode - 1
        success_history = list(initial_success_history)
        
        for episode in range(start_episode, num_episodes):
            if stop_requested:
                raise KeyboardInterrupt

            last_episode_seen = episode
            env.reset()
            env.set_init_state(random.choice(benchmark.get_task_init_states(args.task_id)))
            obs = env.reset()
            # Mirror LeRobot LIBERO env: settle the simulator after reset/init-state.
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            agent.base_policy.reset()

            episode_reward = 0
            step = 0
            done = False
            episode_success = False
            prev_dist = None

            episode_actor_loss = []
            episode_critic_loss = []

            while not done and step < max_steps:
                if stop_requested:
                    raise KeyboardInterrupt

                current_instruction = task.language
                raw_obs, img_agent, state_tensor = build_policy_batch_from_obs(
                    obs,
                    current_instruction,
                    device,
                    prompter=prompter,
                    step=step,
                )
                policy_obs = preprocess_observation(raw_obs)
                policy_obs["task"] = [current_instruction]
                batch_obs = preprocessor(policy_obs)

                img_uint8 = torch.from_numpy(img_agent.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
                residual_batch = {
                    "observation.images.image": img_uint8,
                    "observation.state": state_tensor,
                }

                if global_step < start_steps:
                    # Completely uniformly random delta actions during warmup
                    action_dim = 7 # Typically 7 for Libero (6 joint + 1 gripper)
                    delta_action = (torch.rand((1, action_dim), device=device) * 2 - 1).to(torch.float32)
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            final_action, base_action, _, _, _ = agent(batch_obs, residual_batch=residual_batch, deterministic=False)

                            # Replace the agent's internal delta with pure uniform random
                            scaled_delta_action = agent.residual_scale * delta_action.to(device)
                            final_action = base_action + scaled_delta_action
                            final_action = torch.clamp(final_action, min=-1.0, max=1.0)
                else:
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            final_action, _, delta_action, _, _ = agent(batch_obs, residual_batch=residual_batch, deterministic=False)

                env_action = postprocessor(final_action)
                action_np = env_action.detach().cpu().numpy()[0]
                next_obs, reward, env_done, info = env.step(action_np)

                step_success = extract_success(info, env)
                episode_success = episode_success or step_success

                # Standard RL fix: do not mask target Q value if episode ends due to step limit truncation.
                # Only treat true task success as terminal.
                is_terminal = step_success
                done_signal = bool(env_done or step_success)

                # Reward shaping
                if step_success:
                    reward += args.success_bonus
                else:
                    reward -= args.step_penalty
                    if args.use_dense_shaping:
                        reward, prev_dist = apply_dense_shaping(
                            env=env,
                            next_obs=next_obs,
                            current_instruction=current_instruction,
                            reward=reward,
                            prev_dist=prev_dist,
                        )

                # Prepare next state
                next_img_agent = next_obs["agentview_image"][::-1, ::-1, :].copy()
                if prompter:
                    next_img_agent = prompter.apply_prompts(
                        {"image": next_img_agent, "instruction": current_instruction},
                        update_grounding=False,
                    )["image"]
                next_state_np = np.concatenate(
                    [next_obs["robot0_eef_pos"], quat2axisangle(next_obs["robot0_eef_quat"]), next_obs["robot0_gripper_qpos"]]
                )
                next_state_tensor = torch.from_numpy(next_state_np).to(torch.float32).unsqueeze(0).to(device)

                # Store to buffer as uint8 natively to drastically cut down RAM usage (OOM prevention)
                img_uint8 = torch.from_numpy(img_agent.copy()).permute(2, 0, 1).unsqueeze(0)
                next_img_uint8 = torch.from_numpy(next_img_agent.copy()).permute(2, 0, 1).unsqueeze(0)

                replay_buffer.push(
                    {"observation.images.image": img_uint8, "observation.state": state_tensor.detach().cpu()},
                    delta_action.detach().cpu()[0],
                    reward,
                    {"observation.images.image": next_img_uint8, "observation.state": next_state_tensor.detach().cpu()},
                    is_terminal,
                    success=step_success,
                )

                done = done_signal
                obs = next_obs
                episode_reward += reward
                step += 1
                global_step += 1

                # SAC Update Phase
                if len(replay_buffer) > batch_size and global_step > start_steps:
                    for _ in range(updates_per_step):
                        if stop_requested:
                            raise KeyboardInterrupt

                        b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(
                            batch_size,
                            success_fraction=args.success_replay_ratio,
                        )
                        b_states = {k: v.to(device) for k, v in b_states.items()}
                        b_actions, b_rewards, b_next_states, b_dones = (
                            b_actions.to(device),
                            b_rewards.to(device),
                            {k: v.to(device) for k, v in b_next_states.items()},
                            b_dones.to(device),
                        )

                        if log_alpha is not None:
                            alpha_value = float(torch.clamp(log_alpha.exp(), min=args.min_alpha, max=args.max_alpha).item())
                        else:
                            alpha_value = alpha

                        with torch.no_grad():
                            # SAC standard: use the target network for Q-value estimation
                            # but the CURRENT policy for the next action and its entropy (log prob)
                            next_features_target = extract_features_with_encoders(
                                b_next_states, target_vision_encoder, target_state_encoder
                            )
                            next_features_actor = agent.extract_features(b_next_states)

                            next_action, next_log_prob, _ = agent.sample_action(next_features_actor)
                            if next_log_prob is None:
                                raise RuntimeError("Expected stochastic SAC next_log_prob, got None.")

                            target_q1_val = target_q1(torch.cat([next_features_target, next_action], dim=-1))
                            target_q2_val = target_q2(torch.cat([next_features_target, next_action], dim=-1))
                            target_q = b_rewards + (1 - b_dones) * gamma * (
                                torch.min(target_q1_val, target_q2_val) - alpha_value * next_log_prob
                            )

                        features = agent.extract_features(b_states)
                        q1 = agent.q1(torch.cat([features, b_actions], dim=-1))
                        q2 = agent.q2(torch.cat([features, b_actions], dim=-1))
                        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()

                        # Actor update (using reparameterization trick)
                        features = agent.extract_features(b_states)
                        action_new, log_prob_new, _ = agent.sample_action(features.detach())
                        if log_prob_new is None:
                            raise RuntimeError("Expected stochastic SAC log_prob_new, got None.")
                        q1_new = agent.q1(torch.cat([features.detach(), action_new], dim=-1))
                        q2_new = agent.q2(torch.cat([features.detach(), action_new], dim=-1))
                        q_new = torch.min(q1_new, q2_new)

                        # Penalize the residual magnitude directly before scaling, to provide a strong enough gradient.
                        action_l2_penalty = args.action_l2_penalty_coef * (action_new ** 2).sum(dim=-1).mean()
                        actor_loss = (alpha_value * log_prob_new - q_new).mean() + action_l2_penalty

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        alpha_loss_value = None
                        if log_alpha is not None and alpha_optimizer is not None:
                            alpha_loss = -(log_alpha * (log_prob_new.detach() + target_entropy)).mean()
                            alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            alpha_optimizer.step()
                            with torch.no_grad():
                                log_alpha.clamp_(min=np.log(args.min_alpha), max=np.log(args.max_alpha))
                            alpha_loss_value = float(alpha_loss.item())

                        episode_critic_loss.append(critic_loss.item())
                        episode_actor_loss.append(actor_loss.item())

                        if alpha_loss_value is not None and log_alpha is not None:
                            alpha = float(torch.clamp(log_alpha.exp(), min=args.min_alpha, max=args.max_alpha).item())

                        soft_update(target_vision_encoder, agent.vision_encoder, tau)
                        soft_update(target_state_encoder, agent.state_encoder, tau)
                        soft_update(target_q1, agent.q1, tau)
                        soft_update(target_q2, agent.q2, tau)
        
            print(f"Ep {episode}: Reward: {episode_reward:.2f}, Steps: {step}")
            success_history.append(int(episode_success))
            rolling_window = 20
            recent = success_history[-rolling_window:]
            rolling_success = float(np.mean(recent)) if recent else 0.0
            cumulative_success = float(np.mean(success_history)) if success_history else 0.0
            progress = format_progress_bar(episode + 1, num_episodes)
            print(
                f"[progress] seed={args.seed} {progress} "
                f"ep={episode + 1}/{num_episodes} "
                f"success_mean20={rolling_success:.3f} "
                f"success_mean={cumulative_success:.3f}",
                flush=True,
            )
        
            log_dict = {
                "episode": episode, 
                "reward": episode_reward, 
                "success": int(episode_success), 
                "steps": step
            }
            if len(episode_critic_loss) > 0:
                log_dict["critic_loss"] = np.mean(episode_critic_loss)
                log_dict["actor_loss"] = np.mean(episode_actor_loss)
                log_dict["alpha"] = alpha
                log_dict["success_buffer_size"] = len(replay_buffer.success_buffer)
                
            wandb.log(log_dict)

            if (episode + 1) % max(1, args.checkpoint_interval) == 0:
                target_state_dict = {
                    "vision_encoder": target_vision_encoder.state_dict(),
                    "state_encoder": target_state_encoder.state_dict(),
                    "q1": target_q1.state_dict(),
                    "q2": target_q2.state_dict(),
                }
                alpha_state_dict = {
                    "alpha": float(alpha),
                }
                if log_alpha is not None:
                    alpha_state_dict["log_alpha"] = float(log_alpha.detach().cpu().item())
                if alpha_optimizer is not None:
                    alpha_state_dict["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()

                save_checkpoint_bundle(
                    checkpoint_dir=checkpoint_dir,
                    agent=agent,
                    episode=episode,
                    global_step=global_step,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    wandb_run_id=wandb_run_id,
                    replay_state_dict=replay_buffer.state_dict(
                        max_buffer_items=args.checkpoint_replay_size,
                        max_success_items=args.checkpoint_success_replay_size,
                    ),
                    target_state_dict=target_state_dict,
                    alpha_state_dict=alpha_state_dict,
                    success_history=success_history,
                )

        # Ensure latest state is always persisted at the end of training.
        target_state_dict = {
            "vision_encoder": target_vision_encoder.state_dict(),
            "state_encoder": target_state_encoder.state_dict(),
            "q1": target_q1.state_dict(),
            "q2": target_q2.state_dict(),
        }
        alpha_state_dict = {
            "alpha": float(alpha),
        }
        if log_alpha is not None:
            alpha_state_dict["log_alpha"] = float(log_alpha.detach().cpu().item())
        if alpha_optimizer is not None:
            alpha_state_dict["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()

        save_checkpoint_bundle(
            checkpoint_dir=checkpoint_dir,
            agent=agent,
            episode=num_episodes - 1,
            global_step=global_step,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            wandb_run_id=wandb_run_id,
            replay_state_dict=replay_buffer.state_dict(
                max_buffer_items=args.checkpoint_replay_size,
                max_success_items=args.checkpoint_success_replay_size,
            ),
            target_state_dict=target_state_dict,
            alpha_state_dict=alpha_state_dict,
            success_history=success_history,
        )

    except KeyboardInterrupt:
        print("\n[train] interrupted by user. Saving interrupt checkpoint and exiting...")
        target_state_dict = {
            "vision_encoder": target_vision_encoder.state_dict(),
            "state_encoder": target_state_encoder.state_dict(),
            "q1": target_q1.state_dict(),
            "q2": target_q2.state_dict(),
        }
        alpha_state_dict = {
            "alpha": float(alpha),
        }
        if log_alpha is not None:
            alpha_state_dict["log_alpha"] = float(log_alpha.detach().cpu().item())
        if alpha_optimizer is not None:
            alpha_state_dict["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()

        interrupt_episode = max(start_episode - 1, locals().get("last_episode_seen", start_episode - 1))
        save_checkpoint_bundle(
            checkpoint_dir=checkpoint_dir,
            agent=agent,
            episode=interrupt_episode,
            global_step=global_step,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            wandb_run_id=wandb_run_id,
            replay_state_dict=replay_buffer.state_dict(
                max_buffer_items=args.checkpoint_replay_size,
                max_success_items=args.checkpoint_success_replay_size,
            ),
            target_state_dict=target_state_dict,
            alpha_state_dict=alpha_state_dict,
            success_history=locals().get("success_history", []),
        )
        if getattr(wandb, "run", None) is not None:
            wandb.finish(exit_code=130)
        return
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if previous_sigttou_handler is not None and hasattr(signal, "SIGTTOU"):
            signal.signal(signal.SIGTTOU, previous_sigttou_handler)
        if previous_sigttin_handler is not None and hasattr(signal, "SIGTTIN"):
            signal.signal(signal.SIGTTIN, previous_sigttin_handler)
        try:
            env.close()
        except Exception as exc:
            print(f"[warn] env.close() failed: {exc}")

if __name__ == "__main__":
    with LinuxInhibit(reason="SAC Online RL"):
        main()