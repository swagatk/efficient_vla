#!/usr/bin/env python3
import os
import argparse
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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state_dict, action, reward, next_state_dict, done):
        self.buffer.append((state_dict, action, reward, next_state_dict, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # Merge dicts
        states = {k: torch.cat([s[k] for s in state_batch], dim=0) for k in state_batch[0].keys()}
        next_states = {k: torch.cat([s[k] for s in next_state_batch], dim=0) for k in next_state_batch[0].keys()}
        
        actions = torch.stack(action_batch)
        rewards = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
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
    state = batch.get("observation.state").float()
    features = vision_encoder(img)
    state_features = state_encoder(state)
    return torch.cat([features, state_features], dim=-1)


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
    parser.add_argument("--use_visual_prompting", action="store_true")
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "HuggingFaceVLA/smolvla_libero"
    task_name = "libero_10"
    
    # SAC Hyperparameters
    lr = 1e-4
    gamma = 0.99
    tau = 0.005
    # Alpha controls entropy weight. Lowered to avoid destroying continuous control.
    alpha = 0.01
    batch_size = 64
    buffer_size = 50000
    updates_per_step = 1
    start_steps = 1000 # Steps before training
    num_episodes = 200
    max_steps = 520

    # Lowered residual scale to prevent destroying base policy prior entirely
    agent = SACResidualVLAPolicy(base_model_id, device=device, residual_scale=0.03)

    # Lightweight target networks (no duplicated base SmolVLA).
    target_vision_encoder = deepcopy(agent.vision_encoder).to(device)
    target_state_encoder = deepcopy(agent.state_encoder).to(device)
    target_q1 = deepcopy(agent.q1).to(device)
    target_q2 = deepcopy(agent.q2).to(device)
    for module in (target_vision_encoder, target_state_encoder, target_q1, target_q2):
        for param in module.parameters():
            param.requires_grad = False

    for param in agent.base_policy.parameters():
        param.requires_grad = False

    # Optimizers
    actor_params = list(agent.actor_net.parameters()) + list(agent.actor_mean.parameters()) + list(agent.actor_log_std.parameters())
    critic_params = list(agent.vision_encoder.parameters()) + list(agent.state_encoder.parameters()) + list(agent.q1.parameters()) + list(agent.q2.parameters())
    
    actor_optimizer = optim.Adam(actor_params, lr=lr)
    critic_optimizer = optim.Adam(critic_params, lr=lr)

    replay_buffer = ReplayBuffer(buffer_size)

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

    try:
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

        wandb.init(project="efficient_vla_rl", name="sac_residual_online_rl", config=vars(args))
        
        global_step = 0
        for episode in range(num_episodes):
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

            episode_actor_loss = []
            episode_critic_loss = []

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

                if global_step < start_steps:
                    # Completely uniformly random delta actions during warmup
                    action_dim = 7 # Typically 7 for Libero (6 joint + 1 gripper)
                    delta_action = (torch.rand((1, action_dim), device=device) * 2 - 1).to(torch.float32)
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            final_action, base_action, _, _, _ = agent(batch_obs, deterministic=False)

                            # Replace the agent's internal delta with pure uniform random
                            scaled_delta_action = agent.residual_scale * delta_action.to(device)
                            final_action = base_action + scaled_delta_action
                            final_action = torch.clamp(final_action, min=-1.0, max=1.0)
                else:
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            final_action, _, delta_action, _, _ = agent(batch_obs, deterministic=False)

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
                    reward += 10.0
                else:
                    reward -= 0.01

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
                )

                done = done_signal
                obs = next_obs
                episode_reward += reward
                step += 1
                global_step += 1

                # SAC Update Phase
                if len(replay_buffer) > batch_size and global_step > start_steps:
                    for _ in range(updates_per_step):
                        b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                        b_states = {k: v.to(device) for k, v in b_states.items()}
                        b_actions, b_rewards, b_next_states, b_dones = (
                            b_actions.to(device),
                            b_rewards.to(device),
                            {k: v.to(device) for k, v in b_next_states.items()},
                            b_dones.to(device),
                        )

                        with torch.no_grad():
                            # SAC standard: use the target network for Q-value estimation
                            # but the CURRENT policy for the next action and its entropy (log prob)
                            next_features_target = extract_features_with_encoders(
                                b_next_states, target_vision_encoder, target_state_encoder
                            )
                            next_features_actor = agent.extract_features(b_next_states)

                            next_action, next_log_prob, _ = agent.sample_action(next_features_actor)

                            target_q1_val = target_q1(torch.cat([next_features_target, next_action], dim=-1))
                            target_q2_val = target_q2(torch.cat([next_features_target, next_action], dim=-1))
                            target_q = b_rewards + (1 - b_dones) * gamma * (
                                torch.min(target_q1_val, target_q2_val) - alpha * next_log_prob
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
                        q1_new = agent.q1(torch.cat([features.detach(), action_new], dim=-1))
                        q2_new = agent.q2(torch.cat([features.detach(), action_new], dim=-1))
                        q_new = torch.min(q1_new, q2_new)

                        # Add L2 penalty on action_new magnitude to prevent high-frequency destructive tremor
                        action_l2_penalty = 0.05 * (action_new ** 2).sum(dim=-1).mean()
                        actor_loss = (alpha * log_prob_new - q_new).mean() + action_l2_penalty

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        episode_critic_loss.append(critic_loss.item())
                        episode_actor_loss.append(actor_loss.item())

                        soft_update(target_vision_encoder, agent.vision_encoder, tau)
                        soft_update(target_state_encoder, agent.state_encoder, tau)
                        soft_update(target_q1, agent.q1, tau)
                        soft_update(target_q2, agent.q2, tau)
        
            print(f"Ep {episode}: Reward: {episode_reward:.2f}, Steps: {step}")
        
            log_dict = {
                "episode": episode, 
                "reward": episode_reward, 
                "success": int(episode_success), 
                "steps": step
            }
            if len(episode_critic_loss) > 0:
                log_dict["critic_loss"] = np.mean(episode_critic_loss)
                log_dict["actor_loss"] = np.mean(episode_actor_loss)
                
            wandb.log(log_dict)
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[warn] env.close() failed: {exc}")

if __name__ == "__main__":
    with LinuxInhibit(reason="SAC Online RL"):
        main()