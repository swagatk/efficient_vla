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

def extract_success(info):
    if not isinstance(info, dict): return False
    for key in ("success", "is_success", "task_success", "episode_success"):
        if key in info:
            val = info[key]
            return bool(val.item() if hasattr(val, "item") else val)
    return False

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=3, help="LIBERO task index to train on")
    parser.add_argument("--use_visual_prompting", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "HuggingFaceVLA/smolvla_libero"
    task_name = "libero_10"
    
    # SAC Hyperparameters
    lr = 1e-4
    gamma = 0.99
    tau = 0.005
    # Alpha controls entropy weight. 0.2 is often too high for robotic continuous control. 0.01-0.05 is more standard.
    alpha = 0.05
    batch_size = 64
    buffer_size = 50000
    updates_per_step = 1
    start_steps = 1000 # Steps before training
    num_episodes = 200
    max_steps = 520

    wandb.init(project="efficient_vla_rl", name="sac_residual_online_rl", config=vars(args))

    # Boosted residual scale so the actor can actually alter the trajectory meaningfully to find success
    agent = SACResidualVLAPolicy(base_model_id, device=device, residual_scale=0.1)
    
    # Target networks
    target_agent = SACResidualVLAPolicy(base_model_id, device=device, residual_scale=0.1)
    target_agent.load_state_dict(agent.state_dict())
    for param in target_agent.parameters():
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
    
    prompter = VisualPromptingWrapper(use_image_box=True, use_text_hint=True, device=device) if args.use_visual_prompting else None
    
    global_step = 0
    for episode in range(num_episodes):
        env.reset()
        env.set_init_state(random.choice(benchmark.get_task_init_states(args.task_id)))
        obs = env.env._get_observations()
        
        episode_reward = 0
        step = 0
        done = False
        
        episode_actor_loss = []
        episode_critic_loss = []
        
        while not done and step < max_steps:
            current_instruction = task.language
            img_agent = obs["agentview_image"][::-1, :, :].copy()
            if prompter:
                prompt_obs = prompter.apply_prompts({"image": img_agent, "instruction": current_instruction}, update_grounding=(step % 5 == 0))
                img_agent = prompt_obs["image"]
            
            img_tensor = torch.from_numpy(img_agent).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            state_tensor = torch.from_numpy(np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][0:1]])).to(torch.bfloat16).unsqueeze(0).to(device)
            
            img_wrist = obs["robot0_eye_in_hand_image"][::-1, :, :].copy()
            img_tensor_wrist = torch.from_numpy(img_wrist).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            # Tokenization mock for base policy forward if needed
            processor = agent.base_policy.model.vlm_with_expert.processor
            text_out = processor(text=current_instruction, return_tensors='pt')
            
            batch_obs = {
                'observation.images.image': img_tensor,
                'observation.images.image2': img_tensor_wrist,
                'observation.state': state_tensor,
                'observation.language.tokens': text_out['input_ids'].to(device),
                'observation.language.attention_mask': text_out['attention_mask'].to(device).bool(),
                'language_instruction': [current_instruction]
            }
            
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

            action_np = final_action.detach().cpu().numpy()[0]
            next_obs, reward, done, info = env.step(action_np)
            step_success = extract_success(info)
            done = bool(done or step_success)
            
            # Reward shaping
            if step_success: reward += 10.0
            else: reward -= 0.01

            # Prepare next state
            next_img_agent = next_obs["agentview_image"][::-1, :, :].copy()
            if prompter: next_img_agent = prompter.apply_prompts({"image": next_img_agent, "instruction": current_instruction}, update_grounding=False)["image"]
            next_img_tensor = torch.from_numpy(next_img_agent).to(torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            next_state_tensor = torch.from_numpy(np.concatenate([next_obs["robot0_joint_pos"], next_obs["robot0_gripper_qpos"][0:1]])).to(torch.float32).unsqueeze(0).to(device)
            next_batch_obs = {'observation.images.image': next_img_tensor, 'observation.state': next_state_tensor}

            # Store to buffer as uint8 natively to drastically cut down RAM usage (OOM prevention)
            img_uint8 = torch.from_numpy(img_agent.copy()).permute(2, 0, 1).unsqueeze(0)
            next_img_uint8 = torch.from_numpy(next_img_agent.copy()).permute(2, 0, 1).unsqueeze(0)

            replay_buffer.push(
                {'observation.images.image': img_uint8, 'observation.state': state_tensor.detach().cpu()},
                delta_action.detach().cpu()[0],
                reward,
                {'observation.images.image': next_img_uint8, 'observation.state': next_state_tensor.detach().cpu()},
                done
            )

            obs = next_obs
            episode_reward += reward
            step += 1
            global_step += 1
            
            # SAC Update Phase
            if len(replay_buffer) > batch_size and global_step > start_steps:
                for _ in range(updates_per_step):
                    b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                    b_states = {k: v.to(device) for k, v in b_states.items()}
                    b_actions, b_rewards, b_next_states, b_dones = b_actions.to(device), b_rewards.to(device), {k: v.to(device) for k, v in b_next_states.items()}, b_dones.to(device)

                    with torch.no_grad():
                        next_features = target_agent.extract_features(b_next_states)
                        next_action, next_log_prob, _ = target_agent.sample_action(next_features)
                        target_q1 = target_agent.q1(torch.cat([next_features, next_action], dim=-1))
                        target_q2 = target_agent.q2(torch.cat([next_features, next_action], dim=-1))
                        target_q = b_rewards + (1 - b_dones) * gamma * (torch.min(target_q1, target_q2) - alpha * next_log_prob)

                    features = agent.extract_features(b_states)
                    q1 = agent.q1(torch.cat([features, b_actions], dim=-1))
                    q2 = agent.q2(torch.cat([features, b_actions], dim=-1))
                    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    # Actor update (using reparameterization trick)
                    features = agent.extract_features(b_states) # Detached for actor if needed, but here we backprop through vision for actor too? Usually vision is optimized by critic.
                    action_new, log_prob_new, _ = agent.sample_action(features.detach())
                    q1_new = agent.q1(torch.cat([features.detach(), action_new], dim=-1))
                    q2_new = agent.q2(torch.cat([features.detach(), action_new], dim=-1))
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss = (alpha * log_prob_new - q_new).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    episode_critic_loss.append(critic_loss.item())
                    episode_actor_loss.append(actor_loss.item())

                    soft_update(target_agent, agent, tau)
        
        print(f"Ep {episode}: Reward: {episode_reward:.2f}, Steps: {step}")
        
        log_dict = {
            "episode": episode, 
            "reward": episode_reward, 
            "success": int(step_success), 
            "steps": step
        }
        if len(episode_critic_loss) > 0:
            log_dict["critic_loss"] = np.mean(episode_critic_loss)
            log_dict["actor_loss"] = np.mean(episode_actor_loss)
            
        wandb.log(log_dict)

if __name__ == "__main__":
    with LinuxInhibit(reason="SAC Online RL"):
        main()