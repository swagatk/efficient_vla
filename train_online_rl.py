#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

# Fix for PyTorch 2.6+ weights_only=True default when loading libero init states
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# LIBERO imports
import libero.libero as libero_pkg
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

from rl_residual_agent import ResidualVLAPolicy
from linux_inhibit import LinuxInhibit
import atexit

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0
    values = values + [0] # Dummy next value for the end of the episode
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def main():
    # Initialize sleep prevention and performance mode
    inhibit = LinuxInhibit(reason="Online RL Training")
    inhibit.__enter__()
    atexit.register(inhibit.__exit__, None, None, None)

    # --- Configurations ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "HuggingFaceVLA/smolvla_libero"
    learning_rate = 1e-4
    ppo_epochs = 4
    bc_anchor_coef = 0.5   # Weight to prevent policy drift
    num_episodes = 100     # Total RL episodes
    max_steps = 520
    task_name = "libero_10"
    checkpoint_dir = "checkpoints"
    checkpoint_interval = 5
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    
    run_id = wandb.util.generate_id()
    start_episode = 0
    checkpoint = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        run_id = checkpoint.get('wandb_run_id', run_id)
        start_episode = checkpoint['episode'] + 1

    wandb.init(
        project="efficient_vla_rl",
        name="lightweight_online_rl",
        id=run_id,
        resume="allow",
        config={
            "learning_rate": learning_rate,
            "ppo_epochs": ppo_epochs,
            "bc_anchor_coef": bc_anchor_coef,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "task_name": task_name
        }
    )

    print(f"Initializing Lightweight Online RL on {device}...")
    
    # 1. Initialize Agent
    agent = ResidualVLAPolicy(base_policy_path=base_model_id, device=device)
    
    # Only optimize the residual components, leaving the base VLA completely untouched
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    torch.cuda.empty_cache()
    
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        agent.load_trainable_state_dict(checkpoint['agent_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming from episode {start_episode + 1}")

    # 2. Initialize LIBERO Environment
    benchmark = get_benchmark_dict()[task_name]()
    task = benchmark.get_task(0) # Train on the first task for this prototype
    benchmark_root = os.path.dirname(libero_pkg.__file__)
    bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)
    
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=256,
        camera_widths=256,
    )
    
    # 3. RL Training Loop
    for episode in range(start_episode, num_episodes):
        print(f"\n--- Starting Episode {episode + 1}/{num_episodes} ---")
        
        env.reset()
        init_states = benchmark.get_task_init_states(0)
        env.set_init_state(init_states[0])
        obs = env.env._get_observations()
        
        rollout_buffer = []

        step = 0
        done = False
        episode_reward = 0.0
        prev_dist = None
        
        # --- A. ROLLOUT PHASE ---
        while not done and step < max_steps:
            # Grab image and flip it upside down (MuJoCo GL convention)
            img_agent = obs["agentview_image"][::-1, :, :].copy() # Flip upside down
            
            current_instruction = task.language
            
            # Format image to LeRobot tensor shape [1, C, H, W]
            img_tensor_agent = torch.from_numpy(img_agent).to(torch.bfloat16)
            if img_tensor_agent.max() > 1.0: img_tensor_agent /= 255.0
            img_tensor_agent = img_tensor_agent.permute(2, 0, 1).unsqueeze(0).to(device)

            img_wrist = obs["robot0_eye_in_hand_image"][::-1, :, :].copy()
            img_tensor_wrist = torch.from_numpy(img_wrist).to(torch.bfloat16)
            if img_tensor_wrist.max() > 1.0: img_tensor_wrist /= 255.0
            img_tensor_wrist = img_tensor_wrist.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Construct 8-dim state: 7 joints + 1 gripper
            joint_pos = obs["robot0_joint_pos"]
            gripper_pos = obs["robot0_gripper_qpos"][0:1] # Just take first finger as proxy
            state_np = np.concatenate([joint_pos, gripper_pos])
            state_tensor = torch.from_numpy(state_np).to(torch.bfloat16).unsqueeze(0).to(device)
            
            # Dummy instruction batch expected by LeRobot
            # Tokenize language instruction
            processor = agent.base_policy.model.vlm_with_expert.processor
            text_out = processor(text=current_instruction, return_tensors='pt')
            
            batch = {
                'observation.images.image': img_tensor_agent,
                'observation.images.image2': img_tensor_wrist,
                'observation.state': state_tensor,
                'observation.language.tokens': text_out['input_ids'].to(device),
                'observation.language.attention_mask': text_out['attention_mask'].to(device).bool(),
                'language_instruction': [current_instruction]
            }
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                final_action, delta_mean, value, log_prob, delta_action = agent(batch, deterministic=False)
            
            # Step environment
            action_np = final_action.cpu().numpy()[0]
            next_obs, reward, done, info = env.step(action_np)
            
            # Sparse Reward Shaping
            # --- Dense Reward Shaping ---
            if done: 
                reward += 10.0 # Huge bonus for completing the task
            else:
                reward -= 0.01
                
                # TODO: Week 6 Plan - Add Dense Reward Shaping here!
                # e.g., reward += distance_to_target_bonus
                # e.g., reward += 1.0 if gripper_closed_around_object else 0.0
                try:
                    # 1. Distance to Object Shaping
                    eef_site_id = env.env.sim.model.site_name2id("gripper0_grip_site")
                    eef_pos = env.env.sim.data.site_xpos[eef_site_id]
                    
                    distances = []
                    for obj in env.env.objects:
                        obj_id = env.env.sim.model.body_name2id(obj.root_body)
                        obj_pos = env.env.sim.data.body_xpos[obj_id]
                        dist = np.linalg.norm(eef_pos - obj_pos)
                        distances.append(dist)
                    
                    if distances:
                        current_min_dist = min(distances)
                        
                        if prev_dist is not None:
                            # Potential-based shaping: reward agent for moving closer
                            dist_improvement = prev_dist - current_min_dist
                            reward += 5.0 * dist_improvement
                        
                        prev_dist = current_min_dist
                        
                        # 2. Gripper Closed Shaping (Bonus if grasping an object)
                        gripper_qpos = next_obs.get("robot0_gripper_qpos", [0.0])
                        if sum(gripper_qpos) > 0.01 and current_min_dist < 0.05:
                            reward += 0.02
                except Exception as e:
                    pass # Fail gracefully if MuJoCo sim structure differs

            # Store the transition in the buffer
            rollout_buffer.append((batch, log_prob, value.squeeze(), reward, done, delta_action))

            obs = next_obs
            episode_reward += reward
            step += 1
            
        print(f"Episode length: {step}, Total Reward: {episode_reward:.2f}")
        
        # --- B. UPDATE PHASE (PPO + BC ANCHOR) ---
        ep_value_loss, ep_bc_loss, ep_ppo_loss = 0.0, 0.0, 0.0
        
        if len(rollout_buffer) > 0:
            # 1. Unpack buffer and calculate advantages
            batches, old_log_probs, old_values, rewards, _, old_delta_actions = zip(*rollout_buffer)

            values_np = [v.item() for v in old_values]
            advantages = compute_advantages(list(rewards), values_np)

            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
            returns_tensor = advantages_tensor + torch.tensor(values_np, dtype=torch.float32).to(device)
            
            # Normalize advantages for stability
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
            old_log_probs_tensor = torch.stack(list(old_log_probs)).detach().view(-1)

            for epoch in range(ppo_epochs):
                # Re-evaluate policy on all stored batches to get the new log_probs and values
                new_log_probs_list, new_values_list, new_delta_means_list = [], [], []
                for b, old_delta_act in zip(batches, old_delta_actions):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Forward pass to get new distribution means (deterministic=True avoids resampling)
                        _, delta_mean, value, _, _ = agent(b, deterministic=True)
                        # Re-calculate log_prob of the OLD action under the NEW distribution
                        dist = torch.distributions.Normal(delta_mean, 0.05) # action_std = 0.05
                        new_log_prob = dist.log_prob(old_delta_act).sum(dim=-1)
                        
                    new_log_probs_list.append(new_log_prob)
                    new_values_list.append(value.squeeze())
                    new_delta_means_list.append(delta_mean)

                new_log_probs_tensor = torch.stack(new_log_probs_list).view(-1)
                new_values_tensor = torch.stack(new_values_list).view(-1)
                new_delta_means_tensor = torch.cat(new_delta_means_list) if new_delta_means_list[0].dim() > 1 else torch.stack(new_delta_means_list)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Surrogate PPO Loss
                    ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
                    clip_adv = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages_tensor
                    ppo_loss = -torch.min(ratio * advantages_tensor, clip_adv).mean()
                    
                    # Critic / Value Loss
                    value_loss = nn.MSELoss()(new_values_tensor, returns_tensor.to(torch.bfloat16))
                    
                    # Behavior Cloning Anchor Loss
                    bc_loss = agent.compute_bc_anchor_loss(new_delta_means_tensor)
                    
                    total_loss = ppo_loss + (0.5 * value_loss) + (bc_anchor_coef * bc_loss)
                
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping to prevent exploding updates
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                optimizer.step()
                
                ep_value_loss = value_loss.item()
                ep_bc_loss = bc_loss.item()
                ep_ppo_loss = ppo_loss.item()
                
            print(f"Update complete. Value Loss: {ep_value_loss:.4f}, BC Anchor Loss: {ep_bc_loss:.4f}, PPO Loss: {ep_ppo_loss:.4f}")

        wandb.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "episode_length": step,
            "value_loss": ep_value_loss,
            "bc_anchor_loss": ep_bc_loss,
            "ppo_loss": ep_ppo_loss,
            "success": 1.0 if done else 0.0
        })

        if (episode + 1) % checkpoint_interval == 0:
            print(f"Saving checkpoint at episode {episode + 1}...")
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.get_trainable_state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_run_id': wandb.run.id,
            }, checkpoint_path)

    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()