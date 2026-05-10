#!/usr/bin/env python3
import os
import argparse
import sys
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
import random

from ppo_residual_agent import ResidualVLAPolicy
from linux_inhibit import LinuxInhibit
import atexit
import signal
from visual_prompt_wrapper import VisualPromptingWrapper

def compute_advantages(rewards, values, episode_ends, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0
    values = values + [0]  # Dummy next value for the end of the final episode
    for i in reversed(range(len(rewards))):
        next_non_terminal = 0.0 if episode_ends[i] else 1.0
        delta = rewards[i] + gamma * values[i + 1] * next_non_terminal - values[i]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages.insert(0, gae)
    return advantages


def extract_success(info):
    """Best-effort extraction of a task success flag from env info."""
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


def install_signal_handlers():
    """Use hard Ctrl+C termination and graceful SIGTERM shutdown."""
    shutdown_state = {"requested": False}

    def request_shutdown(sig, frame):
        shutdown_state["requested"] = True
        signal_name = signal.Signals(sig).name
        print(f"\n[!] {signal_name} received. Cleaning up before exit...")
        raise KeyboardInterrupt

    # Use the OS default for Ctrl+C so the process dies immediately even if Python is
    # currently inside a long-running native call from Torch, MuJoCo, or Transformers.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Keep SIGTERM graceful so `kill <pid>` still goes through normal cleanup when possible.
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.siginterrupt(signal.SIGTERM, True)

    return shutdown_state

def main(inhibitor=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO task index to train on (0-9)")
    parser.add_argument(
        "--use_visual_prompting",
        action="store_true",
        help="Enable the GroundingDINO visual prompt wrapper during RL training.",
    )
    args = parser.parse_args()
    shutdown_state = install_signal_handlers()

    # --- Configurations ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "HuggingFaceVLA/smolvla_libero"
    learning_rate = 1e-5
    ppo_epochs = 4
    bc_anchor_coef = 0.5   # Weight to prevent policy drift
    num_episodes = 100     # Total RL episodes
    max_steps = 520
    warmup_deterministic_episodes = 8
    task_name = "libero_10"
    checkpoint_dir = "checkpoints"
    checkpoint_interval = 5
    update_every_n_episodes = 4  # Accumulate experience over 4 episodes before updating
    
    checkpoint_dir = f"checkpoints_task_{args.task_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"latest_checkpoint_task_{args.task_id}.pt")
    
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
            "warmup_deterministic_episodes": warmup_deterministic_episodes,
            "task_name": task_name,
            "task_id": args.task_id,
            "use_visual_prompting": args.use_visual_prompting,
        }
    )

    print(f"Initializing Lightweight Online RL on {device}...")
    
    # Initialize Visual Prompter
    prompter = None
    if args.use_visual_prompting:
        print("Initializing Visual Prompter...")
        prompter = VisualPromptingWrapper(use_image_box=True, use_text_hint=True, device=device)
    
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
    task = benchmark.get_task(args.task_id)
    benchmark_root = os.path.dirname(libero_pkg.__file__)
    bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)
    
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=256,
        camera_widths=256,
    )
    
    rollout_buffer = []
    
    # 3. RL Training Loop
    try:
        for episode in range(start_episode, num_episodes):
            if shutdown_state["requested"]:
                raise KeyboardInterrupt

            print(f"\n--- Starting Episode {episode + 1}/{num_episodes} ---")
            
            env.reset()
            init_states = benchmark.get_task_init_states(args.task_id)
            env.set_init_state(random.choice(init_states))
            obs = env.env._get_observations()
            
    
            step = 0
            done = False
            episode_reward = 0.0
            episode_success = False
            prev_dist = None
            episode_delta_norms = []
            episode_scaled_delta_norms = []
            episode_action_drift_norms = []
            
            # --- A. ROLLOUT PHASE ---
            while not done and step < max_steps:
                if shutdown_state["requested"]:
                    raise KeyboardInterrupt

                # Grab image and flip it upside down (MuJoCo GL convention)
                img_agent = obs["agentview_image"][::-1, :, :].copy() # Flip upside down
                
                current_instruction = task.language
                
                # Apply visual prompting
                if prompter is not None:
                    prompt_obs = {
                        "image": img_agent,
                        "instruction": current_instruction
                    }
                    # Run object grounding every 5 steps to save latency and compute
                    prompt_obs = prompter.apply_prompts(prompt_obs, update_grounding=(step % 5 == 0))
                    img_agent = prompt_obs["image"]
                    current_instruction = prompt_obs["instruction"]
                
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
                
                deterministic_rollout = episode < warmup_deterministic_episodes
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    final_action, base_action, delta_mean, value, log_prob, delta_action, scaled_delta_action = agent(
                        batch,
                        deterministic=deterministic_rollout,
                    )

                episode_delta_norms.append(delta_action.norm(dim=-1).mean().item())
                episode_scaled_delta_norms.append(scaled_delta_action.norm(dim=-1).mean().item())
                episode_action_drift_norms.append((final_action - base_action).norm(dim=-1).mean().item())
                
                # Step environment
                action_np = final_action.detach().cpu().numpy()[0]
                next_obs, reward, done, info = env.step(action_np)
                step_success = extract_success(info)
                done = bool(done or step_success)
                episode_success = episode_success or step_success
                episode_end = done or (step + 1 >= max_steps)
                
                # --- Dense Reward Shaping ---
                if step_success:
                    reward += 10.0 # Huge bonus for completing the task
                else:
                    reward -= 0.01
                    
                    try:
                        # 1. Distance to Object Shaping
                        eef_site_id = env.env.sim.model.site_name2id("gripper0_grip_site")
                        eef_pos = env.env.sim.data.site_xpos[eef_site_id]
                        
                        distances = []
                        target_distances = []
                        for obj in env.env.objects:
                            obj_id = env.env.sim.model.body_name2id(obj.root_body)
                            obj_pos = env.env.sim.data.body_xpos[obj_id]
                            dist = np.linalg.norm(eef_pos - obj_pos)
                            distances.append(dist)
                            
                            # Simple heuristic: check if the object name is in the instruction
                            obj_name = obj.name.lower().replace('_', ' ')
                            obj_name = ''.join(c for c in obj_name if not c.isdigit()).strip()
                            if obj_name in current_instruction.lower():
                                target_distances.append(dist)
                        
                        # Only shape toward explicitly matched target objects.
                        active_distances = target_distances
                        
                        if active_distances:
                            current_min_dist = min(active_distances)
                            
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
                rollout_batch = {
                    'observation.images.image': batch['observation.images.image'].detach(),
                    'observation.state': batch['observation.state'].detach(),
                }
                rollout_buffer.append(
                    (
                        rollout_batch,
                        log_prob.detach(),
                        value.squeeze().detach(),
                        reward,
                        episode_end,
                        delta_action.detach(),
                    )
                )
    
                obs = next_obs
                episode_reward += reward
                step += 1
                
            print(f"Episode length: {step}, Total Reward: {episode_reward:.2f}")
            
            # --- B. UPDATE PHASE (PPO + BC ANCHOR) ---
            ep_value_loss, ep_bc_loss, ep_ppo_loss = 0.0, 0.0, 0.0
            
            if len(rollout_buffer) > 0 and (episode + 1) % update_every_n_episodes == 0:
                # 1. Unpack buffer and calculate advantages
                batches, old_log_probs, old_values, rewards, episode_ends, old_delta_actions = zip(*rollout_buffer)
    
                values_np = [v.item() for v in old_values]
                advantages = compute_advantages(list(rewards), values_np, list(episode_ends))
    
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
                returns_tensor = advantages_tensor + torch.tensor(values_np, dtype=torch.float32).to(device)
                
                # Normalize advantages for stability
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
                old_log_probs_tensor = torch.stack(list(old_log_probs)).detach().view(-1)
                
                buffer_size = len(rollout_buffer)
                ppo_batch_size = 64 # Process chunks of 64 to prevent CUDA OOM

                for epoch in range(ppo_epochs):
                    if shutdown_state["requested"]:
                        raise KeyboardInterrupt

                    # Shuffle the buffer for mini-batching
                    indices = np.random.permutation(buffer_size)
                    
                    epoch_value_loss, epoch_bc_loss, epoch_ppo_loss, epoch_entropy = 0.0, 0.0, 0.0, 0.0
                    num_batches = 0
                    
                    for start_idx in range(0, buffer_size, ppo_batch_size):
                        if shutdown_state["requested"]:
                            raise KeyboardInterrupt

                        batch_indices = indices[start_idx:start_idx + ppo_batch_size].tolist()
                        
                        # Vectorized update using the fast residual-only forward pass
                        batch_dict = {
                            'observation.images.image': torch.cat([batches[i]['observation.images.image'] for i in batch_indices]),
                            'observation.state': torch.cat([batches[i]['observation.state'] for i in batch_indices]),
                        }
                        old_delta_acts_tensor = torch.cat([old_delta_actions[i] for i in batch_indices], dim=0)

                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            delta_means, values = agent.forward_residual(batch_dict)
                            action_std = torch.exp(agent.log_std)
                            dist = torch.distributions.Normal(delta_means, action_std)
                            new_log_probs_tensor = dist.log_prob(old_delta_acts_tensor).sum(dim=-1)
                        
                        new_values_tensor = values.view(-1)
                        new_delta_means_tensor = delta_means
        
                        mb_advantages = advantages_tensor[batch_indices]
                        mb_returns = returns_tensor[batch_indices]
                        mb_old_log_probs = old_log_probs_tensor[batch_indices]
        
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            ratio = torch.exp(new_log_probs_tensor - mb_old_log_probs)
                            clip_adv = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * mb_advantages
                            ppo_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()
                            
                            value_loss = nn.MSELoss()(new_values_tensor, mb_returns.to(torch.bfloat16))
                            bc_loss = agent.compute_bc_anchor_loss(new_delta_means_tensor)
                            
                            # Add Entropy Bonus to prevent variance collapse and exploding ratios
                            entropy = dist.entropy().sum(dim=-1).mean()
                            entropy_coef = 0.001
                            
                            total_loss = ppo_loss + (0.5 * value_loss) + (bc_anchor_coef * bc_loss) - (entropy_coef * entropy)
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                        optimizer.step()
                        
                        epoch_value_loss += value_loss.item()
                        epoch_bc_loss += bc_loss.item()
                        epoch_ppo_loss += ppo_loss.item()
                        epoch_entropy += entropy.item()
                        num_batches += 1
                        
                    # Log the average metrics over the final epoch
                    ep_value_loss = epoch_value_loss / num_batches
                    ep_bc_loss = epoch_bc_loss / num_batches
                    ep_ppo_loss = epoch_ppo_loss / num_batches
                    ep_entropy = epoch_entropy / num_batches
                    
                rollout_buffer = [] # Clear the buffer after updating
                print(f"Update complete. Value Loss: {ep_value_loss:.4f}, BC Anchor Loss: {ep_bc_loss:.4f}, PPO Loss: {ep_ppo_loss:.4f}, Entropy: {ep_entropy:.4f}")
    
                wandb.log({
                    "episode": episode + 1,
                    "reward": episode_reward,
                    "episode_length": step,
                    "value_loss": ep_value_loss,
                    "bc_anchor_loss": ep_bc_loss,
                    "ppo_loss": ep_ppo_loss,
                    "entropy": ep_entropy,
                    "success": 1.0 if episode_success else 0.0,
                    "delta_action_norm": float(np.mean(episode_delta_norms)) if episode_delta_norms else 0.0,
                    "scaled_delta_action_norm": float(np.mean(episode_scaled_delta_norms)) if episode_scaled_delta_norms else 0.0,
                    "action_drift_norm": float(np.mean(episode_action_drift_norms)) if episode_action_drift_norms else 0.0,
                })
            else:
                # Log just the environment metrics when an update doesn't occur
                wandb.log({
                    "episode": episode + 1,
                    "reward": episode_reward,
                    "episode_length": step,
                    "success": 1.0 if episode_success else 0.0,
                    "delta_action_norm": float(np.mean(episode_delta_norms)) if episode_delta_norms else 0.0,
                    "scaled_delta_action_norm": float(np.mean(episode_scaled_delta_norms)) if episode_scaled_delta_norms else 0.0,
                    "action_drift_norm": float(np.mean(episode_action_drift_norms)) if episode_action_drift_norms else 0.0,
                })
    
            if (episode + 1) % checkpoint_interval == 0:
                print(f"Saving checkpoint at episode {episode + 1}...")
                torch.save({
                    'episode': episode,
                    'agent_state_dict': agent.get_trainable_state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'wandb_run_id': wandb.run.id,
                }, checkpoint_path)
                
    except KeyboardInterrupt:
        print("\n[!] Shutdown requested. Cleaning up before exit...")
        return 130
    except Exception as e:
        print(f"\n[!] Training stopped due to error: {e}")
        return 1
    finally:
        print("[*] Closing environment...")
        try:
            env.close()
        except Exception as e:
            print(f"    Error closing environment: {e}")
            
        print("[*] Finalizing WandB (this may take a moment to upload logs)...")
        wandb.finish()

    print("[*] Exiting script.")
    return 0

if __name__ == "__main__":
    inhibitor = LinuxInhibit("RL Training")
    with inhibitor:
        sys.exit(main(inhibitor))