import os
import torch
import numpy as np
import libero.libero as libero
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle
from sac_residual_agent import SACResidualVLAPolicy

def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

def run_episodes(env, agent, num_episodes, residual_scale=0.0):
    all_success = []
    all_steps = []
    all_rewards = []
    all_base_norms = []
    all_saturations = []

    for ep in range(num_episodes):
        obs = env.reset()
        init_state = env.get_cpu_state()
        env.set_init_state(init_state)
        obs = env.reset()

        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())

        ep_reward = 0
        ep_steps = 0
        ep_success = False
        ep_base_norms = []
        ep_abs_actions = []

        for step in range(520):
            eef_pos = obs['robot0_eef_pos']
            eef_quat = obs['robot0_eef_quat']
            eef_axisangle = quat2axisangle(eef_quat)
            gripper_qpos = obs['robot0_gripper_qpos']
            state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos])
            
            with torch.no_grad():
                image_agent = torch.from_numpy(obs['agentview_image'].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                image_eye = torch.from_numpy(obs['robot0_eye_in_hand_image'].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                image_agent = torch.flip(image_agent, [2])
                image_eye = torch.flip(image_eye, [2])
                
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(agent.device)
                base_action = agent.model.get_action(image_agent.to(agent.device), image_eye.to(agent.device), state_tensor)
                base_action = base_action.detach().cpu().numpy()[0]
                
                if residual_scale > 0:
                    res = np.random.uniform(-residual_scale, residual_scale, size=base_action.shape)
                else:
                    res = 0
                
                action = base_action + res
                action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            ep_base_norms.append(np.linalg.norm(base_action))
            ep_abs_actions.append(np.abs(action))
            
            if env.check_success():
                ep_success = True
                break
            if done:
                break
        
        all_success.append(ep_success)
        all_steps.append(ep_steps)
        all_rewards.append(ep_reward)
        all_base_norms.append(np.mean(ep_base_norms))
        ep_abs_actions_arr = np.array(ep_abs_actions)
        all_saturations.append(np.mean(ep_abs_actions_arr > 0.98))
        print(f"Ep {ep} | Success: {ep_success} | Steps: {ep_steps} | Base Norm: {all_base_norms[-1]:.4f} | Sat: {all_saturations[-1]:.4f}")

    return {
        "success_rate": np.mean(all_success),
        "mean_steps": np.mean(all_steps),
        "mean_reward": np.mean(all_rewards),
        "mean_base_norm": np.mean(all_base_norms),
        "mean_saturation": np.mean(all_saturations)
    }

def main():
    benchmark_dict = get_benchmark_dict()
    benchmark = benchmark_dict["libero_10"]()
    task_id = 3
    task = benchmark.get_task(task_id)
    
    # Path found: /home/swagat/lerobot-libero/libero/libero/bddl_files
    bddl_root = "/home/swagat/lerobot-libero/libero/libero/bddl_files"
    full_bddl_path = os.path.join(bddl_root, "libero_10", task.problem_folder)

    env_args = {
        "bddl_file_name": full_bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    
    env = OffScreenRenderEnv(
        **env_args,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACResidualVLAPolicy("HuggingFaceVLA/smolvla_libero", device=device, residual_scale=0.03)
    
    print(f"\nEvaluating Task: {task.name}")
    print("\n--- Running 5 episodes with Base Policy only ---")
    base_metrics = run_episodes(env, agent, 5, residual_scale=0.0)
    print(f"\nBase Policy Metrics: {base_metrics}")

    print("\n--- Running 5 episodes with Random Residual (scale 0.03) ---")
    resid_metrics = run_episodes(env, agent, 5, residual_scale=0.03)
    print(f"\nResidual (0.03) Metrics: {resid_metrics}")

    env.close()

if __name__ == "__main__":
    main()
