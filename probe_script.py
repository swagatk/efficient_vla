import os
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

# PATCH
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import libero.libero as libero_pkg
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PreTrainedPolicy
except ImportError:
    try:
        from lerobot.common.policies.pretrained import PreTrainedPolicy
    except ImportError:
        try:
            from lerobot.policies.pretrained import PreTrainedPolicy
        except ImportError:
            from lerobot.policies import PreTrainedPolicy

def extract_success(info):
    if not isinstance(info, dict): return False
    for key in ("success", "is_success", "task_success", "episode_success"):
        if key in info:
            val = info[key]
            return bool(val.item() if hasattr(val, "item") else val)
    return False

def run_probe(policy, env, task, benchmark, device, num_episodes=5, max_steps=520, residual_scale=0.0):
    successes = 0
    results = []
    
    init_states = benchmark.get_task_init_states(3)
    processor = policy.model.vlm_with_expert.processor
    
    print(f"\nRunning probe with residual_scale={residual_scale}...")
    
    for ep in range(num_episodes):
        env.reset()
        env.set_init_state(random.choice(init_states))
        obs = env.env._get_observations()
        
        total_reward = 0
        steps = 0
        done = False
        success = False
        
        while not done and steps < max_steps:
            img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
            img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()
            
            img_tensor = torch.from_numpy(img_agent).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            img_tensor_wrist = torch.from_numpy(img_wrist).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            state_np = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
            state_tensor = torch.from_numpy(state_np).to(torch.bfloat16).unsqueeze(0).to(device)
            
            current_instruction = task.language
            text_out = processor(text=current_instruction, return_tensors='pt')
            
            batch_obs = {
                'observation.images.image': img_tensor,
                'observation.images.image2': img_tensor_wrist,
                'observation.state': state_tensor,
                'observation.language.tokens': text_out['input_ids'].to(device),
                'observation.language.attention_mask': text_out['attention_mask'].to(device).bool(),
                'language_instruction': [current_instruction]
            }
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    base_action = policy.select_action(batch_obs)
            
            if residual_scale > 0:
                # SAC warmup worst-case small random residual around +/-0.03 (sample uniform each step)
                delta_action = (torch.rand((1, 7), device=device) * 2 - 1) * residual_scale
                final_action = base_action + delta_action
                final_action = torch.clamp(final_action, min=-1.0, max=1.0)
            else:
                final_action = base_action
            
            action_np = final_action.detach().cpu().to(torch.float32).numpy()[0]
            obs, reward, env_done, info = env.step(action_np)
            
            total_reward += reward
            steps += 1
            success = extract_success(info)
            if success or env_done:
                done = True
                
        successes += int(success)
        results.append((success, steps, total_reward))
        print(f"Episode {ep+1}: Success={success}, Steps={steps}, Reward={total_reward:.4f}")
        
    print(f"Success Rate: {successes/num_episodes:.2f}")
    return successes/num_episodes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "HuggingFaceVLA/smolvla_libero"
    task_name = "libero_10"
    task_id = 3
    
    print(f"Loading policy: {base_model_id}")
    policy = PreTrainedPolicy.from_pretrained(base_model_id)
    policy.to(torch.bfloat16)
    policy.to(device)
    policy.eval()
    
    benchmark = get_benchmark_dict()[task_name]()
    task = benchmark.get_task(task_id)
    bddl_file_path = os.path.join(os.path.dirname(libero_pkg.__file__), "bddl_files", task.problem_folder, task.bddl_file)
    
    env = OffScreenRenderEnv(bddl_file_name=bddl_file_path, camera_heights=256, camera_widths=256)
    
    # Run 1: Pure base action
    run_probe(policy, env, task, benchmark, device, num_episodes=5, max_steps=520, residual_scale=0.0)
    
    # Run 2: Constant residual perturbation scale 0.03
    run_probe(policy, env, task, benchmark, device, num_episodes=5, max_steps=520, residual_scale=0.03)

if __name__ == "__main__":
    main()
