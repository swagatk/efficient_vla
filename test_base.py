import os
import torch
import numpy as np

# Patch torch.load
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import libero.libero as libero_pkg
from robosuite.utils.transform_utils import quat2axisangle

device = "cuda"
base_policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero").to(torch.bfloat16).to(device)
benchmark = get_benchmark_dict()["libero_10"]()
task = benchmark.get_task(3)
bddl = os.path.join(os.path.dirname(libero_pkg.__file__), "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.reset()
env.set_init_state(benchmark.get_task_init_states(3)[0])
obs = env.env._get_observations()

step, done, success = 0, False, False
while not done and step < 50:
    current_instruction = task.language
    # CORRECT FLIP
    img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
    img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()
    
    img_tensor = torch.from_numpy(img_agent).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    img_tensor_wrist = torch.from_numpy(img_wrist).to(torch.bfloat16).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    # CORRECT STATE ARRAY
    state_np = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    state_tensor = torch.from_numpy(state_np).to(torch.bfloat16).unsqueeze(0).to(device)
    
    processor = base_policy.model.vlm_with_expert.processor
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_action = base_policy.select_action(batch_obs)
            
    action_np = base_action.detach().cpu().to(torch.float32).numpy()[0]
    next_obs, reward, done, info = env.step(action_np)
    
    if "success" in info and info["success"]:
        success = True
        break
    obs = next_obs
    step += 1

print("Successful:", success)
