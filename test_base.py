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

from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation

device = "cuda"
base_policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero").to(torch.bfloat16).to(device)
preprocessor, postprocessor = make_pre_post_processors(base_policy.config, "HuggingFaceVLA/smolvla_libero")

benchmark = get_benchmark_dict()["libero_10"]()
task = benchmark.get_task(3)
bddl = os.path.join(os.path.dirname(libero_pkg.__file__), "bddl_files", task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.reset()
env.set_init_state(benchmark.get_task_init_states(3)[0])
obs = env.env._get_observations()

step, done, success = 0, False, False
while not done and step < 400:
    current_instruction = task.language
    img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
    img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()
    
    state_np = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    
    raw_obs = {
        "pixels": {
            "image": img_agent,
            "image2": img_wrist,
        },
        "agent_pos": state_np.astype(np.float32),
    }
    
    policy_obs = preprocess_observation(raw_obs)
    policy_obs["task"] = [current_instruction]
    batch_obs = preprocessor(policy_obs)
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_action = base_policy.select_action(batch_obs)
            
    env_action = postprocessor(base_action)
    action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]
    next_obs, reward, done, info = env.step(action_np)
    
    if env.check_success():
        success = True
        break
    obs = next_obs
    step += 1

print("Successful:", success)
