import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["HF_HUB_OFFLINE"] = "1"
import sys

# PyAV patch
import av, types
if not hasattr(av, "option"):
    av.option = types.SimpleNamespace(Option=object)

# PyTorch 2.6+ patch
import torch
_orig_load = torch.load
def _p_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _p_load

project_root = "/home/swagat/GIT/efficient_vla"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "supervised_task_specific_lora_adapters"))
sys.path.insert(0, "/home/swagat/lerobot/src")

import eval_lora_policy

import numpy as np
from robosuite.utils.transform_utils import quat2axisangle
import libero.libero as libero_pkg
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_name = "HuggingFaceVLA/smolvla_libero"

print("Loading SmolVLA policy...")
policy = SmolVLAPolicy.from_pretrained(policy_name).to(device)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(policy.config, policy_name)

benchmark = get_benchmark_dict()["libero_10"]()
task_id = 1
task = benchmark.get_task(task_id)
bddl_path = os.path.join(os.path.dirname(libero_pkg.__file__), "bddl_files", task.problem_folder, task.bddl_file)
init_states = benchmark.get_task_init_states(task_id)

print(f"\nEvaluating Task {task_id}: {task.language}")
print(f"Testing Episode 0 (Initial State 0)...")

env = OffScreenRenderEnv(
    bddl_file_name=bddl_path,
    camera_heights=256,
    camera_widths=256,
    ignore_done=True,
)
try:
    obs = env.reset()
    env.set_init_state(init_states[0])
    obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()

    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    policy.reset()
    done = False
    step = 0
    ep_success = False
    instruction = task.language

    while not done and step < 520:
        # Flipped observation
        img_agent = obs["agentview_image"][::-1, ::-1, :].copy()
        img_wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1, :].copy()
        state_np = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"]
        ]).astype(np.float32)

        raw_obs = {
            "pixels": {
                "image": img_agent,
                "image2": img_wrist,
            },
            "agent_pos": state_np,
        }
        policy_obs = preprocess_observation(raw_obs)
        policy_obs["task"] = [instruction]
        batch = preprocessor(policy_obs)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_action = policy.select_action(batch)
                env_action = postprocessor(base_action)
                action_np = env_action.detach().cpu().to(torch.float32).numpy()[0]

        obs, reward, done, info = env.step(action_np)
        step += 1

        is_success = bool(env.check_success()) if hasattr(env, "check_success") else False
        if is_success or reward > 0.9:
            ep_success = True
            done = True

        if step % 50 == 0:
            print(f"  Step {step}/520 | Reward: {reward} | Success: {is_success}", flush=True)

    print(f"\nEpisode 0 Finished: Steps={step}, Success={ep_success}")
finally:
    env.close()
