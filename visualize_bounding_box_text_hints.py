import os
import random
import gc
import shutil
import json
import warnings
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import yaml
from pathlib import Path
import gymnasium as gym

# Suppress the PyTorch FutureWarnings triggered by GroundingDINO
warnings.filterwarnings("ignore", message=".*weights_only.*")

# Fix for PyTorch 2.6+ where weights_only=True by default breaks LIBERO's torch.load() of init states
# Monkey-patch torch.load to default to weights_only=False for trusted local environments
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Attempt to import LIBERO and the GroundingDINO wrapper
try:
    import libero.libero as libero_pkg
    from libero.libero.benchmark import get_benchmark_dict
    from visual_prompt_wrapper import VisualPromptingWrapper
except ImportError as e:
    print(f"Import Error: {e}. Please ensure libero and its dependencies are installed.")
    libero_pkg = None
    get_benchmark_dict = None
    VisualPromptingWrapper = None


def setup_libero_config(dataset_root: str, libero_config_path: str):
    """Generates the config.yaml file required by LIBERO."""
    cfg_dir = Path(os.path.expanduser(libero_config_path))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.yaml"

    benchmark_root = Path(libero_pkg.__file__).resolve().parent
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(Path(os.path.expanduser(dataset_root)).resolve()),
        "assets": str(benchmark_root / "assets"),
    }

    with cfg_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[setup] Wrote LIBERO config to: {cfg_path}")


def get_initial_observation(benchmark, task_name):
    """Creates a LIBERO environment, resets it, and returns the initial observation."""
    try:
        from libero.libero.envs import OffScreenRenderEnv
        import libero.libero as libero_pkg

        task_id = benchmark.get_task_names().index(task_name)
        task = benchmark.get_task(task_id)
        benchmark_root = os.path.dirname(libero_pkg.__file__)
        try:
            bddl_file_path = os.path.join(benchmark_root, "bddl_files", task.problem_folder, task.bddl_file)
        except AttributeError as attr_err:
            print(f"[DEBUG] Task object attributes: {dir(task)}")
            print(f"[DEBUG] Task object dict: {getattr(task, '__dict__', 'No __dict__')}\n[DEBUG] Error: {attr_err}")
            raise

        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        init_states = benchmark.get_task_init_states(task_id)
        if init_states is not None and len(init_states) > 0:
            random_state = random.choice(init_states)
            env.set_init_state(random_state)
            if hasattr(env, '_get_observations'):
                obs = env._get_observations()
        env.close()

        # Force garbage collection to clean up the EGL context immediately
        # This prevents EGL_NOT_INITIALIZED errors during Python's final teardown
        del env
        gc.collect()

        adapted_obs = {
            # MuJoCo natively renders upside down (OpenGL convention). Flip it upright.
            "image": obs["agentview_image"][::-1, :, :].copy(),
            "instruction": task.language,
        }
        return adapted_obs
    except Exception as e:
        print(f"Error creating env or getting observation for '{task_name}': {e}")
        return None


def main(num_samples=10, out_dir=os.path.expanduser("~/GIT/efficient_vla/groundingdino_text_hint_samples")):
    if VisualPromptingWrapper is None or libero_pkg is None:
        print("Aborting: `libero` or `visual_prompt_wrapper` is not available.")
        return

    # Setup for headless rendering and LIBERO config
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
    os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
    dataset_root = os.path.expanduser(os.environ.get("DATASET_ROOT", "~/libero_dataset"))
    libero_config_path = os.path.expanduser(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero"))
    setup_libero_config(dataset_root, libero_config_path)

    # Initialize the visual prompter with GroundingDINO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompter = VisualPromptingWrapper(use_image_box=True, use_text_hint=True, device=device)
    if prompter.model is None:
        print("Aborting: GroundingDINO model failed to load. Check paths in visual_prompt_wrapper.py.")
        return

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    benchmark = get_benchmark_dict()["libero_10"]()
    tasks = benchmark.get_task_names()

    annotations = []

    # Ensure we process each task exactly once (up to num_samples)
    tasks_to_process = random.sample(tasks, min(num_samples, len(tasks)))

    for i, task_name in enumerate(tasks_to_process):
        print(f"[{i+1}/{len(tasks_to_process)}] Processing task: {task_name}")

        initial_obs = get_initial_observation(benchmark, task_name)
        if initial_obs is None:
            continue

        original_instruction = initial_obs["instruction"]

        # Generate prompts (text hint only, no box on image) by calling the wrapper
        prompted_obs = prompter.apply_prompts(initial_obs.copy())

        img_np = prompted_obs['image']
        img_pil = Image.fromarray(img_np)
        generated_instruction = prompted_obs['instruction']

        sanitized_task_name = task_name.replace("/", "_")
        
        if 'debug_image' in prompted_obs:
            debug_img_np = prompted_obs['debug_image']
            debug_img_pil = Image.fromarray(debug_img_np)
            debug_filename = f"sample_{i}_{sanitized_task_name}_debug.png"
            debug_out_path = os.path.join(out_dir, debug_filename)
            debug_img_pil.save(debug_out_path)
            print(f"  Saved debug to {debug_out_path}")

        filename = f"sample_{i}_{sanitized_task_name}.png"
        out_path = os.path.join(out_dir, filename)
        
        img_pil.save(out_path)
        print(f"  Saved to {out_path}")

        annotations.append({
            "image_name": filename,
            "original_instruction": original_instruction,
            "groundingdino_instruction": generated_instruction
        })

    json_path = os.path.join(out_dir, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved annotations to {json_path}")

    # Final cleanup before interpreter shutdown
    gc.collect()

if __name__ == "__main__":
    main(num_samples=10)
