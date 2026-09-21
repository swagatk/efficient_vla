#!/usr/bin/env python3
import faulthandler
faulthandler.enable()

import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import OpenGL
    from OpenGL import EGL
    OpenGL.ERROR_CHECKING = False
except ImportError:
    pass

import sys
from pathlib import Path

# Add project root and local LIBERO
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/home/swagat/lerobot/src")
if os.path.isdir(os.path.expanduser("~/LIBERO")):
    sys.path.insert(0, os.path.expanduser("~/LIBERO"))

print("=" * 60, flush=True)
print("Hybrid Diffusion Pre-flight Import & Dependency Check", flush=True)
print("=" * 60, flush=True)

# 1. PyTorch & CUDA
import torch
print(f"✅ PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})", flush=True)
if torch.cuda.is_available():
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}", flush=True)

# 2. PyAV Patch
try:
    import av
    import types
    if not hasattr(av, "option"):
        av.option = types.SimpleNamespace(Option=object)
    print(f"✅ PyAV version: {av.__version__} (with compatibility patch)", flush=True)
except Exception as e:
    print(f"⚠️ PyAV warning: {e}", flush=True)

# 3. Transformers
import transformers
print(f"✅ Transformers version: {transformers.__version__}", flush=True)

# 4. LeRobot & SmolVLA granular imports
print("Checking LeRobot modules step-by-step:", flush=True)
try:
    print("  • importing lerobot...", flush=True)
    import lerobot
    print("  • importing lerobot.policies.pretrained...", flush=True)
    from lerobot.policies.pretrained import PreTrainedPolicy
    print("  • importing lerobot.policies.smolvla.configuration_smolvla...", flush=True)
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    print("  • importing lerobot.policies.smolvla.smolvlm_with_expert...", flush=True)
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
    print("  • importing lerobot.policies.smolvla.modeling_smolvla...", flush=True)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("  • importing lerobot.policies.factory...", flush=True)
    from lerobot.policies.factory import make_pre_post_processors
    print("✅ LeRobot & SmolVLAPolicy imported successfully", flush=True)
except Exception as e:
    print(f"❌ LeRobot error: {e}", flush=True)

# 5. LIBERO & Robosuite
try:
    import libero.libero as libero_pkg
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    bench = get_benchmark_dict()["libero_10"]()
    print(f"✅ LIBERO benchmark imported successfully ({len(bench.tasks)} tasks in libero_10)", flush=True)
except Exception as e:
    print(f"❌ LIBERO error: {e}", flush=True)

# 6. Hybrid Diffusion Agent & Training Script
try:
    from hybrid_diffusion_agent import HybridFrozenBrainDiffusionHands, ConditionalDDPMHead
    print("✅ HybridFrozenBrainDiffusionHands & ConditionalDDPMHead imported successfully", flush=True)
    import train_hybrid_diffusion
    print("✅ train_hybrid_diffusion imported successfully without errors", flush=True)
except Exception as e:
    print(f"❌ HybridDiffusion error: {e}", flush=True)

# 7. Dataset Check
dataset_paths = [
    Path("/home/swagat/lerobot_datasets/libero_10"),
    Path("/home/swagat/libero_dataset/libero_10"),
]
found_dataset = False
for dp in dataset_paths:
    if dp.is_dir():
        h5_files = list(dp.rglob("*.h5")) + list(dp.rglob("*.hdf5"))
        if h5_files:
            print(f"✅ Local LIBERO HDF5 demonstration dataset found: {dp} ({len(h5_files)} files)", flush=True)
            found_dataset = True
            break
if not found_dataset:
    print("ℹ️ Local HDF5 folder not detected at default paths; will fallback to HuggingFace 'lerobot/libero_10'.", flush=True)

# 8. OffScreenRenderEnv & MuJoCo binding check
try:
    print("Testing OffScreenRenderEnv creation & step...", flush=True)
    bench_root = os.path.dirname(libero_pkg.__file__)
    test_env = train_hybrid_diffusion.create_libero_env(0, bench, bench_root)
    obs = test_env.reset()
    init_states = bench.get_task_init_states(0)
    if init_states is not None and len(init_states) > 0:
        obs = test_env.set_init_state(init_states[0])
    import numpy as np
    obs, reward, done, info = test_env.step(np.zeros(7, dtype=np.float32))
    test_env.close()
    print("✅ OffScreenRenderEnv created, reset, and stepped successfully!", flush=True)
except Exception as e:
    print(f"❌ OffScreenRenderEnv error: {e}", flush=True)

print("=" * 60, flush=True)
print("All checks completed.", flush=True)
print("=" * 60, flush=True)
