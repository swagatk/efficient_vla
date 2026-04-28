#!/usr/bin/env python3
import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from visual_prompt_wrapper import VisualPromptingWrapper


class VisualPromptingEnv:
    """
    Environment wrapper that applies visual prompting to observations before the policy sees them.
    """

    def __init__(self, env, wrapper: VisualPromptingWrapper):
        self.env = env
        self.wrapper = wrapper
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return self._apply_prompts(obs), info
        return self._apply_prompts(out)

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            obs = self._apply_prompts(obs)
            return obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            obs = self._apply_prompts(obs)
            return obs, reward, done, info
        return out

    def _find_image_and_instruction_keys(
        self, obs: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        image_key = None
        instruction_key = None

        preferred_image_keys = [
            "image",
            "rgb",
            "observation.image",
            "observation.images.main",
            "observation.images.front",
        ]
        preferred_instruction_keys = [
            "instruction",
            "task",
            "language_instruction",
            "lang",
            "text",
        ]

        for k in preferred_image_keys:
            if k in obs and isinstance(obs[k], np.ndarray) and obs[k].ndim >= 2:
                image_key = k
                break

        if image_key is None:
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    image_key = k
                    break

        for k in preferred_instruction_keys:
            if k in obs and isinstance(obs[k], str):
                instruction_key = k
                break

        if instruction_key is None:
            for k, v in obs.items():
                if isinstance(v, str):
                    instruction_key = k
                    break

        return image_key, instruction_key

    def _apply_prompts(self, obs: Any) -> Any:
        if not isinstance(obs, dict):
            return obs

        if "image" in obs and "instruction" in obs:
            try:
                return self.wrapper.apply_prompts(obs)
            except Exception as e:
                print(f"[visual-prompt] prompt application failed (native keys): {e}")
                return obs

        image_key, instruction_key = self._find_image_and_instruction_keys(obs)
        if image_key is None or instruction_key is None:
            return obs

        adapted = {
            "image": obs[image_key],
            "instruction": obs[instruction_key],
        }

        try:
            prompted = self.wrapper.apply_prompts(adapted)
            obs[image_key] = prompted["image"]
            obs[instruction_key] = prompted["instruction"]
            return obs
        except Exception as e:
            print(f"[visual-prompt] prompt application failed (adapted keys): {e}")
            return obs


def ensure_libero_config(dataset_root: str, libero_config_path: str) -> Path:
    try:
        import libero.libero as libero_pkg
    except ImportError as e:
        raise RuntimeError(
            "LIBERO is not installed. Install it before running this script."
        ) from e

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

    print(f"[setup] wrote LIBERO config: {cfg_path}")
    for k, v in config.items():
        print(f"[setup]   {k}: {v}")

    return cfg_path


def install_env_patch(prompter: VisualPromptingWrapper):
    patched_modules = []

    def patch_make(module_name: str):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            return

        if not hasattr(mod, "make"):
            return

        original_make = mod.make

        def patched_make(*args, **kwargs):
            env = original_make(*args, **kwargs)
            return VisualPromptingEnv(env, prompter)

        mod.make = patched_make
        patched_modules.append((mod, original_make))
        print(f"[patch] patched {module_name}.make")

    patch_make("gymnasium")
    patch_make("gym")

    if not patched_modules:
        print(
            "[patch] warning: could not patch gymnasium/gym. "
            "Visual prompting may not be injected."
        )

    def restore():
        for mod, original_make in patched_modules:
            mod.make = original_make

    return restore


def run_lerobot_eval_inprocess(eval_argv):
    """
    Run LeRobot evaluator in the same process so gym make patching can take effect.
    """
    candidates = [
        "lerobot.scripts.eval",
        "lerobot.scripts.lerobot_eval",
        "lerobot.eval",
    ]

    last_err = None
    for module_name in candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            last_err = e
            continue

        main_fn = getattr(mod, "main", None)
        if main_fn is None:
            continue

        print(f"[run] using in-process entrypoint: {module_name}.main")
        old_argv = sys.argv[:]
        try:
            sys.argv = ["lerobot-eval"] + eval_argv
            try:
                ret = main_fn()
            except TypeError:
                ret = main_fn(eval_argv)
        finally:
            sys.argv = old_argv

        if isinstance(ret, int):
            return ret
        return 0

    raise RuntimeError(
        "Could not find an in-process LeRobot eval entrypoint. "
        f"Last import error: {last_err}"
    )


def run_lerobot_eval_subprocess(eval_argv):
    cmd = ["lerobot-eval"] + eval_argv
    print("[run] subprocess command:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def build_eval_args(args):
    eval_args = [
        f"--policy.path={args.model_id}",
        f"--policy.device={args.policy_device}",
        "--env.type=libero",
        f"--env.task={args.task_suite}",
        "--env.render_mode=rgb_array",
        "--env.max_parallel_tasks=1",
        f"--env.observation_height={args.observation_height}",
        f"--env.observation_width={args.observation_width}",
        f"--eval.batch_size={args.batch_size}",
        "--eval.use_async_envs=false",
        f"--eval.n_episodes={args.n_episodes}",
        f"--eval.max_episodes_rendered={args.max_episodes_rendered}",
        f"--output_dir={args.output_dir}",
    ]

    if args.episode_horizon is not None:
        eval_args.append(f"--env.episode_length={args.episode_horizon}")

    if args.extra_eval_arg:
        eval_args.extend(args.extra_eval_arg)

    return eval_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA with Visual Prompting (Week 2 Stage 1)"
    )
    parser.add_argument(
        "--model_id",
        default="HuggingFaceVLA/smolvla_libero",
        help="SmolVLA HuggingFace repo id or local path",
    )
    parser.add_argument(
        "--task_suite",
        default="libero_10",
        help="LIBERO task suite name (e.g., libero_10, libero_spatial)",
    )
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        default="./outputs/eval/week2_visual_prompting",
    )
    parser.add_argument(
        "--dataset_root",
        default="~/libero_dataset",
        help="Path used in LIBERO config as datasets root",
    )
    parser.add_argument(
        "--libero_config_path",
        default="~/.libero",
        help="Directory containing LIBERO config.yaml",
    )
    parser.add_argument("--max_episodes_rendered", type=int, default=10)
    parser.add_argument(
        "--episode_horizon",
        type=int,
        default=520,
        help="Max steps per episode (maps to --env.episode_length)",
    )
    parser.add_argument("--observation_height", type=int, default=256)
    parser.add_argument("--observation_width", type=int, default=256)

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--policy_device", default=default_device)
    parser.add_argument(
        "--grounding_device",
        default=default_device,
        help="Device for GroundingDINO in visual_prompt_wrapper",
    )

    parser.add_argument(
        "--box_overlay",
        action="store_true",
        help="Enable box overlay on image observation",
    )
    parser.add_argument(
        "--text_hint",
        action="store_true",
        help="Enable text spatial hint appended to instruction",
    )

    parser.add_argument(
        "--inprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run lerobot evaluator in-process (required for prompt injection). "
             "Use --no-inprocess to force subprocess mode.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and exit without running eval",
    )
    parser.add_argument(
        "--extra_eval_arg",
        action="append",
        default=[],
        help=(
            "Extra raw argument to forward to lerobot-eval. "
            "Can be passed multiple times, e.g. --extra_eval_arg=--eval.seed=0"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["DATASET_ROOT"] = os.path.expanduser(args.dataset_root)
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
    os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = os.path.expanduser(args.libero_config_path)

    ensure_libero_config(args.dataset_root, args.libero_config_path)

    enable_prompting = args.box_overlay or args.text_hint
    if enable_prompting:
        prompter = VisualPromptingWrapper(
            use_image_box=args.box_overlay,
            use_text_hint=args.text_hint,
            device=args.grounding_device,
        )
    else:
        prompter = None
        print("[visual-prompt] both prompt modes disabled; running baseline eval.")

    eval_argv = build_eval_args(args)
    print("[eval] arguments:")
    for x in eval_argv:
        print(f"[eval]   {x}")

    if args.dry_run:
        print("[eval] dry-run enabled, exiting.")
        return 0

    if enable_prompting and args.inprocess:
        restore_patch = install_env_patch(prompter)
        try:
            code = run_lerobot_eval_inprocess(eval_argv)
        finally:
            restore_patch()
        return code

    if enable_prompting and not args.inprocess:
        print(
            "[visual-prompt] warning: subprocess mode usually cannot apply runtime "
            "gym monkey-patches. Prefer --inprocess for actual prompt injection."
        )

    return run_lerobot_eval_subprocess(eval_argv)


if __name__ == "__main__":
    raise SystemExit(main())
