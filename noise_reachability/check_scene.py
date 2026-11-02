#!/usr/bin/env python3
"""Step 8's scene check (README.md, step 8): does one init state give one scene?

Runs one fixed noise twice from the same init state, with different environment seeds and a hard reset, and compares
the body poses of the two scenes, FAIL-010's measure. A failed episode ends at the same step whatever the scene, so
outcomes could not see a scene that moves. The verdict is the poses alone: the executed actions are recorded (largest
difference and first differing step of SCENE_STEPS) for diagnosis, because one last-bit difference between two rollouts
on a GPU would otherwise read as a moved scene and silently change which states are tested. The result, pass or fail,
goes to ``<output_dir>/scene_state<i>.json``. A state that fails is not an error: exit code 0, and the analysis replaces
it. A rerun skips a finished state. Run from the repo root with the repo root on PYTHONPATH (FAIL-003).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "noise_reachability"))

SCENE_NOISE_ID = 3000           # used by no test
SCENE_SEEDS = (7000, 7001)
SCENE_STEPS = 120               # 12 inferences at n_action_steps=10
POSE_TOLERANCE = 1e-6           # metres, FAIL-010's comparison


def scene_file(out, state) -> Path:
    return Path(out) / f"scene_state{state}.json"


def moved_bodies(a: dict, b: dict, tolerance: float = POSE_TOLERANCE) -> dict:
    """{body: largest coordinate change} for bodies that moved; a body in one scene only counts as moved."""
    moved = {k: float("inf") for k in set(a) ^ set(b)}
    for k in set(a) & set(b):
        change = max(abs(x - y) for x, y in zip(a[k], b[k], strict=True))
        if change > tolerance:
            moved[k] = change
    return moved


def action_difference(a: list, b: list) -> dict:
    """Largest difference and first differing step of two action sequences, each a list of float lists."""
    diffs = [max(abs(x - y) for x, y in zip(p, q, strict=True)) for p, q in zip(a, b, strict=False)]
    return {"max": max(diffs, default=0.0), "first_step": next((i for i, d in enumerate(diffs) if d > 0), None),
            "lengths": [len(a), len(b)]}


def inner_env(env):
    """The robosuite environment under the LeRobot and LIBERO wrappers."""
    e = env
    for _ in range(6):
        if hasattr(e, "sim"):
            return e
        e = getattr(e, "_env", None) or getattr(e, "env", None)
    raise AttributeError("no sim under this environment")


def body_poses(env) -> dict:
    m = inner_env(env).sim.model
    return {m.body_id2name(i): [float(x) for x in m.body_pos[i]] for i in range(m.nbody)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from vla_core.policy import DEFAULT_RENAME_MAP

    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--init_state", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520, help="the environment's horizon; the check stops at SCENE_STEPS")
    args = p.parse_args(argv)
    path = scene_file(args.output_dir, args.init_state)
    if path.exists():
        print(f"[scene] {path.name} exists", flush=True)
        return 0
    import numpy as np
    import torch

    from deep_noise_search import install_fixed_noise, make_noise, uninstall_fixed_noise
    from vla_core.envs import libero_env_config, make_libero_env
    from vla_core.evaluation import run_episode
    from vla_core.policy import load_base_policy
    from vla_core.provenance import append_jsonl, collect, git_state

    if "robosuite_mujoco_patch" not in sys.modules:
        raise SystemExit("robosuite_mujoco_patch is not loaded: run with the repo root on PYTHONPATH (FAIL-003)")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = collect(str(REPO_ROOT), vars(args))
    record["git_dirty_noise_reachability"] = git_state(str(REPO_ROOT), ("noise_reachability",))["git_dirty"]
    append_jsonl(str(path.parent / "provenance.jsonl"), record)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=args.max_steps, hard_reset=True)
    env = make_libero_env(env_cfg)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    cfg = base.policy.config
    runs = []
    try:
        install_fixed_noise(base.policy, make_noise("repeat", SCENE_NOISE_ID, cfg.chunk_size, cfg.max_action_dim))
        for seed in SCENE_SEEDS:
            actions, poses = [], {}

            def choose(obs):
                action = base.act(obs, env.task_description)[0]
                actions.append(np.asarray(action, dtype=float).ravel().tolist())
                return action

            base.reset()
            r = run_episode(env, args.init_state, seed, choose, SCENE_STEPS, on_reset=lambda _: poses.update(body_poses(env)))
            runs.append({"seed": seed, "steps": r["steps"], "success": int(r["success"]), "actions": actions, "poses": poses})
            print(f"[scene] task {args.task_id} state {args.init_state} seed {seed}: {r}", flush=True)
    finally:
        uninstall_fixed_noise(base.policy)
        env.close()
    moved = moved_bodies(runs[0]["poses"], runs[1]["poses"])
    actions = action_difference(runs[0]["actions"], runs[1]["actions"])
    path.write_text(json.dumps({"task_id": args.task_id, "init_state": args.init_state, "noise_id": SCENE_NOISE_ID,
                                "seeds": SCENE_SEEDS, "same": not moved, "moved": moved, "actions": actions,
                                "outcomes": [[r["success"], r["steps"]] for r in runs]}, indent=1) + "\n")
    print(f"[scene] task {args.task_id} state {args.init_state}: {'same' if not moved else 'DIFFERENT'}, moved {moved}, "
          f"actions {actions}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
