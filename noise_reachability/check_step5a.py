#!/usr/bin/env python3
"""Step 5a checks on the real SmolVLA, one observation of task 0 (README.md, step 5a).

1. A fixed noise equal to the one the base draws for a seed gives the base's chunk bit for bit.
2. A fixed noise is repeatable: the same chunk from different torch seeds. A different noise changes the chunk.
3. The same noise reaches every inference: two inferences on one observation give the same chunk.
4. With a fixed noise the environment seed changes nothing (the chunk is the same after a reset with another seed),
   and jitter does change the chunk: why hits are confirmed by jitter and not by reruns.

Exit code 1 if a check fails. Run from the repo root with the repo root on PYTHONPATH (FAIL-003).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    import torch
    from lerobot.utils.constants import ACTION

    from deep_noise_search import install_fixed_noise, make_noise, uninstall_fixed_noise
    from vla_core.envs import libero_env_config, make_libero_env, reset_to_init_state
    from vla_core.evaluation import seed_everything
    from vla_core.policy import DEFAULT_RENAME_MAP, load_base_policy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    args = p.parse_args(argv)
    if "robosuite_mujoco_patch" not in sys.modules:
        raise SystemExit("robosuite_mujoco_patch is not loaded: run with the repo root on PYTHONPATH (FAIL-003)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=520, hard_reset=True)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    policy, model = base.policy, base.policy.model
    env = make_libero_env(env_cfg)
    failures = []

    def check(name, ok, detail=""):
        print(f"[check] {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
        if not ok:
            failures.append(name)

    try:
        task = env.task_description
        obs, _ = reset_to_init_state(env, 0, seed=0)

        def chunk(seed):
            seed_everything(seed)
            base.reset()
            base.act(obs, task)
            return torch.stack(list(policy._queues[ACTION]))

        def with_noise(noise, seed):
            uninstall_fixed_noise(policy)
            install_fixed_noise(policy, noise)
            try:
                return chunk(seed)
            finally:
                uninstall_fixed_noise(policy)

        cfg = policy.config
        shape = (1, cfg.chunk_size, cfg.max_action_dim)
        base_chunk = chunk(11)
        seed_everything(11)
        drawn = model.sample_noise(shape, torch.device(device))
        same = with_noise(drawn, 99)
        check("the base's own noise, fixed, gives its chunk", torch.equal(same, base_chunk),
              f"max |diff| {float((same - base_chunk).abs().max()):.3g}")
        for family in ("full", "repeat"):
            z = make_noise(family, 0, cfg.chunk_size, cfg.max_action_dim)
            a, b = with_noise(z, 1), with_noise(z, 2)
            check(f"{family}: repeats across torch seeds", torch.equal(a, b))
            c = with_noise(make_noise(family, 1, cfg.chunk_size, cfg.max_action_dim), 1)
            diff = float((a - c).abs().max())
            check(f"{family}: another noise changes the chunk", diff > 1e-3, f"max |diff| {diff:.3g}")
        z = make_noise("full", 0, cfg.chunk_size, cfg.max_action_dim)
        install_fixed_noise(policy, z)
        try:
            first = chunk(1)
            policy._queues[ACTION].clear()
            base.act(obs, task)
            second = torch.stack(list(policy._queues[ACTION]))
        finally:
            uninstall_fixed_noise(policy)
        check("the same noise reaches a second inference", torch.equal(first, second))
        obs, _ = reset_to_init_state(env, 0, seed=5000)
        other = with_noise(z, 1)
        check("the environment seed does not change a fixed-noise chunk", torch.equal(other, first))
        jittered = with_noise(make_noise("full", 0, cfg.chunk_size, cfg.max_action_dim, 5000), 1)
        diff = float((jittered - first).abs().max())
        check("jitter changes the chunk", diff > 1e-4, f"max |diff| {diff:.3g}")
    finally:
        env.close()
    print("[check] all passed" if not failures else f"[check] FAILED: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
