#!/usr/bin/env python3
"""Step 5b checks on the real SmolVLA, one observation of task 0 (README.md, step 5b).

1. With ``None`` chosen, the repeated-noise wrapper gives the base's chunk bit for bit.
2. A chosen vector gives the chunk of the same noise fixed by step 5a's wrapper, and is repeatable.
3. The untrained actor's mean gives the chunk of the zero noise, and two seeds sample different chunks.
4. The features the actor sees are finite and of the expected size.

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
    from noise_steering import install_repeated, uninstall
    from train_noise_awr import feature_dim, fresh_learner
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
    policy = base.policy
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

        def steered(choose, seed):
            install_repeated(policy, choose)
            try:
                return chunk(seed)
            finally:
                uninstall(policy)

        plain = chunk(11)
        off = steered(lambda f: None, 11)
        check("None chosen gives the base's chunk", torch.equal(off, plain),
              f"max |diff| {float((off - plain).abs().max()):.3g}")
        cfg = policy.config
        z = make_noise("repeat", 0, cfg.chunk_size, cfg.max_action_dim)
        install_fixed_noise(policy, z)
        try:
            fixed = chunk(1)
        finally:
            uninstall_fixed_noise(policy)
        a, b = steered(lambda f: z[0, 0], 1), steered(lambda f: z[0, 0], 2)
        check("a chosen vector gives step 5a's repeated noise", torch.equal(a, fixed),
              f"max |diff| {float((a - fixed).abs().max()):.3g}")
        check("a chosen vector repeats across torch seeds", torch.equal(a, b))
        seen = []
        size = feature_dim(policy)
        learner = fresh_learner(size, 0, device)
        learner.actor.eval()

        def choose(feats, deterministic=False):
            seen.append(feats)
            return learner.actor.sample(feats.float(), deterministic)[0]

        zero_a = steered(lambda f: choose(f, True), 1)
        zero_b = steered(lambda f: choose(f, True), 2)
        sampled_a = steered(choose, 1)
        sampled_b = steered(choose, 2)
        check("the untrained mean repeats", torch.equal(zero_a, zero_b))
        zeros = steered(lambda f: torch.zeros(cfg.max_action_dim), 1)
        check("the untrained mean is the zero noise", torch.equal(zero_a, zeros))
        check("sampling depends on the seed", float((sampled_a - sampled_b).abs().max()) > 1e-3)
        check("the features have the expected size and are finite",
              all(f.shape == (1, size) and bool(torch.isfinite(f.float()).all()) for f in seen), f"size {size}")
    finally:
        env.close()
    print("[check] all passed" if not failures else f"[check] FAILED: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
