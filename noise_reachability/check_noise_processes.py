#!/usr/bin/env python3
"""M1's noise processes on the real SmolVLA, one observation (noise_processes.py; the check of check_step5a.py for the
new families).

1. ar_repeat with rho = 1 gives `repeat`'s chunk at each of three inferences, bit for bit (the switched-off case).
2. ar_repeat with rho = 0.5 gives `repeat`'s chunk at the first inference and another chunk at the second.
3. scaled and zero repeat across torch seeds (no global random numbers), and zero gives the same chunk twice.

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
    from noise_processes import NoiseProcess, install_noise_process
    from vla_core.envs import libero_env_config, make_libero_env, reset_to_init_state
    from vla_core.evaluation import seed_everything
    from vla_core.policy import DEFAULT_RENAME_MAP, load_base_policy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--noise_id", type=int, default=0)
    args = p.parse_args(argv)
    if "robosuite_mujoco_patch" not in sys.modules:
        raise SystemExit("robosuite_mujoco_patch is not loaded: run with the repo root on PYTHONPATH (FAIL-003)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=520, hard_reset=True)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    policy = base.policy
    cfg = policy.config
    env = make_libero_env(env_cfg)
    failures = []

    def check(name, ok, detail=""):
        print(f"[check] {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
        if not ok:
            failures.append(name)

    try:
        task = env.task_description
        obs, _ = reset_to_init_state(env, 0, seed=0)

        def chunks(install, n, seed):
            """The chunks of n inferences on the same observation, with the noise installed once."""
            uninstall_fixed_noise(policy)
            install()
            try:
                seed_everything(seed)
                base.reset()
                out = []
                for _ in range(n):
                    policy._queues[ACTION].clear()
                    base.act(obs, task)
                    out.append(torch.stack(list(policy._queues[ACTION])))
                return out
            finally:
                uninstall_fixed_noise(policy)

        k, c, d = args.noise_id, cfg.chunk_size, cfg.max_action_dim
        rep = chunks(lambda: install_fixed_noise(policy, make_noise("repeat", k, c, d)), 3, 1)
        ar1 = chunks(lambda: install_noise_process(policy, NoiseProcess("ar_repeat", k, c, d, rho=1.0)), 3, 2)
        diffs = [float((a - b).abs().max()) for a, b in zip(rep, ar1)]
        check("ar_repeat rho 1 = repeat at every inference, bit for bit", all(torch.equal(a, b) for a, b in zip(rep, ar1)),
              f"max |diff| per inference {diffs}")
        ar5 = chunks(lambda: install_noise_process(policy, NoiseProcess("ar_repeat", k, c, d, rho=0.5)), 2, 3)
        check("ar_repeat rho 0.5 = repeat at the first inference", torch.equal(ar5[0], rep[0]))
        diff = float((ar5[1] - rep[1]).abs().max())
        check("ar_repeat rho 0.5 changes the second inference", diff > 1e-3, f"max |diff| {diff:.3g}")
        for fam, kw in (("scaled", {"sigma": 0.5}), ("zero", {})):
            a = chunks(lambda: install_noise_process(policy, NoiseProcess(fam, k, c, d, **kw)), 2, 4)
            b = chunks(lambda: install_noise_process(policy, NoiseProcess(fam, k, c, d, **kw)), 2, 5)
            check(f"{fam}: repeats across torch seeds", all(torch.equal(x, y) for x, y in zip(a, b)))
            if fam == "zero":
                check("zero: the same chunk at two inferences", torch.equal(a[0], a[1]))
            else:
                diff = float((a[0] - a[1]).abs().max())
                check("scaled: a fresh noise at the second inference", diff > 1e-3, f"max |diff| {diff:.3g}")
    finally:
        env.close()
    print("[check] all passed" if not failures else f"[check] FAILED: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
