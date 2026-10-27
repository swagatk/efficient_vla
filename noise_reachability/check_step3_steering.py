#!/usr/bin/env python3
"""Step 3 checks on the real SmolVLA before training (README.md, step 3, "Checks before running").

On one observation of --task_id (init state 0, one environment reset):

1. steering with a callback that returns ``None`` gives the base policy's chunk bit for bit;
2. a callback that returns the drawn noise unchanged gives the same chunk (the replacement path is exact);
3. a fixed chosen noise gives the same chunk twice, and a different one from the base;
4. the features have the size ``train_noise_sac.feature_dim`` expects, are finite, and repeat exactly;
5. the time per inference with and without steering.

Exit code 1 if a check fails. Run from the repo root with the repo root on PYTHONPATH (FAIL-003).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    import json

    import torch
    from lerobot.utils.constants import ACTION

    from noise_steering import executed_shape, install, uninstall
    from train_noise_sac import feature_dim, load_base, require_mujoco_patch
    from vla_core.envs import reset_to_init_state
    from vla_core.evaluation import seed_everything
    from vla_core.policy import DEFAULT_RENAME_MAP

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_id", type=int, default=7)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520)
    p.add_argument("--timing_inferences", type=int, default=20)
    args = p.parse_args(argv)
    require_mujoco_patch()

    env, base, device = load_base(args, hard_reset=True)
    task = env.task_description
    obs, _ = reset_to_init_state(env, 0, seed=0)
    policy = base.policy
    shape = executed_shape(policy)
    failures = []

    def check(name, ok, detail=""):
        print(f"[check] {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
        if not ok:
            failures.append(name)

    def chunk(seed):
        """The executed chunk of one inference (normalised actions, before the postprocessor)."""
        seed_everything(seed)
        base.reset()
        base.act(obs, task)
        return torch.stack(list(policy._queues[ACTION]))  # the n_action_steps - 1 actions left in the queue

    feats_seen = []

    def run_with(callback, seed=11):
        uninstall(policy)
        install(policy, callback)
        try:
            return chunk(seed)
        finally:
            uninstall(policy)

    try:
        base_chunk = chunk(11)
        none_chunk = run_with(lambda f, d: feats_seen.append(f.clone()))
        check("callback None reproduces the base chunk", torch.equal(none_chunk, base_chunk),
              f"max |diff| {float((none_chunk - base_chunk).abs().max()):.3g}")
        same_chunk = run_with(lambda f, d: d.clone())
        check("returning the drawn noise reproduces the base chunk", torch.equal(same_chunk, base_chunk))
        z = torch.linspace(-2.0, 2.0, shape[0] * shape[1]).reshape(shape)
        a1, a2 = run_with(lambda f, d: z), run_with(lambda f, d: z)
        check("a fixed chosen noise repeats exactly", torch.equal(a1, a2))
        diff = float((a1 - base_chunk).abs().max())
        check("a chosen noise changes the chunk", diff > 1e-3, f"max |diff| to base {diff:.3g}")
        run_with(lambda f, d: feats_seen.append(f.clone()))
        f0, f1 = feats_seen[0], feats_seen[-1]
        check("feature size", tuple(f0.shape) == (1, feature_dim(policy)), f"{tuple(f0.shape)}, expected (1, {feature_dim(policy)})")
        check("features finite and repeatable", bool(torch.isfinite(f0).all()) and torch.equal(f0, f1),
              f"norm {float(f0.norm()):.3g}, state part {f0[0, -8:].tolist()}")

        def timed(callback):
            uninstall(policy)
            if callback is not None:
                install(policy, callback)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            for k in range(args.timing_inferences):
                chunk(100 + k)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            uninstall(policy)
            return (time.perf_counter() - t0) / args.timing_inferences * 1000

        t_base, t_steer = timed(None), timed(lambda f, d: z)
        print(f"[check] time per inference (with the pre/post-processing of one step): base {t_base:.0f} ms, "
              f"steered {t_steer:.0f} ms", flush=True)
    finally:
        uninstall(policy)
        env.close()
    print("[check] all passed" if not failures else f"[check] FAILED: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
