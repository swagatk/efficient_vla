#!/usr/bin/env python3
"""Step 5a: how deep is a systematic init state? (README.md, step 5a; also runs the throughput pilot.)

One work unit is one noise family on one init state of one task. Families:

- ``sampled``: the base policy, a fresh noise at every inference (step 1's sampling, more seeds);
- ``full``: one fixed noise of shape (chunk_size, max_action_dim), used at every inference of the episode;
- ``repeat``: one fixed vector of size max_action_dim, repeated along the chunk (the form of arXiv:2610.13943),
  used at every inference of the episode.

``noise_id`` k gives episode seed 1000 + k (``seed`` column of the results) in every family, so the families are
comparable, and fixed noises come from a generator of their own, the same on every machine. A fixed noise makes an
episode deterministic: the environment seed changes nothing after the init state is set (``check_step5a.py``). So
a hit is confirmed by perturbing the noise: with ``--jitter_seeds`` one fixed noise (``--noise_ids`` names one) runs
once per jitter seed with Gaussian jitter of std JITTER_STD added in the family's form; ``seed`` is then the jitter
seed.

Results go to ``<output_dir>/per_episode_<family>_state<i>.csv``, or ``confirm_<family>_n<id>_state<i>.csv``; a
rerun skips finished episodes. At most --max_episodes_per_process run per call (exit code 3: call again).
Run from the repo root with the repo root on PYTHONPATH (FAIL-003).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXIT_MORE_EPISODES = 3
FAMILIES = ("sampled", "full", "repeat")
EPISODE_SEED_BASE = 1000
NOISE_SEED_BASE = {"full": 10_000_000, "repeat": 20_000_000}
JITTER_SEED_BASE = 30_000_000
JITTER_STD = 0.3


def _draw(family: str, seed: int, chunk_size: int, max_action_dim: int):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    if family == "full":
        return torch.randn(1, chunk_size, max_action_dim, generator=g, dtype=torch.float32)
    return torch.randn(1, 1, max_action_dim, generator=g, dtype=torch.float32).expand(1, chunk_size, -1).clone()


def make_noise(family: str, noise_id: int, chunk_size: int, max_action_dim: int, jitter_seed: int | None = None):
    """The fixed noise of ``family`` number ``noise_id``, shape (1, chunk_size, max_action_dim), float32 on CPU,
    with jitter of JITTER_STD in the family's own form when ``jitter_seed`` is given."""
    if family not in NOISE_SEED_BASE:
        raise ValueError(f"{family!r} has no fixed noise")
    z = _draw(family, NOISE_SEED_BASE[family] + noise_id, chunk_size, max_action_dim)
    if jitter_seed is not None:
        z = z + JITTER_STD * _draw(family, JITTER_SEED_BASE + jitter_seed, chunk_size, max_action_dim)
    return z


def install_fixed_noise(policy, noise) -> None:
    """Make every inference of ``policy`` start from ``noise``, by wrapping that instance's ``sample_actions``."""
    model = policy.model
    if "sample_actions" in vars(model):
        raise RuntimeError("a sampler wrapper is already installed on this policy")
    shape = (1, model.config.chunk_size, model.config.max_action_dim)
    if tuple(noise.shape) != shape:
        raise ValueError(f"noise has shape {tuple(noise.shape)}, expected {shape}")
    original = model.sample_actions  # the bound class method

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if noise is not None:
            raise ValueError("the fixed noise wrapper supplies the noise")
        if state.shape[0] != 1:
            raise ValueError(f"fixed noise expects one observation per call, got batch {state.shape[0]}")
        return original(images, img_masks, lang_tokens, lang_masks, state, noise=fixed.to(state.device), **kwargs)

    fixed = noise
    model.sample_actions = sample_actions


def uninstall_fixed_noise(policy) -> None:
    vars(policy.model).pop("sample_actions", None)


def plan_jobs(family, noise_ids, jitter_seeds, done) -> list[tuple[int, int]]:
    """(noise_id, episode seed) pairs still to run; ``done`` holds finished episode seeds."""
    if jitter_seeds:
        if len(noise_ids) != 1 or family not in NOISE_SEED_BASE:
            raise ValueError("--jitter_seeds perturbs one fixed noise")
        jobs = [(noise_ids[0], s) for s in jitter_seeds]
    else:
        jobs = [(k, EPISODE_SEED_BASE + k) for k in noise_ids]
    return [(k, s) for k, s in jobs if s not in done]


def result_file(out: Path, family: str, state: int, noise_ids, jitter_seeds) -> Path:
    if jitter_seeds:
        return out / f"confirm_{family}_n{noise_ids[0]}_state{state}.csv"
    return out / f"per_episode_{family}_state{state}.csv"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from vla_core.policy import DEFAULT_RENAME_MAP

    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--init_state", type=int, required=True)
    p.add_argument("--noise_ids", required=True, help="e.g. 0-69")
    p.add_argument("--jitter_seeds", default=None, help="perturb one fixed noise with these jitter seeds")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520)
    p.add_argument("--max_episodes_per_process", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from vla_core.envs import libero_env_config, make_libero_env, parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode
    from vla_core.policy import load_base_policy
    from vla_core.provenance import append_jsonl, collect, git_state

    noise_ids = parse_index_ranges(args.noise_ids)
    jitter_seeds = parse_index_ranges(args.jitter_seeds) if args.jitter_seeds else None
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = result_file(out, args.family, args.init_state, noise_ids, jitter_seeds)
    state = args.init_state
    done = {s for s, _ in load_episode_rows(path)}
    jobs = plan_jobs(args.family, noise_ids, jitter_seeds, done)
    print(f"[search] {path.name}: {len(jobs)} episodes to run", flush=True)
    if not jobs:
        return 0
    if "robosuite_mujoco_patch" not in sys.modules:
        raise SystemExit("robosuite_mujoco_patch is not loaded: run with the repo root on PYTHONPATH (FAIL-003)")
    record = collect(str(REPO_ROOT), vars(args))
    record["git_dirty_noise_reachability"] = git_state(str(REPO_ROOT), ("noise_reachability",))["git_dirty"]
    append_jsonl(str(out / "provenance.jsonl"), record)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=args.max_steps, hard_reset=True)
    env = make_libero_env(env_cfg)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    cfg = base.policy.config
    try:
        for k, s in jobs[: args.max_episodes_per_process]:
            uninstall_fixed_noise(base.policy)
            if args.family != "sampled":
                install_fixed_noise(base.policy, make_noise(args.family, k, cfg.chunk_size, cfg.max_action_dim,
                                                            s if jitter_seeds else None))
            base.reset()
            t0 = time.time()
            r = run_episode(env, state, s, lambda obs: base.act(obs, env.task_description)[0], args.max_steps)
            append_episode_row(path, {"task_id": args.task_id, "seed": s, "init_state": state,
                                      "success": int(r["success"]), "steps": r["steps"]})
            print(f"[search] {args.family} noise {k} seed {s} init {state}: success={int(r['success'])} "
                  f"steps={r['steps']} time={time.time() - t0:.1f}s", flush=True)
    finally:
        uninstall_fixed_noise(base.policy)
        env.close()
    return EXIT_MORE_EPISODES if len(jobs) > args.max_episodes_per_process else 0


if __name__ == "__main__":
    sys.exit(main())
