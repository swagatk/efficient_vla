#!/usr/bin/env python3
"""Step 2: evaluate the base policy and noise-selection policies on one LIBERO-10 task, paired (README.md).

Policies are named ``base`` (the frozen SmolVLA, one noise sample per inference) or ``<score>_n<N>``
(``noise_selection.py``: N samples per inference, ``score`` in first / accel / medoid). Every policy runs on
--init_states x --seeds with the same environment and torch seed per (seed, init state)
(``vla_core.evaluation.run_episode``), hard resets, and results are appended to
``<output_dir>/per_episode_<policy>.csv``; a rerun skips finished episodes. At most
--max_episodes_per_process episodes run per call (exit code 3 means "call again"); see run_step2_noise_selection.sh.

Run from the repo root with the repo root on PYTHONPATH (loads robosuite_mujoco_patch via sitecustomize,
FAIL-003).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from vla_core.policy import DEFAULT_RENAME_MAP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_selection import install, parse_policy_label, uninstall  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXIT_MORE_EPISODES = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--policies", default="base,medoid_n8,accel_n8", help="comma-separated policy labels")
    p.add_argument("--seeds", default="0,1,2", help="evaluation seeds, e.g. 0-2")
    p.add_argument("--init_states", default="0-9", help="benchmark init states, e.g. 10-19")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10, help="the step-0 baseline setting")
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520)
    p.add_argument("--max_episodes_per_process", type=int, default=10)
    return p.parse_args(argv)


def require_mujoco_patch():
    if "robosuite_mujoco_patch" not in sys.modules:
        raise SystemExit("robosuite_mujoco_patch is not loaded: run with the repo root on PYTHONPATH "
                         "(see run_step2_noise_selection.sh; FAIL-003)")


def evaluate(args) -> int:
    import torch

    from vla_core.envs import libero_env_config, make_libero_env, parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode
    from vla_core.policy import load_base_policy
    from vla_core.provenance import append_jsonl, collect, git_state

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    specs = {p: parse_policy_label(p) for p in policies}
    seeds, states = parse_index_ranges(args.seeds), parse_index_ranges(args.init_states)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = {p: load_episode_rows(out / f"per_episode_{p}.csv") for p in policies}
    jobs = [(p, s, i) for s in seeds for i in states for p in policies if (s, i) not in done[p]]
    print(f"[eval] task {args.task_id}: {len(jobs)} episodes to run in {out}", flush=True)

    if jobs:
        require_mujoco_patch()
        record = collect(str(REPO_ROOT), vars(args))
        record["git_dirty_noise_reachability"] = git_state(str(REPO_ROOT), ("noise_reachability",))["git_dirty"]
        append_jsonl(str(out / "provenance.jsonl"), record)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env_cfg = libero_env_config(args.task_id, episode_length=args.max_steps, hard_reset=True)
        env = make_libero_env(env_cfg)
        base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                                rename_map=args.rename_map)
        try:
            for p, s, i in jobs[: args.max_episodes_per_process]:
                uninstall(base.policy)
                sel = install(base.policy, *specs[p]) if specs[p] else None
                base.reset()
                t0 = time.time()
                r = run_episode(env, i, s, lambda obs: base.act(obs, env.task_description)[0], args.max_steps)
                append_episode_row(out / f"per_episode_{p}.csv", {"task_id": args.task_id, "seed": s, "init_state": i,
                                                                 "success": int(r["success"]), "steps": r["steps"]})
                extra = ""
                if sel is not None:
                    counts = [sel.chosen.count(k) for k in range(sel.n_samples)]
                    extra = f" inferences={len(sel.chosen)} chosen_per_sample={counts}"
                print(f"[eval] {p} task {args.task_id} seed {s} init {i}: success={int(r['success'])} "
                      f"steps={r['steps']} time={time.time() - t0:.0f}s{extra}", flush=True)
        finally:
            uninstall(base.policy)
            env.close()
        if len(jobs) > args.max_episodes_per_process:
            return EXIT_MORE_EPISODES

    done = {p: load_episode_rows(out / f"per_episode_{p}.csv") for p in policies}
    for p in policies:
        rows = [done[p][(s, i)] for s in seeds for i in states]
        n = sum(int(r["success"]) for r in rows)
        print(f"[eval] task {args.task_id} {p}: {n}/{len(rows)} = {100 * n / len(rows):.1f}%")
    return 0


def main(argv=None) -> int:
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
