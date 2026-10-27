#!/usr/bin/env python3
"""Step 3: train a noise policy with SAC on one LIBERO-10 task, then evaluate it against the base (README.md).

Training (``--mode train``): episodes from --train_init_states (one drawn at random per episode, reproducible
from --seed and the episode index), soft resets. At every inference the actor (``noise_sac.py``) chooses the
noise of the executed positions and real action dimensions through ``noise_steering.py``; the first
--warmup_episodes execute the base policy's own noise (clipped to the actor's range) and train only the
critics. After each episode the transitions go to the replay buffer and --updates_per_decision gradient steps
run per decision collected. One process runs at most --episodes_per_process episodes, then saves the learner,
the replay buffer and the log to --output_dir and exits with code 3 ("call again"); the next call resumes.

Evaluation (``--mode eval``): ``base`` (the frozen SmolVLA) and ``actor`` (the final checkpoint, deterministic
noise) on --init_states x --seeds with the same environment and torch seed per (seed, init state), hard
resets, results appended to ``<output_dir>/eval/per_episode_<policy>.csv``; at most
--max_episodes_per_process episodes per call (exit code 3 = call again).

Run from the repo root with the repo root on PYTHONPATH (loads robosuite_mujoco_patch via sitecustomize,
FAIL-003): use run_step3_noise_sac.sh.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from vla_core.policy import DEFAULT_RENAME_MAP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_noise_selection import require_mujoco_patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXIT_MORE_EPISODES = 3
# run_episode seeds: training episodes never share one with evaluation seeds 0-99.
TRAIN_SEED_OFFSET = 100_000
LOG_FIELDS = ["episode", "init_state", "success", "steps", "decisions", "warmup", "updates", "critic_loss",
              "q_mean", "actor_loss", "entropy", "alpha", "noise_abs_mean", "time_s"]
# Settings a resumed run and the evaluation must share with the run that wrote the checkpoint.
RUN_KEYS = ("task_id", "seed", "policy_path", "n_action_steps", "rename_map", "max_steps", "train_init_states",
            "num_episodes", "warmup_episodes", "updates_per_decision", "batch_size", "buffer_size", "noise_scale",
            "init_std", "lr", "gamma", "tau", "init_alpha", "hidden", "layers", "train_hard_reset")
# Settings the evaluation must share with the training run (the base policy and its environment).
BASE_KEYS = ("task_id", "policy_path", "n_action_steps", "rename_map", "max_steps")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--output_dir", required=True, help="checkpoint, replay parts, train_log.csv; eval/ for evaluation")
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10, help="the step-0 baseline setting")
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520)
    # Training (README.md, step 3)
    p.add_argument("--train_init_states", default="10-49")
    p.add_argument("--num_episodes", type=int, default=1500)
    p.add_argument("--warmup_episodes", type=int, default=20)
    p.add_argument("--episodes_per_process", type=int, default=25)
    p.add_argument("--updates_per_decision", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=200_000, help="larger than the transitions of a run")
    p.add_argument("--noise_scale", type=float, default=2.5)
    p.add_argument("--init_std", type=float, default=0.45)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--init_alpha", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--train_hard_reset", type=int, default=0, help="1 = rebuild the simulator every episode")
    # Evaluation
    p.add_argument("--policies", default="base,actor", help="comma-separated: base, actor")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--init_states", default="0-9")
    p.add_argument("--max_episodes_per_process", type=int, default=10)
    return p.parse_args(argv)


def train_episode_seed(seed: int, episode: int) -> int:
    """``seed`` argument of run_episode for training episode ``episode``."""
    if not 0 <= episode < 10_000 or seed < 0:
        raise ValueError(f"need seed >= 0 and 0 <= episode < 10000, got {seed}, {episode}")
    return TRAIN_SEED_OFFSET + seed * 10_000 + episode


def train_init_state(seed: int, episode: int, states: list[int]) -> int:
    return int(np.random.default_rng([seed, episode]).choice(states))


def run_config(args) -> dict:
    return {k: getattr(args, k) for k in RUN_KEYS}


def check_same_run(saved: dict, args, keys=RUN_KEYS) -> None:
    for k in keys:
        if saved.get(k) != getattr(args, k):
            raise SystemExit(f"checkpoint was written with {k}={saved.get(k)!r}, this call has {getattr(args, k)!r}")


def read_log(path: Path, before_episode: int) -> list[dict]:
    """Rows of train_log.csv for episodes below ``before_episode`` (a crashed process may have logged more)."""
    if not path.exists():
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if int(r["episode"]) < before_episode]


def write_log(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def save_checkpoint(out: Path, learner, replay, args, start: int, episodes_done: int, parts: list[str],
                    part_start: int) -> list[str]:
    """Write this process's transitions as a new replay part, then the checkpoint that lists every part.

    Parts are never rewritten or deleted, and the checkpoint is replaced only after its new part is on disk
    (``os.sync``), so an interrupted save, or a Drive mount that loses the last uploads, leaves the previous
    checkpoint with all its parts. Returns the new list of parts.
    """
    import torch

    part = f"replay_part_{start:05d}-{episodes_done - 1:05d}.npz"
    replay.save_part(out / part, part_start)
    os.sync()
    parts = [*parts, part]
    tmp = out / "checkpoint.pt.tmp"
    torch.save({"learner": learner.state_dict(), "run_config": run_config(args), "episodes_done": episodes_done,
                "replay_parts": parts, "replay_size": len(replay)}, tmp)
    os.replace(tmp, out / "checkpoint.pt")
    os.sync()
    return parts


def load_base(args, hard_reset):
    import torch

    from vla_core.envs import libero_env_config, make_libero_env
    from vla_core.policy import load_base_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=args.max_steps, hard_reset=hard_reset)
    env = make_libero_env(env_cfg)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    return env, base, device


def feature_dim(policy, state_dim: int = 8) -> int:
    return policy.model.vlm_with_expert.config.text_config.hidden_size + state_dim


def provenance(out: Path, args) -> None:
    from vla_core.provenance import append_jsonl, collect, git_state

    record = collect(str(REPO_ROOT), vars(args))
    record["git_dirty_noise_reachability"] = git_state(str(REPO_ROOT), ("noise_reachability",))["git_dirty"]
    append_jsonl(str(out / "provenance.jsonl"), record)


def train(args) -> int:
    import torch

    from noise_sac import ChunkReplay, EpisodeTransitions, NoiseSAC, NoiseSACConfig
    from noise_steering import executed_shape, install, uninstall
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import run_episode

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_states = parse_index_ranges(args.train_init_states)
    overlap = set(train_states) & set(parse_index_ranges(args.init_states))
    if overlap:
        raise SystemExit(f"training and evaluation init states overlap: {sorted(overlap)}")
    most_transitions = args.num_episodes * (args.max_steps // args.n_action_steps + 1)
    if most_transitions > args.buffer_size:
        raise SystemExit(f"--buffer_size {args.buffer_size} is below the {most_transitions} transitions a run can hold")
    ckpt_path, log_path = out / "checkpoint.pt", out / "train_log.csv"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None
    if ckpt is not None:
        check_same_run(ckpt["run_config"], args)
    start = ckpt["episodes_done"] if ckpt else 0
    if start >= args.num_episodes:
        print(f"[train] task {args.task_id}: all {args.num_episodes} episodes done", flush=True)
        return 0
    stop = min(start + args.episodes_per_process, args.num_episodes)
    require_mujoco_patch()
    provenance(out, args)
    env, base, device = load_base(args, hard_reset=bool(args.train_hard_reset))
    task = env.task_description
    shape = executed_shape(base.policy)
    cfg = NoiseSACConfig(feature_dim=feature_dim(base.policy), action_dim=shape[0] * shape[1],
                         noise_scale=args.noise_scale, init_std=args.init_std, hidden=args.hidden,
                         layers=args.layers, lr=args.lr, gamma=args.gamma, tau=args.tau, init_alpha=args.init_alpha)
    learner = NoiseSAC(cfg, device)
    replay = ChunkReplay(args.buffer_size, cfg.feature_dim, cfg.action_dim, seed=args.seed * 10_000 + start)
    parts = list(ckpt["replay_parts"]) if ckpt else []
    if ckpt:
        learner.load_state_dict(ckpt["learner"])
        missing = [x for x in parts if not (out / x).exists()]
        if missing:
            raise SystemExit(f"replay parts listed in checkpoint.pt are missing: {missing}")
        replay.load_parts([out / x for x in parts])
        if len(replay) != ckpt["replay_size"]:
            raise SystemExit(f"replay has {len(replay)} transitions, the checkpoint expects {ckpt['replay_size']}")
    part_start = len(replay)
    log_rows = read_log(log_path, start)
    if len(log_rows) != start:
        raise SystemExit(f"train_log.csv has {len(log_rows)} rows below episode {start}")
    print(f"[train] task {args.task_id}: episodes {start}-{stop - 1} of {args.num_episodes}, "
          f"replay {len(replay)} transitions", flush=True)

    state = {}  # per-episode: transitions, warm-up flag, bootstrap features

    def choose_noise(feats, drawn):
        f = feats[0].float().cpu().numpy()
        if state["bootstrap"] is not None:  # the final observation of a failed episode: features only
            state["bootstrap"].append(f)
            return None
        if state["warmup"]:
            u = np.clip(drawn.float().cpu().numpy() / args.noise_scale, -0.999, 0.999).astype(np.float32)
        else:
            u = learner.act(f, deterministic=False)
        state["transitions"].decide(f, u.reshape(-1))
        state["noise_abs"].append(float(np.abs(u).mean()) * args.noise_scale)
        return torch.from_numpy(args.noise_scale * u.reshape(shape))

    install(base.policy, choose_noise)
    try:
        for episode in range(start, stop):
            t0 = time.time()
            init_state = train_init_state(args.seed, episode, train_states)
            warmup = episode < args.warmup_episodes
            state.update(transitions=EpisodeTransitions(), warmup=warmup, bootstrap=None, noise_abs=[])
            last = {}
            base.reset()
            r = run_episode(env, init_state, train_episode_seed(args.seed, episode),
                            lambda obs: base.act(obs, task)[0], args.max_steps,
                            on_step=lambda obs, a, rew, nxt, term, info: last.update(obs=nxt))
            boot = None
            if not r["success"]:
                state["bootstrap"] = []
                base.reset()  # empty the action queue so the next call runs an inference
                base.act(last["obs"], task)
                boot = state["bootstrap"][0]
            transitions = state["transitions"].finish(r["success"], boot)
            for t in transitions:
                replay.add(t)
            losses = []
            if len(replay) >= args.batch_size:
                for _ in range(args.updates_per_decision * len(transitions)):
                    losses.append(learner.update(replay.sample(args.batch_size, device), update_actor=not warmup))
            row = {"episode": episode, "init_state": init_state, "success": int(r["success"]), "steps": r["steps"],
                   "decisions": len(transitions), "warmup": int(warmup), "updates": len(losses),
                   "noise_abs_mean": round(float(np.mean(state["noise_abs"])), 4), "time_s": round(time.time() - t0, 1)}
            for k in ("critic_loss", "q_mean", "actor_loss", "entropy", "alpha"):
                vals = [l[k] for l in losses if k in l]
                row[k] = round(float(np.mean(vals)), 6) if vals else ""
            log_rows.append(row)
            write_log(log_path, log_rows)
            recent = [int(x["success"]) for x in log_rows[-50:]]
            print(f"[train] task {args.task_id} ep {episode + 1}/{args.num_episodes} init {init_state}: "
                  f"success={row['success']} steps={row['steps']} decisions={row['decisions']} "
                  f"updates={row['updates']} alpha={row['alpha']} q={row['q_mean']} "
                  f"|noise|={row['noise_abs_mean']} mean50={np.mean(recent):.2f} time={row['time_s']}s"
                  + (" (warm-up)" if warmup else ""), flush=True)
    finally:
        uninstall(base.policy)
        env.close()
    save_checkpoint(out, learner, replay, args, start, stop, parts, part_start)
    print(f"[train] task {args.task_id}: saved after {stop} episodes, replay {len(replay)}", flush=True)
    return 0 if stop >= args.num_episodes else EXIT_MORE_EPISODES


def evaluate(args) -> int:
    import torch

    from noise_sac import NoiseSAC
    from noise_steering import executed_shape, install, uninstall
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies or set(policies) - {"base", "actor"}:
        raise SystemExit(f"--policies must name base and/or actor, got {args.policies!r}")
    seeds, states = parse_index_ranges(args.seeds), parse_index_ranges(args.init_states)
    out = Path(args.output_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    done = {p: load_episode_rows(out / f"per_episode_{p}.csv") for p in policies}
    jobs = [(p, s, i) for s in seeds for i in states for p in policies if (s, i) not in done[p]]
    print(f"[eval] task {args.task_id}: {len(jobs)} episodes to run in {out}", flush=True)

    if jobs:
        learner = None
        if "actor" in policies:
            ckpt = torch.load(Path(args.output_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
            check_same_run(ckpt["run_config"], args, BASE_KEYS)
            if ckpt["episodes_done"] != ckpt["run_config"]["num_episodes"]:
                raise SystemExit(f"training is not finished: {ckpt['episodes_done']} of "
                                 f"{ckpt['run_config']['num_episodes']} episodes")
        require_mujoco_patch()
        provenance(out, args)
        env, base, device = load_base(args, hard_reset=True)
        if "actor" in policies:
            learner = NoiseSAC.from_state_dict(ckpt["learner"], device)
            learner.actor.eval()
        shape = executed_shape(base.policy)

        def actor_noise(feats, drawn):
            u = learner.act(feats[0].float().cpu().numpy(), deterministic=True)
            return torch.from_numpy(learner.cfg.noise_scale * u.reshape(shape))

        try:
            for p, s, i in jobs[: args.max_episodes_per_process]:
                uninstall(base.policy)
                st = install(base.policy, actor_noise) if p == "actor" else None
                base.reset()
                t0 = time.time()
                r = run_episode(env, i, s, lambda obs: base.act(obs, env.task_description)[0], args.max_steps)
                append_episode_row(out / f"per_episode_{p}.csv", {"task_id": args.task_id, "seed": s, "init_state": i,
                                                                 "success": int(r["success"]), "steps": r["steps"]})
                extra = f" inferences={st.n_inferences}" if st else ""
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
    args = parse_args(argv)
    return train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
