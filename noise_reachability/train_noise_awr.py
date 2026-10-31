#!/usr/bin/env python3
"""Step 5b: NSL-AWR on task 0, trained on init states 10-49, evaluated on 0-9 (README.md, step 5b).

``--mode collect``: one part of one epoch, sampled from the actor of the previous epoch; writes
``collect/eNNN/partK.npz`` when the part is finished, so an interrupted part runs again with the same seeds.
``--mode update``: merges an epoch's parts into ``replay/eNNN.npz``, trains on every epoch so far, writes
``actor_eNNN.pt`` and ``train_log.csv``. Parts and replay are never rewritten.
``--mode eval``: arms ``base``, ``prior`` (untrained, sampled), ``zero`` (untrained, mean), ``actor`` (final, sampled),
``actor_mean`` (final, mean) on --init_states x --seeds, appended to ``<eval_dir>/per_episode_<arm>.csv``; exit code 3
means "call again".

Run from the repo root with the repo root on PYTHONPATH (FAIL-003): use run_step5b_noise_awr.sh.
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
from train_noise_sac import check_same_run, feature_dim, load_base, provenance  # noqa: E402

EXIT_MORE_EPISODES = 3
TRAIN_SEED_OFFSET = 200_000      # step 3 used 100,000; neither meets an evaluation seed
EPOCH_FILE = "e{:03d}"
ARMS = ("base", "prior", "zero", "actor", "actor_mean")
LEARNED_ARMS = ("actor", "actor_mean")
MEAN_ARMS = ("zero", "actor_mean")
LOG_FIELDS = ["epoch", "init_state", "seed", "success", "steps", "decisions"]
UPDATE_FIELDS = ["epoch", "replay_decisions", "replay_success_decisions", "value_loss", "actor_loss", "weight_mean",
                 "mean_abs", "std_mean", "time_s"]
RUN_KEYS = ("task_id", "train_seed", "policy_path", "n_action_steps", "rename_map", "max_steps", "train_init_states",
            "parts", "updates_per_epoch")
BASE_KEYS = ("task_id", "policy_path", "n_action_steps", "rename_map", "max_steps")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["collect", "update", "eval"], required=True)
    p.add_argument("--output_dir", required=True, help="this training seed's folder (collect, update, eval)")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520)
    p.add_argument("--train_init_states", default="10-49")
    p.add_argument("--epoch", type=int, default=0, help="collect, update: this epoch; eval: the final epoch")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--updates_per_epoch", type=int, default=100, help="joint updates, and as many value-only updates")
    p.add_argument("--arm", choices=ARMS, help="eval")
    p.add_argument("--seeds", default="16-31", help="eval")
    p.add_argument("--init_states", default="0-9", help="eval")
    p.add_argument("--eval_dir", help="eval: folder of the per-episode csv")
    p.add_argument("--max_episodes_per_process", type=int, default=10)
    return p.parse_args(argv)


def train_episode_seed(train_seed: int, episode: int) -> int:
    """``seed`` argument of run_episode for training episode ``episode``."""
    if not 0 <= episode < 10_000 or train_seed < 0:
        raise ValueError(f"need train_seed >= 0 and 0 <= episode < 10000, got {train_seed}, {episode}")
    return TRAIN_SEED_OFFSET + train_seed * 10_000 + episode


def part_jobs(states: list[int], epoch: int, part: int, parts: int) -> list[tuple[int, int]]:
    """(init state, episode index) of one part of one epoch: every state once per epoch, across the parts."""
    if not 0 <= part < parts:
        raise ValueError(f"part {part} of {parts}")
    return [(states[i], epoch * len(states) + i) for i in range(part, len(states), parts)]


def run_config(args) -> dict:
    return {k: getattr(args, k) for k in RUN_KEYS}


def checkpoint_path(out: Path, epoch: int) -> Path:
    return out / f"actor_{EPOCH_FILE.format(epoch)}.pt"


def write_npz(path: Path, **arrays) -> None:
    tmp = f"{path}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def fresh_learner(feature_size: int, train_seed: int, device):
    import torch

    from noise_awr import NoiseAWR, NoiseAWRConfig

    torch.manual_seed(train_seed)
    return NoiseAWR(NoiseAWRConfig(feature_dim=feature_size), device)


def load_learner(path: Path, device, args=None, keys=RUN_KEYS):
    import torch

    from noise_awr import NoiseAWR

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if args is not None:
        check_same_run(ckpt["run_config"], args, keys)
    return NoiseAWR.from_state_dict(ckpt["learner"], device)


def episode_arrays(feats: list, zs: list, success: bool, gamma: float) -> dict:
    """One episode's decisions as replay arrays; features that do not survive float16 stop the run."""
    from noise_awr import episode_returns

    n = len(zs)
    stored = np.stack(feats).astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError("features are not finite in float16")
    return {"feats": stored, "z": np.stack(zs).astype(np.float32),
            "ret": episode_returns(n, success, gamma), "success": np.full(n, float(success), dtype=np.float32)}


def concat_arrays(episodes: list[dict]) -> dict:
    return {k: np.concatenate([e[k] for e in episodes]) for k in episodes[0]}


def collect(args) -> int:
    import torch

    from noise_steering import install_repeated, uninstall
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import run_episode

    out = Path(args.output_dir)
    folder = out / "collect" / EPOCH_FILE.format(args.epoch)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"part{args.part}.npz"
    if path.exists():
        print(f"[collect] {path} exists", flush=True)
        return 0
    jobs = part_jobs(parse_index_ranges(args.train_init_states), args.epoch, args.part, args.parts)
    prev = checkpoint_path(out, args.epoch - 1)
    if args.epoch > 0 and not prev.exists():
        raise SystemExit(f"epoch {args.epoch} needs {prev}")
    require_mujoco_patch()
    own = folder / f"part{args.part}"   # one provenance file per process: Drive loses concurrent appends
    own.mkdir(exist_ok=True)
    provenance(own, args)
    env, base, device = load_base(args, hard_reset=True)
    task = env.task_description
    learner = load_learner(prev, device, args) if args.epoch else fresh_learner(feature_dim(base.policy),
                                                                               args.train_seed, device)
    learner.actor.eval()
    record = {}

    def choose(feats):
        z = learner.actor.sample(feats.float())[0]
        record["feats"].append(feats[0].float().cpu().numpy())
        record["z"].append(z.cpu().numpy())
        return z

    episodes, rows = [], []
    install_repeated(base.policy, choose)
    try:
        for state, index in jobs:
            record.update(feats=[], z=[])
            base.reset()
            t0 = time.time()
            seed = train_episode_seed(args.train_seed, index)
            r = run_episode(env, state, seed, lambda obs: base.act(obs, task)[0], args.max_steps)
            episodes.append(episode_arrays(record["feats"], record["z"], r["success"], learner.cfg.gamma))
            rows.append((state, seed, int(r["success"]), r["steps"], len(record["z"])))
            print(f"[collect] epoch {args.epoch} init {state}: success={int(r['success'])} steps={r['steps']} "
                  f"decisions={len(record['z'])} time={time.time() - t0:.1f}s", flush=True)
    finally:
        uninstall(base.policy)
        env.close()
    write_npz(path, rows=np.array(rows, dtype=np.int64), **concat_arrays(episodes))
    return 0


def merge_epoch(out: Path, epoch: int, args, states: list[int]) -> Path:
    """The epoch's parts as ``replay/eNNN.npz``, checked against the episodes the epoch should hold."""
    name = EPOCH_FILE.format(epoch)
    merged = out / "replay" / f"{name}.npz"
    if merged.exists():
        return merged
    merged.parent.mkdir(exist_ok=True)
    parts = [dict(np.load(out / "collect" / name / f"part{k}.npz")) for k in range(args.parts)]
    rows = np.concatenate([p.pop("rows") for p in parts])
    expected = {j[0] for k in range(args.parts) for j in part_jobs(states, epoch, k, args.parts)}
    if sorted(rows[:, 0].tolist()) != sorted(expected) or len(expected) != len(states):
        raise SystemExit(f"epoch {epoch}: parts hold states {sorted(rows[:, 0].tolist())}")
    if rows[:, 4].sum() != sum(len(p["z"]) for p in parts):
        raise SystemExit(f"epoch {epoch}: decisions do not match the episode rows")
    write_npz(merged, rows=rows, **concat_arrays(parts))
    os.sync()
    return merged


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_rows(path: Path, before_epoch: int) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if int(r["epoch"]) < before_epoch]


def replay_arrays(out: Path, epoch: int) -> tuple[dict, list[dict]]:
    """(arrays of epochs 0..epoch, the train-log rows of those epochs, rebuilt from the replay files)."""
    from noise_awr import REPLAY_ARRAYS

    held = [dict(np.load(out / "replay" / f"{EPOCH_FILE.format(e)}.npz")) for e in range(epoch + 1)]
    log = [dict(zip(LOG_FIELDS, (e, *r))) for e, h in enumerate(held) for r in h["rows"].tolist()]
    return {k: np.concatenate([h[k] for h in held]) for k in REPLAY_ARRAYS}, log


def run_updates(learner, replay, updates: int, device) -> list[dict]:
    """``updates`` joint updates, each followed by a value-only update."""
    cfg, out = learner.cfg, []
    for _ in range(updates):
        out.append(learner.update(replay.sample(cfg.batch_size, cfg.success_fraction, device), update_actor=True))
        learner.update(replay.sample(cfg.batch_size, cfg.success_fraction, device), update_actor=False)
    return out


def update(args) -> int:
    import torch

    from noise_awr import Replay
    from vla_core.envs import parse_index_ranges

    out = Path(args.output_dir)
    ckpt = checkpoint_path(out, args.epoch)
    if ckpt.exists():
        print(f"[update] {ckpt} exists", flush=True)
        return 0
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provenance(out, args)
    states = parse_index_ranges(args.train_init_states)
    for e in range(args.epoch + 1):     # a lost upload of an earlier epoch is merged again from its parts
        merge_epoch(out, e, args, states)
    arrays, log = replay_arrays(out, args.epoch)
    learner = load_learner(checkpoint_path(out, args.epoch - 1), device, args) if args.epoch else \
        fresh_learner(arrays["feats"].shape[1], args.train_seed, device)
    learner.actor.train()
    replay = Replay(arrays, seed=args.train_seed * 10_000 + args.epoch)
    losses = run_updates(learner, replay, args.updates_per_epoch, device)
    with torch.no_grad():
        probe = torch.from_numpy(arrays["feats"][-2048:].astype(np.float32)).to(device)
        mean, log_std = learner.actor(probe)
    tmp = f"{ckpt}.tmp"
    torch.save({"learner": learner.state_dict(), "run_config": run_config(args), "epoch": args.epoch}, tmp)
    os.sync()
    os.replace(tmp, ckpt)
    os.sync()
    write_rows(out / "train_log.csv", LOG_FIELDS, log)

    def average(key):
        vals = [x[key] for x in losses if key in x]
        return round(float(np.mean(vals)), 6) if vals else ""

    ups = read_rows(out / "update_log.csv", args.epoch)
    ups.append({"epoch": args.epoch, "replay_decisions": len(replay), "replay_success_decisions": len(replay.succ),
                "value_loss": average("value_loss"), "actor_loss": average("actor_loss"),
                "weight_mean": average("weight_mean"), "mean_abs": round(float(mean.abs().mean()), 6),
                "std_mean": round(float(log_std.exp().mean()), 6), "time_s": round(time.time() - t0, 1)})
    write_rows(out / "update_log.csv", UPDATE_FIELDS, ups)
    print(f"[update] epoch {args.epoch}: {ups[-1]}", flush=True)
    return 0


def evaluate(args) -> int:
    from noise_steering import install_repeated, uninstall
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode

    if not args.arm or not args.eval_dir:
        raise SystemExit("eval needs --arm and --eval_dir")
    out = Path(args.eval_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"per_episode_{args.arm}.csv"
    done = load_episode_rows(path)
    jobs = [(s, i) for s in parse_index_ranges(args.seeds) for i in parse_index_ranges(args.init_states)
            if (s, i) not in done]
    print(f"[eval] {args.arm}: {len(jobs)} episodes to run", flush=True)
    if not jobs:
        return 0
    ckpt = checkpoint_path(Path(args.output_dir), args.epoch)
    if args.arm in LEARNED_ARMS and not ckpt.exists():
        raise SystemExit(f"{ckpt} is missing")
    require_mujoco_patch()
    provenance(out, args)
    env, base, device = load_base(args, hard_reset=True)
    task = env.task_description
    learner = None
    if args.arm in LEARNED_ARMS:
        learner = load_learner(ckpt, device, args, BASE_KEYS)
    elif args.arm != "base":
        learner = fresh_learner(feature_dim(base.policy), args.train_seed, device)
    deterministic = args.arm in MEAN_ARMS
    if learner is not None:
        learner.actor.eval()
        install_repeated(base.policy, lambda feats: learner.actor.sample(feats.float(), deterministic)[0])
    try:
        for s, i in jobs[: args.max_episodes_per_process]:
            base.reset()
            t0 = time.time()
            r = run_episode(env, i, s, lambda obs: base.act(obs, task)[0], args.max_steps)
            append_episode_row(path, {"task_id": args.task_id, "seed": s, "init_state": i,
                                      "success": int(r["success"]), "steps": r["steps"]})
            print(f"[eval] {args.arm} seed {s} init {i}: success={int(r['success'])} steps={r['steps']} "
                  f"time={time.time() - t0:.1f}s", flush=True)
    finally:
        uninstall(base.policy)
        env.close()
    return EXIT_MORE_EPISODES if len(jobs) > args.max_episodes_per_process else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return {"collect": collect, "update": update, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
