#!/usr/bin/env python3
"""Per-position flow-matching loss of the base on recorded successes (audit A0, notes repo Stage 2 plan).

Training targets in self_filtered_bc / step 10 are the next ``chunk_size`` *executed* actions of a frame
(``rollout_data.action_chunk``). At a replan frame (step a multiple of ``n_action_steps``) positions
``0 .. n_action_steps-1`` are the chunk the base executed there; later positions come from later replans. This
prints the base's loss at each chunk position on the replan frames of several episode sets, with the same fixed
noise and times for every position of a batch, to ask whether constant-noise (``repeat``) targets fit the base
better at positions >= n_action_steps than sampled-noise targets do.

    python noise_reachability/check_target_positions.py --set A=<dir>/episodes --set N_repeat='<dir>/episodes/repeat_*'

A set is NAME=GLOB[,GLOB...] of episode .npz files (GLOB without .npz gets ``*.npz`` appended when it is a
folder). Only successful episodes are used. Run from the repo root with the repo root on PYTHONPATH (FAIL-003).
Reference only: no decision rule depends on it.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "self_filtered_bc"))

BINS = ((0, 1), (1, 10), (10, 20), (20, 30), (30, 40), (40, 50))


def only_position(pad, k):
    """Pad mask that keeps only chunk position ``k`` of each frame (and still drops positions already padded)."""
    out = pad.clone() if hasattr(pad, "clone") else pad.copy()
    out[..., :] = True
    out[..., k] = pad[..., k]
    return out


def bin_means(sums, counts, bins=BINS):
    """Frame-weighted mean loss over each [lo, hi) range of positions."""
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    return {f"{lo}-{hi - 1}": float(sums[lo:hi].sum() / counts[lo:hi].sum()) if counts[lo:hi].sum() else float("nan")
            for lo, hi in bins}


def expand(spec):
    paths = []
    for g in spec.split(","):
        g = os.path.join(g, "*.npz") if os.path.isdir(g) else (g if g.endswith(".npz") else g + ".npz")
        paths += glob.glob(g)
    return sorted(set(paths))


def main(argv=None) -> int:
    import torch
    from torch.utils.data import DataLoader, Subset

    from rollout_data import PAD_KEY, RolloutDataset, load_meta
    from self_filtered_bc import flow_loss, policy_batch, require_mujoco_patch
    from vla_core.envs import libero_env_config
    from vla_core.evaluation import seed_everything
    from vla_core.policy import DEFAULT_RENAME_MAP, load_base_policy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", action="append", required=True, help="NAME=GLOB[,GLOB...]")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_episodes", type=int, default=20, help="per set, drawn with --seed (RAM bound)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON with the per-position results")
    args = p.parse_args(argv)
    require_mujoco_patch()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=520, hard_reset=True)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    policy, chunk = base.policy, base.policy.config.chunk_size
    result = {"args": vars(args), "sets": {}}
    for spec in args.set:
        name, globs = spec.split("=", 1)
        paths = [q for q in expand(globs) if load_meta(q)["success"]]
        tasks = {load_meta(q).get("task_id", args.task_id) for q in paths}
        if not paths or tasks != {args.task_id}:
            raise SystemExit(f"set {name}: {len(paths)} successful episodes, task ids {tasks} (want {args.task_id})")
        rng = np.random.default_rng(args.seed)
        if len(paths) > args.max_episodes:
            paths = sorted(rng.choice(paths, size=args.max_episodes, replace=False).tolist())
        ds = RolloutDataset(paths, chunk_size=chunk)
        keep = [i for i, (e, f) in enumerate(ds.index) if int(ds.episodes[e]["frame_steps"][f]) % args.n_action_steps == 0]
        loader = DataLoader(Subset(ds, keep), batch_size=args.batch_size)
        sums, counts = np.zeros(chunk), np.zeros(chunk)
        with torch.no_grad():
            for b, sample in enumerate(loader):
                batch = policy_batch(sample, base.preprocessor, device)
                pad = batch[PAD_KEY]
                for k in range(chunk):
                    valid = int((~pad[:, k]).sum())
                    if not valid:
                        continue
                    seed_everything(10_000 + b)        # same noise and times for every position of this batch
                    loss = flow_loss(policy, {**batch, PAD_KEY: only_position(pad, k)}, device, True)
                    sums[k] += float(loss) * valid
                    counts[k] += valid
        per_pos = (sums / np.maximum(counts, 1)).tolist()
        result["sets"][name] = {"episodes": [Path(q).name for q in paths], "replan_frames": len(keep),
                                "per_position": per_pos, "frames_per_position": counts.tolist(),
                                "bins": bin_means(sums, counts)}
        bins = " ".join(f"{k} {v:.4f}" for k, v in result["sets"][name]["bins"].items())
        print(f"[positions] {name}: {len(paths)} episodes, {len(keep)} replan frames; {bins}", flush=True)
        del ds, loader
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
