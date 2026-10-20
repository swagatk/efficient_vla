#!/usr/bin/env python3
"""Self-filtered behaviour cloning: LoRA-fine-tune the frozen SmolVLA on its own successful rollouts.

Stages (``--stage``), see README.md for the hypothesis and run_self_filtered_bc.sh for the exact commands:

  collect  run the base policy on training init states x collection seeds, save every episode
           (rollout_data.py) and append a row to <data_dir>/collection.csv
  train    fit a LoRA adapter on SmolVLA's action expert with its own flow-matching loss on the episodes chosen
           by --selection (success: the method; all_matched: the control)
  eval     run base and/or LoRA on held-out init states with vla_core's episode seeds, write
           per_episode_<policy>.csv and, with both, the paired comparison

The base policy, environment and episode loop are vla_core's, so the base runs exactly as in the residual-SAC
evaluation (and as lerobot-eval). Run from the repo root (sitecustomize loads robosuite_mujoco_patch, FAIL-003).
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollout_data import (PAD_KEY, SELECTIONS, EpisodeRecorder, RolloutDataset, load_meta,  # noqa: E402
                          save_episode, select_episodes)
from vla_core.policy import DEFAULT_RENAME_MAP  # noqa: E402

EXIT_MORE_EPISODES = 3
TRACKED_PATHS = ("self_filtered_bc", "vla_core", "baseline_performance", "robosuite_mujoco_patch.py",
                 "sitecustomize.py")
# Settings a LoRA adapter must share with the base policy it is evaluated on.
BASE_KEYS = ("policy_path", "n_action_steps", "rename_map", "task_id")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["collect", "train", "eval"], required=True)
    p.add_argument("--task_id", type=int, default=6, help="LIBERO-10 task index")
    p.add_argument("--seed", type=int, default=0, help="Training seed (Python, NumPy, PyTorch, LoRA init, batches)")
    p.add_argument("--policy_path", type=str, default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=1,
                   help="Override the checkpoint's n_action_steps (1 = replan every step, FAIL-005); -1 keeps it")
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520, help="Episode horizon")
    p.add_argument("--max_episodes_per_process", type=int, default=0,
                   help="Stop after this many episodes with exit code 3 (0 = no limit); keeps hard-reset "
                        "evaluation within FAIL-004's limit when set to 3")
    # collect
    p.add_argument("--data_dir", type=str, default="self_filtered_bc_data/task_6")
    p.add_argument("--init_states", type=str, default="10-49", help="Training init states to collect on")
    p.add_argument("--collect_seeds", type=str, default="0-2", help="Episode seeds per init state")
    p.add_argument("--frame_stride", type=int, default=2, help="Store images every N steps (actions every step)")
    p.add_argument("--hard_reset", type=int, default=0,
                   help="Collection: 1 = rebuild the simulator every episode. Soft resets gave the same outcomes "
                        "and step counts on 20/20 episodes (week6/sac_residual_task6_pilot1/diagnostic/)")
    # train
    p.add_argument("--selection", choices=SELECTIONS, default="success")
    p.add_argument("--val_init_states", type=str, default="",
                   help="Training init states whose episodes are held out of training for the validation loss")
    p.add_argument("--output_dir", type=str, default="checkpoints_sfbc/task_6/success_seed0")
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, default="all", help="'attn', 'all' or comma-separated module names")
    p.add_argument("--train_action_heads", type=int, default=0, help="1 = also fine-tune the action in/out heads")
    p.add_argument("--bf16", type=int, default=1, help="bfloat16 autocast on CUDA")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--wandb_project", type=str, default=None, help="Log to W&B as well (off by default)")
    # eval
    p.add_argument("--eval_policies", type=str, default="base,lora", help="Comma-separated: base, lora")
    p.add_argument("--lora_dir", type=str, default=None, help="Default: <output_dir>/adapter")
    p.add_argument("--eval_init_states", type=str, default="0-9")
    p.add_argument("--eval_seeds", type=str, default="0,1,2")
    p.add_argument("--eval_output_dir", type=str, default=None, help="Default: <output_dir>/eval")
    args = p.parse_args(argv)
    if args.n_action_steps is not None and args.n_action_steps < 0:
        args.n_action_steps = None
    args.rename_map = args.rename_map or None
    args.lora_dir = args.lora_dir or os.path.join(args.output_dir, "adapter")
    args.eval_output_dir = args.eval_output_dir or os.path.join(args.output_dir, "eval")
    return args


def require_mujoco_patch():
    if "robosuite_mujoco_patch" not in sys.modules:
        raise SystemExit("robosuite_mujoco_patch is not loaded: run from the repo root or with it on PYTHONPATH "
                         "(FAIL-003)")


def log(msg: str) -> None:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} maxrss {rss_gb:.1f}GB] {msg}", flush=True)


def write_provenance(out_dir, args, extra=None) -> None:
    from vla_core.provenance import append_jsonl, collect, git_state

    os.makedirs(out_dir, exist_ok=True)
    record = collect(str(REPO_ROOT), {**vars(args), **(extra or {})})
    record["git_dirty_sfbc"] = git_state(str(REPO_ROOT), TRACKED_PATHS)["git_dirty"]
    append_jsonl(os.path.join(out_dir, "provenance.jsonl"), record)


def make_env_and_base(args, hard_reset: bool):
    import torch

    from vla_core.envs import libero_env_config, make_libero_env
    from vla_core.policy import load_base_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=args.max_steps, hard_reset=hard_reset)
    env = make_libero_env(env_cfg)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    return env, base, device


def recording_actor(base, task, recorder):
    """choose_action for run_episode: the base policy's action, recorded with the observation it came from."""
    def choose(obs):
        action, env_obs = base.act(obs, task)
        recorder.add(env_obs, action)
        return action
    return choose


# ----------------------------------------------------------------------------------------------------- collect

def collect(args) -> int:
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode

    require_mujoco_patch()
    data_dir = Path(args.data_dir)
    episodes_dir = data_dir / "episodes"
    csv_path = data_dir / "collection.csv"
    done = load_episode_rows(csv_path)
    jobs = [(s, i) for s in parse_index_ranges(args.collect_seeds) for i in parse_index_ranges(args.init_states)
            if (s, i) not in done]
    log(f"[collect] {len(done)} episodes done, {len(jobs)} to run in {data_dir}")
    if not jobs:
        return 0
    write_provenance(data_dir, args, {"stage": "collect"})
    limit = args.max_episodes_per_process or len(jobs)
    env, base, _ = make_env_and_base(args, hard_reset=bool(args.hard_reset))
    try:
        for s, i in jobs[:limit]:
            base.reset()
            recorder = EpisodeRecorder(args.frame_stride)
            t0 = time.time()
            r = run_episode(env, i, s, recording_actor(base, env.task_description, recorder), args.max_steps)
            meta = {"task_id": args.task_id, "task": env.task_description, "seed": s, "init_state": i,
                    "success": bool(r["success"]), "steps": r["steps"], "policy_path": args.policy_path,
                    "n_action_steps": args.n_action_steps, "rename_map": args.rename_map}
            save_episode(episodes_dir / f"seed{s}_init{i:02d}.npz", recorder, meta)
            append_episode_row(csv_path, {"task_id": args.task_id, "seed": s, "init_state": i,
                                          "success": int(r["success"]), "steps": r["steps"]})
            log(f"[collect] seed {s} init {i}: success={int(r['success'])} steps={r['steps']} "
                f"({time.time() - t0:.0f}s)")
    finally:
        env.close()
    rows = load_episode_rows(csv_path)
    log(f"[collect] {sum(int(r['success']) for r in rows.values())}/{len(rows)} successful so far")
    return EXIT_MORE_EPISODES if len(jobs) > limit else 0


# ------------------------------------------------------------------------------------------------------- train

def split_episodes(data_dir, selection, val_init_states, seed):
    """(training paths, validation paths) for --selection; validation episodes come from held-out training
    init states and are never trained on."""
    paths = sorted((Path(data_dir) / "episodes").glob("*.npz"))
    if not paths:
        raise SystemExit(f"no episodes in {data_dir}/episodes; run --stage collect first")
    metas = [load_meta(p) for p in paths]
    val = [k for k, m in enumerate(metas) if m["init_state"] in val_init_states]
    rest = [k for k in range(len(paths)) if k not in set(val)]
    chosen = select_episodes([metas[k] for k in rest], selection, seed=seed)
    val_chosen = select_episodes([metas[k] for k in val], "success", seed=seed) if val else []
    return [paths[rest[k]] for k in chosen], [paths[val[k]] for k in val_chosen], metas


def policy_batch(sample_batch: dict, preprocessor, device) -> dict:
    """Collated RolloutDataset samples -> the batch SmolVLA's forward expects (renamed, normalised, tokenised)."""
    batch = dict(sample_batch)
    pad = batch.pop(PAD_KEY)
    out = preprocessor(batch)
    out[PAD_KEY] = pad.to(device)
    return out


def flow_loss(policy, batch, device, bf16: bool):
    import torch

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and device == "cuda"):
        loss, _ = policy.forward(batch)
    return loss


def validation_loss(policy, preprocessor, loader, device, bf16, seed) -> float:
    """Mean flow-matching loss with fixed noise and times (seeded), so values are comparable across steps."""
    import torch

    from vla_core.evaluation import seed_everything

    state = torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    seed_everything(10_000 + seed)
    losses = []
    with torch.no_grad():
        for sample in loader:
            losses.append(flow_loss(policy, policy_batch(sample, preprocessor, device), device, bf16).item())
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])
    return float(np.mean(losses)) if losses else float("nan")


def train(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    from vla_core.envs import libero_env_config, parse_index_ranges
    from vla_core.evaluation import seed_everything
    from vla_core.lora import (LoraSettings, add_lora, resolve_targets, save_lora, set_heads_trainable,
                               trainable_parameters)
    from vla_core.policy import load_base_policy

    seed_everything(args.seed)
    val_states = set(parse_index_ranges(args.val_init_states)) if args.val_init_states else set()
    train_paths, val_paths, metas = split_episodes(args.data_dir, args.selection, val_states, args.seed)
    if not train_paths:
        raise SystemExit(f"selection {args.selection!r} gives no training episodes")
    for m in metas:
        for key in ("policy_path", "task_id"):
            if m[key] != getattr(args, key):
                raise SystemExit(f"episode collected with {key}={m[key]!r}, training has {getattr(args, key)!r}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_info = {"train_episodes": [p.name for p in train_paths], "val_episodes": [p.name for p in val_paths],
                 "n_collected": len(metas), "n_collected_success": sum(m["success"] for m in metas)}
    write_provenance(out, args, {"stage": "train", **data_info})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_base_policy(args.policy_path, libero_env_config(args.task_id, episode_length=args.max_steps),
                            device=device, n_action_steps=args.n_action_steps, rename_map=args.rename_map)
    policy, model = base.policy, base.policy.model
    settings = LoraSettings(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                            target_modules=resolve_targets(args.lora_targets))
    model.vlm_with_expert.lm_expert = add_lora(model.vlm_with_expert.lm_expert, settings)
    set_heads_trainable(model, bool(args.train_action_heads))
    params = trainable_parameters(policy)
    log(f"[train] {len(train_paths)} training episodes ({args.selection}), {len(val_paths)} validation episodes, "
        f"{sum(p.numel() for p in params):,} trainable parameters")

    train_ds = RolloutDataset(train_paths, chunk_size=policy.config.chunk_size)
    val_ds = RolloutDataset(val_paths, chunk_size=policy.config.chunk_size) if val_paths else None
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=gen,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None
    log(f"[train] {len(train_ds)} training frames, {len(val_ds) if val_ds else 0} validation frames")

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)) *
        0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, args.train_steps)))))
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config={**vars(args), **data_info},
                               name=out.name, dir=str(out))
    history = []
    step, it, t_start = 0, iter(loader), time.time()
    policy.eval()  # no dropout in the frozen parts; LoRA dropout is 0 by default
    while step < args.train_steps:
        try:
            sample = next(it)
        except StopIteration:
            it = iter(loader)
            sample = next(it)
        loss = flow_loss(policy, policy_batch(sample, base.preprocessor, device), device, bool(args.bf16))
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        rec = {"step": step, "loss": loss.item(), "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
               "sec_per_step": (time.time() - t_start) / step}
        if val_loader is not None and (step % args.val_every == 0 or step == args.train_steps):
            rec["val_loss"] = validation_loss(policy, base.preprocessor, val_loader, device, bool(args.bf16), args.seed)
        if step % args.log_every == 0 or "val_loss" in rec or step == 1:
            history.append(rec)
            log(f"[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if wandb_run is not None:
                wandb_run.log(rec, step=step)
    save_lora(model.vlm_with_expert.lm_expert, out / "adapter", heads=model if args.train_action_heads else None)
    run_config = {k: getattr(args, k) for k in BASE_KEYS} | {"lora": settings.to_dict(),
                                                             "train_action_heads": args.train_action_heads}
    (out / "adapter" / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (out / "train_history.json").write_text(json.dumps(history))
    if wandb_run is not None:
        wandb_run.finish()
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    log(f"[train] saved {out / 'adapter'}; peak GPU memory {peak:.1f} GB")
    return 0


# -------------------------------------------------------------------------------------------------------- eval

def attach_lora(base, lora_dir, args):
    from vla_core.lora import load_heads, load_lora

    run_config = json.loads((Path(lora_dir) / "run_config.json").read_text())
    for key in BASE_KEYS:
        if run_config.get(key) != getattr(args, key):
            raise SystemExit(f"adapter was trained with {key}={run_config.get(key)!r}, evaluation has "
                             f"{getattr(args, key)!r}")
    model = base.policy.model
    model.vlm_with_expert.lm_expert = load_lora(model.vlm_with_expert.lm_expert, lora_dir)
    if load_heads(model, lora_dir):
        log("[eval] loaded fine-tuned action heads")
    model.to(base.policy.config.device)
    base.policy.eval()


def evaluate(args) -> int:
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode

    require_mujoco_patch()
    sys.path.insert(0, str(REPO_ROOT / "baseline_performance"))
    policies = [p.strip() for p in args.eval_policies.split(",") if p.strip()]
    if not policies or set(policies) - {"base", "lora"}:
        raise SystemExit(f"--eval_policies must name base and/or lora, got {args.eval_policies!r}")
    seeds, states = parse_index_ranges(args.eval_seeds), parse_index_ranges(args.eval_init_states)
    out = Path(args.eval_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = {p: load_episode_rows(out / f"per_episode_{p}.csv") for p in policies}
    log(f"[eval] done so far: " + ", ".join(f"{p} {len(done[p])}" for p in policies))
    for p in policies:  # one policy per process: the LoRA wraps the base policy in place
        jobs = [(s, i) for s in seeds for i in states if (s, i) not in done[p]]
        if not jobs:
            continue
        write_provenance(out, args, {"stage": "eval", "policy": p})
        limit = args.max_episodes_per_process or len(jobs)
        env, base, _ = make_env_and_base(args, hard_reset=True)
        try:
            if p == "lora":
                attach_lora(base, args.lora_dir, args)
            for s, i in jobs[:limit]:
                base.reset()
                task = env.task_description
                r = run_episode(env, i, s, lambda obs: base.act(obs, task)[0], args.max_steps)
                append_episode_row(out / f"per_episode_{p}.csv", {"task_id": args.task_id, "seed": s, "init_state": i,
                                                                 "success": int(r["success"]), "steps": r["steps"]})
                log(f"[eval] {p} seed {s} init {i}: success={int(r['success'])} steps={r['steps']}")
        finally:
            env.close()
        if len(jobs) > limit or p != policies[-1]:
            return EXIT_MORE_EPISODES
        break

    done ={p: load_episode_rows(out / f"per_episode_{p}.csv") for p in policies}
    for p in policies:
        rows = [done[p][(s, i)] for s in seeds for i in states]
        n = sum(int(r["success"]) for r in rows)
        log(f"[eval] {p}: {n}/{len(rows)} = {100 * n / len(rows):.1f}%")
    if set(policies) == {"base", "lora"}:
        from compare_per_state import compare, format_report

        def eps(p):
            return {(args.task_id, s, i): int(done[p][(s, i)]["success"]) for s in seeds for i in states}

        report = format_report(compare(eps("base"), eps("lora")), "base", "lora")
        (out / "comparison_base_vs_lora.txt").write_text(report)
        print(report)
    return 0


def main(argv=None) -> int:
    faulthandler.enable()
    args = parse_args(argv)
    return {"collect": collect, "train": train, "eval": evaluate}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
