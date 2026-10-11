"""
Feature cache for residual-target training of the hybrid diffusion head.

The SmolVLA backbone is frozen, so its outputs for every demonstration frame can be computed
once. `build` replays the LIBERO demo states in the simulator, renders observations through the
same path evaluate_in_environment uses, and stores per frame: the pooled semantic feature, K base
action chunk samples, and the normalized demo action chunk. `train` fits the diffusion head on
those tensors and writes checkpoints that train_hybrid_diffusion.py --eval_only can load.

Exact commands used for reported numbers: run_task3_residual.sh
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_diffusion_agent import ConditionalDDPMHead, flow_matching_loss, sample_flow


def build_cache(args):
    import h5py
    # Importing the training module applies the MuJoCo / torch.load / LIBERO config patches.
    import train_hybrid_diffusion as thd
    from hybrid_diffusion_agent import HybridFrozenBrainDiffusionHands

    torch.manual_seed(args.seed)
    model = HybridFrozenBrainDiffusionHands(
        base_policy_path=args.base_policy_path,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    preprocessor, postprocessor = thd.make_pre_post_processors(
        policy_cfg=model.base_policy.config,
        pretrained_path=args.base_policy_path,
        preprocessor_overrides={"device_processor": {"device": str(args.device)}},
    )

    # The postprocessor is affine (mean/std un-normalization); read its parameters off two probes.
    probe = torch.stack([torch.zeros(args.action_dim), torch.ones(args.action_dim)]).to(args.device)
    probe_out = postprocessor(probe).detach().float().cpu()
    action_mean, action_std = probe_out[0], probe_out[1] - probe_out[0]
    print(f"[cache] action mean={action_mean.tolist()}\n[cache] action std={action_std.tolist()}")

    benchmark = thd.get_benchmark_dict()["libero_10"]()
    benchmark_root = os.path.dirname(thd.libero_pkg.__file__)
    task = benchmark.get_task(args.task_id)
    h5_path = Path(args.data_dir) / f"{task.name}_demo.hdf5"
    print(f"[cache] task {args.task_id}: {task.language}\n[cache] demos: {h5_path}")

    env = thd.create_libero_env(args.task_id, benchmark, benchmark_root)
    env.reset()

    out = {"semantics": [], "base_chunks": [], "gt_chunks": [], "demo_idx": [], "step_idx": []}
    pending, pending_meta = [], []

    def flush():
        if not pending:
            return
        batch = {}
        for key, value in pending[0].items():
            if isinstance(value, torch.Tensor):
                batch[key] = torch.cat([b[key] for b in pending], dim=0)
            elif isinstance(value, list):
                batch[key] = [x for b in pending for x in b[key]]
            else:
                batch[key] = value
        semantics, base = model.encode(batch, with_base_action=True, num_base_samples=args.num_base_samples)
        if args.num_base_samples == 1:
            base = base.unsqueeze(1)
        out["semantics"].append(semantics.cpu())
        out["base_chunks"].append(base.cpu())
        for gt, demo_i, step_i in pending_meta:
            out["gt_chunks"].append(gt)
            out["demo_idx"].append(demo_i)
            out["step_idx"].append(step_i)
        pending.clear()
        pending_meta.clear()

    with h5py.File(h5_path, "r") as h5:
        demo_keys = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if args.max_demos is not None:
            demo_keys = demo_keys[: args.max_demos]
        for demo_i, dk in enumerate(demo_keys):
            states = h5[f"data/{dk}/states"][:]
            actions = torch.from_numpy(h5[f"data/{dk}/actions"][:].astype(np.float32))
            actions = (actions - action_mean) / action_std
            n_steps = actions.shape[0]
            for t in range(n_steps):
                obs = env.set_init_state(states[t])
                raw_obs = thd.get_policy_obs_from_env_obs(obs, flip_mode=args.image_flip_mode)
                policy_obs = thd.preprocess_observation(raw_obs)
                policy_obs["task"] = [task.language]
                pending.append(preprocessor(policy_obs))
                idx = torch.arange(t, t + args.chunk_size).clamp(max=n_steps - 1)
                pending_meta.append((actions[idx], demo_i, t))
                if len(pending) >= args.batch_size:
                    flush()
                    thd.clear_preprocessor_state(preprocessor)
            flush()
            print(f"[cache] demo {demo_i + 1}/{len(demo_keys)} ({n_steps} steps)", flush=True)
    env.close()

    cache = {
        "semantics": torch.cat(out["semantics"]),
        "base_chunks": torch.cat(out["base_chunks"]),
        "gt_chunks": torch.stack(out["gt_chunks"]),
        "demo_idx": torch.tensor(out["demo_idx"]),
        "step_idx": torch.tensor(out["step_idx"]),
        "meta": {
            "task_id": args.task_id,
            "task_language": task.language,
            "base_policy_path": args.base_policy_path,
            "base_num_steps": int(model.base_policy.model.config.num_steps),
            "image_flip_mode": args.image_flip_mode,
            "chunk_size": args.chunk_size,
            "action_dim": args.action_dim,
            "action_mean": action_mean,
            "action_std": action_std,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(cache, args.out)
    print(f"[cache] saved {cache['semantics'].shape[0]} frames to {args.out}")


@torch.no_grad()
def evaluate_head(head, sem, base, gt, steps, generator):
    """Held-out flow loss, and action error of base vs base + sampled residual (normalized units)."""
    k = torch.randint(base.shape[1], (sem.shape[0],), generator=generator, device=sem.device)
    b = base[torch.arange(sem.shape[0], device=sem.device), k]
    target = gt - b
    noise = torch.randn(target.shape, generator=generator, device=sem.device)
    timesteps = torch.rand((sem.shape[0],), generator=generator, device=sem.device)
    _, flow, _ = flow_matching_loss(head, sem, target, b, noise=noise, timesteps=timesteps)
    x0 = torch.randn(target.shape, generator=generator, device=sem.device)
    residual = sample_flow(head, sem, b, steps=steps, noise=x0)
    return {
        "flow_loss": flow.item(),
        "base_mse": target.square().mean().item(),
        "corrected_mse": (target - residual).square().mean().item(),
        "base_mse_step0": target[:, 0].square().mean().item(),
        "corrected_mse_step0": (target[:, 0] - residual[:, 0]).square().mean().item(),
        "residual_abs_step0": residual[:, 0].abs().mean().item(),
    }


def train_head(args):
    torch.manual_seed(args.seed)
    train_args = {k: v for k, v in vars(args).items() if k != "fn"}
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    device = args.device
    sem, base, gt = (cache[k].float().to(device) for k in ("semantics", "base_chunks", "gt_chunks"))
    demo_idx = cache["demo_idx"]

    demos = torch.unique(demo_idx)
    perm = demos[torch.randperm(len(demos), generator=torch.Generator().manual_seed(args.seed))]
    n_val = int(round(args.val_demo_fraction * len(demos)))
    val_mask = torch.isin(demo_idx, perm[:n_val]).to(device)
    tr = (~val_mask).nonzero().squeeze(1)
    va = val_mask.nonzero().squeeze(1)
    print(f"[train] {len(tr)} train frames / {len(va)} val frames ({len(demos) - n_val}/{n_val} demos)")

    head_config = {
        "action_dim": meta["action_dim"],
        "chunk_size": meta["chunk_size"],
        "cond_dim": sem.shape[1],
        "hidden_dim": args.diff_hidden_dim,
        "num_layers": args.diff_layers,
        "cond_on_base_action": True,
        "time_scale": args.time_scale,
        "standardize_cond": True,
    }
    head = ConditionalDDPMHead(**head_config).to(device)
    head.set_cond_stats(sem[tr].mean(0), sem[tr].std(0))
    ema = copy.deepcopy(head).requires_grad_(False)

    steps_per_epoch = max(1, len(tr) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, config={**train_args, "cache_meta": {k: v for k, v in meta.items() if not torch.is_tensor(v)}}, job_type="residual_cache_train")

    os.makedirs(args.out_dir, exist_ok=True)
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = tr[torch.randperm(len(tr), device=device)]
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            idx = order[i * args.batch_size : (i + 1) * args.batch_size]
            k = torch.randint(base.shape[1], (len(idx),), device=device)
            b = base[idx, k]
            cond = sem[idx]
            if args.cond_noise_std > 0:
                # Feature-space augmentation, in units of each feature's across-frame std.
                cond = cond + args.cond_noise_std * head.cond_std * torch.randn_like(cond)
            loss, flow, delta = flow_matching_loss(head, cond, gt[idx] - b, b, delta_l2_weight=args.delta_l2_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            with torch.no_grad():
                decay = min(args.ema_decay, (1 + step) / (10 + step))
                for p_ema, p in zip(ema.parameters(), head.parameters()):
                    p_ema.lerp_(p, 1 - decay)
            epoch_loss += flow.item()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ema.eval()
            row = {"epoch": epoch, "train_flow_loss": epoch_loss / steps_per_epoch}
            if len(va) > 0:
                gen = torch.Generator(device=device).manual_seed(0)
                row.update({f"val/{k}": v for k, v in evaluate_head(ema, sem[va], base[va], gt[va], args.sample_steps, gen).items()})
            gen = torch.Generator(device=device).manual_seed(0)
            sub = tr[:: max(1, len(tr) // max(1000, len(va)))]
            row.update({f"train/{k}": v for k, v in evaluate_head(ema, sem[sub], base[sub], gt[sub], args.sample_steps, gen).items()})
            history.append(row)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            if run is not None:
                run.log(row, step=step)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": {"diffusion_head": ema.state_dict()},
                "head_config": head_config,
                "residual_target": True,
                "cache_meta": meta,
                "train_args": train_args,
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"hybrid_diff_epoch_{epoch:03d}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "latest_checkpoint.pt"))

    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if run is not None:
        run.finish()
    print(f"[train] done. checkpoints in {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Replay demos and cache frozen-backbone features and base action chunks.")
    b.add_argument("--task_id", type=int, required=True)
    b.add_argument("--out", type=str, required=True)
    b.add_argument("--data_dir", type=str, default=os.path.expanduser("~/lerobot_datasets/libero_10"))
    b.add_argument("--base_policy_path", type=str, default="HuggingFaceVLA/smolvla_libero")
    b.add_argument("--image_flip_mode", type=str, default="vertical_horizontal")
    b.add_argument("--num_base_samples", type=int, default=4, help="Base action chunk samples stored per frame.")
    b.add_argument("--batch_size", type=int, default=32)
    b.add_argument("--chunk_size", type=int, default=16)
    b.add_argument("--action_dim", type=int, default=7)
    b.add_argument("--max_demos", type=int, default=None)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", type=str, default="cuda")
    b.set_defaults(fn=build_cache)

    t = sub.add_parser("train", help="Train the residual diffusion head on a feature cache.")
    t.add_argument("--cache", type=str, required=True)
    t.add_argument("--out_dir", type=str, required=True)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight_decay", type=float, default=1e-4)
    t.add_argument("--diff_hidden_dim", type=int, default=256)
    t.add_argument("--diff_layers", type=int, default=5)
    t.add_argument("--time_scale", type=float, default=1000.0)
    t.add_argument("--delta_l2_weight", type=float, default=0.0)
    t.add_argument("--cond_noise_std", type=float, default=0.0, help="Gaussian noise on standardized semantic features during training.")
    t.add_argument("--ema_decay", type=float, default=0.999)
    t.add_argument("--val_demo_fraction", type=float, default=0.1)
    t.add_argument("--sample_steps", type=int, default=10)
    t.add_argument("--eval_every", type=int, default=10)
    t.add_argument("--save_every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", type=str, default="cuda")
    t.add_argument("--wandb_project", type=str, default="hybrid_diffusion_vla", help="Empty string disables W&B.")
    t.set_defaults(fn=train_head)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
