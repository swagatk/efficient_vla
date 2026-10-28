"""Offline re-run of step-3 SAC on the recorded task-7 replay, in the training schedule of train_noise_sac.py:
the buffer grows episode by episode (episode sizes from train_log.csv), and after each episode the learner does
utd x decisions updates, critic only during warm-up. No simulator: the data are the recorded transitions, so
the actor never acts (off-policy on fixed data, harsher than the online run).

Every --holdout_every-th episode is held out. Every --eval_every episodes, on held-out transitions so far:
- auc_q: AUC of min(Q1, Q2)(s, a) for the episode's success;
- auc_action: the same for Q(s, a) - mean_k Q(s, a_k), a_k drawn like the warm-up noise (the action's part).

    python -I replay_sac.py <task_7 dir> --label base [--utd 8] [--target td|mc] [--no_entropy_backup] ...
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "noise_reachability"))
import noise_sac  # noqa: E402
from noise_sac import ChunkReplay, NoiseSAC, NoiseSACConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("--label", required=True)
p.add_argument("--out", default=str(Path(__file__).parent / "results"))
p.add_argument("--episodes", type=int, default=300)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--utd", type=float, default=8)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--init_alpha", type=float, default=0.01)
p.add_argument("--fixed_alpha", type=float, default=None, help="no automatic temperature")
p.add_argument("--target_entropy", type=float, default=None)
p.add_argument("--clip_target", action="store_true", help="clip the TD target to [0, 1], the range of the return")
p.add_argument("--target", choices=["td", "mc"], default="td",
               help="td: bootstrapped soft target; mc: regress Q on the episode's own discounted return")
p.add_argument("--no_entropy_backup", action="store_true", help="td target without the -alpha log pi term")
p.add_argument("--critic_only", action="store_true", help="never update the actor (the MC critic does not use it)")
p.add_argument("--holdout_every", type=int, default=5, help="episodes with index %% this == this - 1 are held out")
p.add_argument("--eval_every", type=int, default=25)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--stop_q", type=float, default=1e4, help="stop once |Q mean| exceeds this")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(args.data)
full = ChunkReplay(200000, 968, 70)
full.load_parts(sorted(d.glob("replay_part_*.npz")))
rows = list(csv.DictReader(open(d / "train_log.csv")))
ends = np.cumsum([int(r["decisions"]) for r in rows])
assert ends[-1] == full.size

orig_td = noise_sac.td_target


def patched_td(reward, terminal, gamma, nq1, nq2, alpha, nlogp):
    t = orig_td(reward, terminal, gamma, nq1, nq2, 0.0 if args.no_entropy_backup else alpha, nlogp)
    return t.clamp(0.0, 1.0) if args.clip_target else t


noise_sac.td_target = patched_td

# Episode structure, Monte Carlo returns and the held-out split.
starts = np.r_[0, ends[:-1]]
success = np.array([int(r["success"]) for r in rows])
mc = np.zeros((full.size, 1), dtype=np.float32)
ep_of = np.zeros(full.size, dtype=np.int64)
for e, (a, b) in enumerate(zip(starts, ends)):
    ep_of[a:b] = e
    assert full.terminal[b - 1, 0] == success[e] and full.terminal[a:b - 1].sum() == 0
    if success[e]:
        mc[a:b, 0] = args.gamma ** np.arange(b - a - 1, -1, -1)
held = (ep_of % args.holdout_every) == args.holdout_every - 1

cfg = NoiseSACConfig(feature_dim=968, action_dim=70, init_alpha=args.fixed_alpha or args.init_alpha,
                     gamma=args.gamma, target_entropy=args.target_entropy)
learner = NoiseSAC(cfg, dev)
if args.fixed_alpha is not None:
    learner.alpha_opt = torch.optim.SGD([learner.log_alpha], lr=0.0)
gpu = {k: torch.from_numpy(getattr(full, k)[:full.size].astype(np.float32)).to(dev) for k in ChunkReplay.ARRAYS}
if args.target == "mc":  # target = reward = return, terminal = 1 so nothing is bootstrapped
    gpu["reward"] = torch.from_numpy(mc).to(dev)
    gpu["terminal"] = torch.ones_like(gpu["terminal"])
train_idx = torch.from_numpy(np.flatnonzero(~held)).to(dev)
n_train_upto = np.searchsorted(np.flatnonzero(~held), ends, side="left")  # train transitions before each episode end


def auc(score, label):
    """P(score of a positive > score of a negative), ties count half."""
    r = rankdata(score)
    n1 = label.sum(); n0 = len(label) - n1
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def evaluate(upto):
    idx = np.flatnonzero(held & (ep_of < upto))
    f = gpu["feats"][idx]; u = gpu["u"][idx]
    q = torch.min(*learner.critic(f, u)).squeeze(1)
    g = torch.Generator(device=dev).manual_seed(123)
    alt = torch.stack([torch.min(*learner.critic(f, (torch.randn(u.shape, device=dev, generator=g) / 2.5)
                                               .clamp(-0.999, 0.999))).squeeze(1) for _ in range(8)]).mean(0)
    lab = success[ep_of[idx]]
    return {"auc_q": auc(q.cpu().numpy(), lab), "auc_action": auc((q - alt).cpu().numpy(), lab),
            "q_held_succ": float(q[torch.from_numpy(lab == 1).to(dev)].mean()),
            "q_held_fail": float(q[torch.from_numpy(lab == 0).to(dev)].mean()), "n_held": int(len(idx))}


gen = torch.Generator(device=dev).manual_seed(args.seed)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log = []
t0, prev = time.time(), 0
for ep in range(args.episodes):
    n_new, prev = int(ends[ep] - prev), int(ends[ep])
    size = int(n_train_upto[ep])
    stats = []
    if size >= args.batch:
        for _ in range(int(round(args.utd * n_new))):
            idx = train_idx[torch.randint(0, size, (args.batch,), device=dev, generator=gen)]
            stats.append(learner.update({k: v[idx] for k, v in gpu.items()},
                                        update_actor=ep >= args.warmup and not args.critic_only))
    row = {"episode": ep, "replay": size}
    if (ep + 1) % args.eval_every == 0:
        row.update(evaluate(ep + 1))
    for k in ("q_mean", "critic_loss", "entropy", "alpha"):
        v = [s[k] for s in stats if k in s]
        row[k] = float(np.mean(v)) if v else None
    log.append(row)
    if (ep + 1) % args.eval_every == 0 or ep == args.episodes - 1:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              f"{time.time() - t0:.0f}s", flush=True)
    if row["q_mean"] is not None and abs(row["q_mean"]) > args.stop_q:
        print(f"stopped: |Q mean| {row['q_mean']:.3g} > {args.stop_q} at episode {ep}", flush=True)
        break
json.dump({"args": vars(args), "log": log}, open(out / f"{args.label}.json", "w"))
