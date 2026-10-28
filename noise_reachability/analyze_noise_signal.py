"""Step 3b: does the executed noise carry a signal a critic can learn? (README.md, step 3b)

Reads collection streams written by ``train_noise_sac.py`` in collection mode (every episode a warm-up
episode, no updates): per stream folder ``replay_part_*.npz``, ``train_log.csv`` and ``provenance.jsonl``.
Episodes whose index is 4 mod 5 are held out. Critics are regressed on Monte Carlo returns of the other
episodes; on held-out transitions it reports

- action AUC: the noise's part of the value, Q(s, a) - mean_k Q(s, a_k), ranking transitions of successful
  episodes above failed ones, with a percentile bootstrap interval over held-out episodes;
- success AUC of Q(s, a) itself;
- the same action AUC for control critics trained with the noises shuffled across training transitions;

and the README's decision. Run from the repo root:

    python -I noise_reachability/analyze_noise_signal.py --roots <stream dir> [<stream dir> ...] --out <file>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HOLDOUT_EVERY = 5
MIN_AUC = 0.55            # hypothesis: action AUC at least this ...
CHANCE = 0.5              # ... and the interval's lower end strictly above chance
# README step 3b: the run settings every stream must have, and the stream seeds.
PROTOCOL = {"task_id": 7, "noise_scale": 5.0, "num_episodes": 250, "n_action_steps": 10, "train_init_states": "10-49",
            "max_steps": 520, "train_hard_reset": 0}
PROTOCOL_SEEDS = [0, 1, 2, 3]
PROTOCOL_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"   # lerobot/smolvla_libero snapshot in policy_path
STREAM_KEYS = ("task_id", "policy_path", "n_action_steps", "rename_map", "max_steps", "train_init_states",
               "num_episodes", "warmup_episodes", "updates_per_decision", "noise_scale", "gamma", "train_hard_reset")


def auc(score, label) -> float:
    """P(score of a positive > score of a negative), ties counting half."""
    from scipy.stats import rankdata

    score, label = np.asarray(score, dtype=np.float64), np.asarray(label).astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError(f"AUC needs both classes, got {n1} positive and {n0} negative")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mc_returns(decisions, success, gamma) -> np.ndarray:
    """Per transition, in episode order: gamma ** (decisions left after it) if its episode succeeded, else 0."""
    out = []
    for n, ok in zip(decisions, success):
        out.append(gamma ** np.arange(n - 1, -1, -1) if ok else np.zeros(n))
    return np.concatenate(out).astype(np.float32)[:, None]


def is_held_out(episode) -> np.ndarray:
    return np.asarray(episode) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1


def bootstrap_auc(score, label, groups, n_boot, seed, level=0.95):
    """Percentile interval of the AUC, resampling whole groups (episodes) with replacement."""
    score, label, groups = np.asarray(score), np.asarray(label), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        lab = label[pick]
        if lab.all() or not lab.any():
            continue  # a resample with one class has no AUC
        stats.append(auc(score[pick], lab))
    if len(stats) < 0.9 * n_boot:
        raise ValueError(f"only {len(stats)} of {n_boot} resamples had both classes")
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def excludes_chance(interval) -> bool:
    lo, hi = interval
    return lo > CHANCE or hi < CHANCE


def decide(action_auc, action_ci, control_ci) -> str:
    """README step 3b: 'not read' if the control is biased; 'supported' if AUC >= MIN_AUC and the lower end
    is strictly above chance; else 'not supported'."""
    if excludes_chance(control_ci):
        return "not read (control biased)"
    return "supported" if action_auc >= MIN_AUC and action_ci[0] > CHANCE else "not supported"


# --- data -------------------------------------------------------------------------------------------------


def _settings(stream: Path) -> dict:
    recs = [json.loads(line) for line in open(stream / "provenance.jsonl") if line.strip()]
    if not recs:
        raise ValueError(f"{stream}: no provenance records")
    first = {k: recs[0]["settings"].get(k) for k in STREAM_KEYS + ("seed",)}
    for r in recs[1:]:
        if {k: r["settings"].get(k) for k in first} != first:
            raise ValueError(f"{stream}: provenance records disagree on the run settings")
    if first["warmup_episodes"] != first["num_episodes"] or first["updates_per_decision"] != 0:
        raise ValueError(f"{stream}: not a collection run (warm-up {first['warmup_episodes']} of "
                         f"{first['num_episodes']} episodes, {first['updates_per_decision']} updates per decision)")
    first["git_commits"] = sorted({str(r.get("git_commit")) for r in recs})
    # Every git_dirty* flag (git_dirty, git_dirty_noise_reachability) must be false, as tools/validate_run.py.
    first["git_dirty"] = any(r.get("git_dirty") is not False or any(v is not False for k, v in r.items()
                                                                     if k.startswith("git_dirty")) for r in recs)
    return first


def _checkpoint_parts(stream: Path, num_episodes: int) -> list:
    """The replay parts checkpoint.pt lists, after checking that it saved every episode."""
    import torch

    ck = torch.load(stream / "checkpoint.pt", map_location="cpu", weights_only=True)
    if ck["episodes_done"] != num_episodes:
        raise ValueError(f"{stream}: checkpoint.pt saved {ck['episodes_done']} of {num_episodes} episodes")
    if not ck["replay_parts"]:
        raise ValueError(f"{stream}: checkpoint.pt lists no replay parts")
    return [stream / x for x in ck["replay_parts"]]


def load_stream(stream) -> dict:
    """One stream's transitions with episode ids, Monte Carlo returns and labels; refuses incomplete data."""
    from noise_sac import ChunkReplay

    stream = Path(stream)
    cfg = _settings(stream)
    rows = list(csv.DictReader(open(stream / "train_log.csv")))
    episodes = [int(r["episode"]) for r in rows]
    if episodes != list(range(cfg["num_episodes"])):
        raise ValueError(f"{stream}: train_log.csv has {len(rows)} episodes, not 0..{cfg['num_episodes'] - 1} in order")
    decisions = np.array([int(r["decisions"]) for r in rows])
    success = np.array([int(r["success"]) for r in rows])
    parts = _checkpoint_parts(stream, cfg["num_episodes"])
    with np.load(parts[0]) as z:
        first = (len(z["u"][0]), len(z["feats"][0]))
    rb = ChunkReplay(int(decisions.sum()), first[1], first[0])
    rb.load_parts(parts)  # checks that the parts are contiguous
    if rb.size != decisions.sum():
        raise ValueError(f"{stream}: replay holds {rb.size} transitions, train_log.csv {decisions.sum()} decisions")
    ep = np.repeat(np.arange(len(rows)), decisions)
    ends = np.cumsum(decisions) - 1
    if not np.array_equal(rb.terminal[ends, 0], success.astype(np.float32)) or \
            rb.terminal[: rb.size, 0].sum() != success.sum():
        raise ValueError(f"{stream}: terminal transitions do not match the logged successes")
    return {"cfg": cfg, "feats": rb.feats[: rb.size].astype(np.float32), "u": rb.u[: rb.size].copy(),
            "episode": ep, "success": success[ep], "ret": mc_returns(decisions, success, cfg["gamma"]),
            "n_episodes": len(rows), "n_success": int(success.sum())}


def load_streams(roots) -> dict:
    streams = [load_stream(r) for r in roots]
    shared = {k: streams[0]["cfg"][k] for k in STREAM_KEYS}
    for s, root in zip(streams, roots):
        if {k: s["cfg"][k] for k in STREAM_KEYS} != shared:
            raise ValueError(f"{root}: run settings differ from {roots[0]}")
    seeds = [s["cfg"]["seed"] for s in streams]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"streams share a seed: {seeds}")
    cat = {k: np.concatenate([s[k] for s in streams]) for k in ("feats", "u", "success", "ret")}
    cat["group"] = np.concatenate([s["cfg"]["seed"] * 100_000 + s["episode"] for s in streams])
    cat["held"] = np.concatenate([is_held_out(s["episode"]) for s in streams])
    cat["cfg"] = shared
    cat["seeds"] = sorted(seeds)
    cat["git_commits"] = sorted({c for s in streams for c in s["cfg"]["git_commits"]})
    cat["git_dirty"] = any(s["cfg"]["git_dirty"] for s in streams)
    cat["streams"] = {s["cfg"]["seed"]: (s["n_episodes"], s["n_success"]) for s in streams}
    return cat


def check_protocol(data) -> None:
    """Refuse data that is not README step 3b's: settings, stream seeds, checkpoint revision, one clean commit."""
    cfg = data["cfg"]
    wrong = {k: (cfg[k], v) for k, v in PROTOCOL.items() if cfg[k] != v}
    if wrong:
        raise ValueError(f"settings differ from the protocol (found, expected): {wrong}")
    if data["seeds"] != PROTOCOL_SEEDS:
        raise ValueError(f"stream seeds {data['seeds']}, the protocol has {PROTOCOL_SEEDS}")
    if PROTOCOL_REVISION not in str(cfg["policy_path"]):
        raise ValueError(f"policy_path {cfg['policy_path']} is not the snapshot {PROTOCOL_REVISION}")
    if len(data["git_commits"]) != 1 or data["git_commits"][0] == "None" or data["git_dirty"]:
        raise ValueError(f"need one clean commit in every provenance record, found {data['git_commits']} "
                         f"(dirty: {data['git_dirty']})")


# --- critics ----------------------------------------------------------------------------------------------


def fit_critic(feats, u, ret, seed, n_updates, hidden=256, layers=3, lr=3e-4, batch=256, device="cpu"):
    """Step 3's twin critic regressed by MSE on the returns; returns it in eval mode."""
    import torch

    from noise_sac import NoiseSACConfig, TwinQ

    torch.manual_seed(seed)
    critic = TwinQ(NoiseSACConfig(feature_dim=feats.shape[1], action_dim=u.shape[1], hidden=hidden,
                                  layers=layers)).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    f, a, r = (torch.as_tensor(x, device=device) for x in (feats, u, ret))
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(n_updates):
        idx = torch.randint(0, len(f), (batch,), device=device, generator=gen)
        q1, q2 = critic(f[idx], a[idx])
        loss = ((q1 - r[idx]) ** 2).mean() + ((q2 - r[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return critic.eval()


def scores(critics, feats, u, noise_scale, n_alt=8, seed=123, device="cpu", chunk=4096):
    """Mean over critics of min(Q1, Q2)(s, a), and of Q(s, a) - mean_k Q(s, a_k) with a_k drawn like the
    collected noise: N(0, 1) / noise_scale clipped to +-0.999."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    q_all, part_all = [], []
    with torch.no_grad():
        for start in range(0, len(feats), chunk):
            f = torch.as_tensor(feats[start:start + chunk], device=device)
            a = torch.as_tensor(u[start:start + chunk], device=device)
            alts = [(torch.randn(a.shape, device=device, generator=gen) / noise_scale).clamp(-0.999, 0.999)
                    for _ in range(n_alt)]
            q = torch.stack([torch.min(*c(f, a)).squeeze(1) for c in critics]).mean(0)
            alt = torch.stack([torch.stack([torch.min(*c(f, x)).squeeze(1) for x in alts]).mean(0)
                               for c in critics]).mean(0)
            q_all.append(q.cpu().numpy())
            part_all.append((q - alt).cpu().numpy())
    return np.concatenate(q_all), np.concatenate(part_all)


def analyse(data, n_critics=3, n_boot=2000, device="cpu", hidden=256, n_updates=None, control_seed=0):
    tr, te = ~data["held"], data["held"]
    if not te.any() or not tr.any():
        raise ValueError("need both training and held-out episodes")
    n_updates = int(tr.sum()) if n_updates is None else n_updates
    ns = data["cfg"]["noise_scale"]
    f_tr, u_tr, r_tr = data["feats"][tr], data["u"][tr], data["ret"][tr]
    u_shuf = u_tr[np.random.default_rng(control_seed).permutation(len(u_tr))]
    out = {"n_train_transitions": int(tr.sum()), "n_held_transitions": int(te.sum()),
           "n_held_episodes": int(len(np.unique(data["group"][te]))), "n_updates": n_updates}
    for name, u_fit in (("critic", u_tr), ("control", u_shuf)):
        critics = [fit_critic(f_tr, u_fit, r_tr, seed, n_updates, hidden=hidden, device=device)
                   for seed in range(n_critics)]
        q, part = scores(critics, data["feats"][te], data["u"][te], ns, device=device)
        lab, grp = data["success"][te], data["group"][te]
        out[name] = {"action_auc": auc(part, lab), "action_ci": bootstrap_auc(part, lab, grp, n_boot, seed=1),
                     "success_auc": auc(q, lab)}
    out["decision"] = decide(out["critic"]["action_auc"], out["critic"]["action_ci"], out["control"]["action_ci"])
    return out


def format_report(res, data) -> str:
    c, k = res["critic"], res["control"]
    streams = ", ".join(f"seed {s}: {ok}/{n}" for s, (n, ok) in sorted(data["streams"].items()))
    return "\n".join([
        "step 3b: noise signal on base rollouts (Monte Carlo critic, held-out episodes)",
        f"task {data['cfg']['task_id']}, successes per stream: {streams}",
        f"held out: {res['n_held_episodes']} episodes, {res['n_held_transitions']} transitions; "
        f"trained on {res['n_train_transitions']} transitions, {res['n_updates']} updates per critic",
        f"success AUC of Q(s, a): {c['success_auc']:.3f}",
        f"action AUC: {c['action_auc']:.3f} (95% interval {c['action_ci'][0]:.3f} to {c['action_ci'][1]:.3f})",
        f"control (noises shuffled): action AUC {k['action_auc']:.3f} "
        f"({k['action_ci'][0]:.3f} to {k['action_ci'][1]:.3f}), success AUC {k['success_auc']:.3f}",
        f"hypothesis (action AUC >= {MIN_AUC}, lower end > {CHANCE}): {res['decision']}",
    ]) + "\n"


def main(argv=None) -> int:
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True, help="collection stream folders (one per --seed)")
    p.add_argument("--out", required=True, help="report file; the numbers also go to <out>.json")
    p.add_argument("--n_critics", type=int, default=3)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    data = load_streams(args.roots)
    check_protocol(data)
    res = analyse(data, n_critics=args.n_critics, n_boot=args.n_boot, device=args.device)
    report = format_report(res, data)
    print(report, end="")
    Path(args.out).write_text(report)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repo root, for vla_core
    from vla_core.provenance import collect

    analysis = collect(str(Path(__file__).resolve().parents[1]), vars(args), packages=("torch", "numpy", "scipy"))
    Path(args.out + ".json").write_text(json.dumps({"args": vars(args), "settings": data["cfg"],
                                                    "streams": data["streams"], "collection_commit": data["git_commits"][0],
                                                    "analysis_provenance": analysis, **res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
