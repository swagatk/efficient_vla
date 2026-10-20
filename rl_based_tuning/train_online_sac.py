#!/usr/bin/env python3
"""Residual SAC on a frozen SmolVLA base policy, on one LIBERO-10 task, and its paired evaluation.

Training: episodes start from the benchmark init states in --train_init_states (default 10-49); the residual
``delta`` in [-1, 1]^7 is added to the base policy's un-normalised action, scaled per dimension to
--residual_scale_std times that dimension's action std (or to --residual_scale everywhere; env action units,
see vla_core/residual.py). The first --warmup_episodes run the base policy alone and train only the critic.

Diagnostics (--diagnostic base|noise): episodes on distinct training init states without learning, the base
policy alone or with the untrained head's exploration noise; see pilot 1 in week6/sac_residual_task6_pilot1/.

Evaluation (--eval): the base policy and/or the trained residual on --eval_init_states (default 0-9, the
init states of baseline_performance/results_300926) x --eval_seeds, with the same environment and noise seed
for both policies on each (seed, init state), hard resets, and the deterministic residual. Results are
appended to <eval_output_dir>/per_episode_<policy>.csv; a rerun skips finished episodes. At most
--max_episodes_per_process episodes run per call (exit code 3 means "call again"); see run_sac_residual.sh.

Run from the repo root with the repo root on PYTHONPATH (loads robosuite_mujoco_patch via sitecustomize,
FAIL-003): use run_sac_residual.sh. Before the October 2026 fix this script added the residual to
normalised actions, clamped the base action to +-1 std, and discarded the benchmark init states with a
second reset; see README.md in this folder.
"""
import argparse
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from vla_core.policy import DEFAULT_RENAME_MAP

REPO_ROOT = Path(__file__).resolve().parent.parent
EXIT_MORE_EPISODES = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_id", type=int, default=3, help="LIBERO-10 task index")
    p.add_argument("--seed", type=int, default=0, help="Training seed (Python, NumPy, PyTorch, init-state order)")
    # Default base policy: lerobot/smolvla_libero, 57.0% on LIBERO-10 through our evaluation path
    # (week6/smolvla_libero_lerobot_eval_kaggle/). It must run one action per inference and names its cameras
    # camera1/camera2. For the older HuggingFaceVLA/smolvla_libero pass --n_action_steps -1 --rename_map '{}'.
    p.add_argument("--policy_path", type=str, default="lerobot/smolvla_libero",
                   help="Frozen base policy (Hub id or local snapshot folder)")
    p.add_argument("--n_action_steps", type=int, default=1,
                   help="Override the checkpoint's n_action_steps; -1 keeps the checkpoint's value")
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP,
                   help="JSON env-key -> policy-key map; '{}' for checkpoints with LIBERO camera names")
    p.add_argument("--max_steps", type=int, default=520, help="Episode horizon")
    p.add_argument("--residual_scale_std", type=float, default=0.15,
                   help="Largest residual per action dimension as a fraction of that dimension's action std in the "
                        "base policy's training data (rotation actions are ~10x smaller than translation ones)")
    p.add_argument("--residual_scale", type=float, default=None,
                   help="Instead: the same largest residual for every dimension, in env action units (pilot 1 used 0.05)")
    p.add_argument("--warmup_episodes", type=int, default=5,
                   help="First training episodes run the base policy alone (zero residual) and update only the critic")
    p.add_argument("--residual_image_size", type=int, default=128, help="Image size seen by the residual head")
    p.add_argument("--train_init_states", type=str, default="10-49", help="Init states used for training")
    p.add_argument("--train_hard_reset", type=int, default=0,
                   help="1 = rebuild the simulator every training episode (slow; see vla_core/envs.py)")
    # Evaluation
    p.add_argument("--eval", action="store_true", help="Evaluate instead of training")
    p.add_argument("--eval_policies", type=str, default="base,residual", help="Comma-separated: base, residual")
    p.add_argument("--eval_init_states", type=str, default="0-9")
    p.add_argument("--eval_seeds", type=str, default="0,1,2")
    p.add_argument("--eval_output_dir", type=str, default=None, help="Default: <checkpoint_dir>/eval")
    p.add_argument("--max_episodes_per_process", type=int, default=10)
    # Checkpoints and logging
    p.add_argument("--checkpoint_path", type=str, default=None, help="Default: <checkpoint_dir>/latest_checkpoint.pt")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Default: checkpoints_task_<task_id>")
    p.add_argument("--checkpoint_interval", type=int, default=5, help="Save every N training episodes")
    p.add_argument("--resume_training", action="store_true")
    p.add_argument("--save_replay", type=int, default=1,
                   help="1 = save the replay buffer next to the checkpoint so a resumed run keeps its experience")
    p.add_argument("--time_limit_hours", type=float, default=None,
                   help="Start no new training episode after this many hours (for time-limited sessions)")
    p.add_argument("--wandb_run_id", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="efficient_vla_rl")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    # Base-policy check before training
    p.add_argument("--preflight_episodes", type=int, default=5)
    p.add_argument("--preflight_min_success", type=float, default=0.5)
    # SAC
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--auto_alpha", action="store_true")
    p.add_argument("--target_entropy", type=float, default=None)
    p.add_argument("--alpha_lr", type=float, default=1e-4)
    p.add_argument("--min_alpha", type=float, default=1e-5)
    p.add_argument("--max_alpha", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--buffer_size", type=int, default=20000,
                   help="Replay capacity (about 2 GB of RAM for 20,000 at 128 px)")
    p.add_argument("--updates_per_step", type=int, default=1)
    p.add_argument("--start_steps", type=int, default=0, help="Steps with uniformly random residuals first")
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--success_replay_ratio", type=float, default=0.2,
                   help="Fraction of each minibatch drawn from transitions of successful episodes")
    p.add_argument("--action_l2_penalty_coef", type=float, default=0.01)
    # Reward
    p.add_argument("--success_bonus", type=float, default=20.0)
    p.add_argument("--step_penalty", type=float, default=0.01)
    p.add_argument("--use_dense_shaping", action="store_true",
                   help="Potential-based shaping on the gripper's distance to objects named in the instruction")
    p.add_argument("--hard_task_preset", action="store_true", help="Preset for tasks the base rarely solves")
    # Diagnostics: run episodes on training init states in the training environment, without learning
    p.add_argument("--diagnostic", choices=["base", "noise"], default=None,
                   help="base: the base policy alone; noise: base + the untrained head's exploration noise. "
                        "Writes <checkpoint_dir>/diagnostic_<mode>.csv")
    p.add_argument("--diagnostic_episodes", type=int, default=20,
                   help="Distinct training init states to run (the same ones for both modes with the same --seed)")
    p.add_argument("--use_visual_prompting", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.use_visual_prompting:
        p.error("--use_visual_prompting is no longer supported (visual prompting degraded the base policy in weeks 2-3)")
    if args.hard_task_preset:
        args.residual_scale_std = max(args.residual_scale_std, 0.25)
        args.alpha, args.auto_alpha, args.use_dense_shaping = 0.005, True, True
        args.success_replay_ratio = 0.25
        args.preflight_min_success = min(args.preflight_min_success, 0.1)
    if args.n_action_steps is not None and args.n_action_steps < 0:
        args.n_action_steps = None
    args.rename_map = args.rename_map or None
    args.checkpoint_dir = args.checkpoint_dir or f"checkpoints_task_{args.task_id}"
    args.checkpoint_path = args.checkpoint_path or os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    args.eval_output_dir = args.eval_output_dir or os.path.join(args.checkpoint_dir, "eval")
    return args


def require_mujoco_patch():
    if "robosuite_mujoco_patch" not in sys.modules:
        raise SystemExit("robosuite_mujoco_patch is not loaded: run with the repo root on PYTHONPATH "
                         "(see run_sac_residual.sh; FAIL-003)")


class DistanceShaping:
    """Potential-based shaping, F = gamma * phi(s') - phi(s) with phi = -scale * (gripper distance to the
    nearest object named in the instruction). It changes no optimal policy (Ng et al., 1999)."""

    def __init__(self, sim_env, instruction: str, gamma: float, scale: float = 5.0):
        self.sim_env, self.gamma, self.scale = sim_env, gamma, scale
        names = instruction.lower()
        self.bodies = []
        for obj in _objects(sim_env):
            label = "".join(c for c in obj.name.lower().replace("_", " ") if not c.isdigit()).strip()
            if label and label in names:
                self.bodies.append(sim_env.sim.model.body_name2id(obj.root_body))
        if not self.bodies:
            raise ValueError(f"no object of the scene is named in the instruction {instruction!r}; "
                             "dense shaping would be zero, run without --use_dense_shaping")
        self.site = sim_env.sim.model.site_name2id("gripper0_grip_site")
        self.prev = self.potential()

    def potential(self) -> float:
        eef = self.sim_env.sim.data.site_xpos[self.site]
        return -self.scale * min(float(np.linalg.norm(eef - self.sim_env.sim.data.body_xpos[b])) for b in self.bodies)

    def step(self, terminal: bool) -> float:
        phi = 0.0 if terminal else self.potential()
        bonus = self.gamma * phi - self.prev
        self.prev = phi
        return bonus


def _objects(sim_env):
    objs = getattr(sim_env, "objects", None)
    if objs is None:
        objs = getattr(sim_env, "objects_dict", {})
    return list(objs.values()) if isinstance(objs, dict) else list(objs)


def normaliser_stats(pipeline, key):
    """(mean, std) of ``key`` from a LeRobot processor pipeline's (un)normaliser step, or None if not found."""
    for step in getattr(pipeline, "steps", []):
        stats = getattr(step, "stats", None) or {}
        if key in stats and {"mean", "std"} <= set(stats[key]):
            return (np.asarray(stats[key]["mean"], dtype=np.float32).reshape(-1),
                    np.asarray(stats[key]["std"], dtype=np.float32).reshape(-1))
    return None


def state_stats(base):
    """Mean and std of observation.state from the base policy's normaliser, or None if not found."""
    return normaliser_stats(base.preprocessor, "observation.state")


def max_residual_per_dim(args, action_std):
    """Largest residual per action dimension in env units: --residual_scale for every dimension if given,
    otherwise --residual_scale_std times the dimension's action std."""
    if args.residual_scale is not None:
        return np.full(len(action_std), args.residual_scale, dtype=np.float32)
    return (args.residual_scale_std * np.asarray(action_std, dtype=np.float32)).astype(np.float32)


def base_action_std(base):
    stats = normaliser_stats(base.postprocessor, "action")
    if stats is None:
        raise SystemExit("no action statistics in the base policy's postprocessor; pass --residual_scale instead")
    return stats[1]


def setup(args, hard_reset):
    """Environment, frozen base policy and device; shared by training and evaluation."""
    import torch

    from vla_core.envs import libero_env_config, make_libero_env
    from vla_core.policy import load_base_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=args.max_steps, hard_reset=hard_reset)
    env = make_libero_env(env_cfg)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    return env, base, device


def make_actor(base, task, head, device, residual_scale, image_size, deterministic, record=None,
               random_delta=lambda: False, zero_delta=lambda: False):
    """choose_action(obs) for vla_core.evaluation.run_episode. ``head=None`` runs the base policy alone.

    ``record``, if given, receives (image, state, base_action, delta) for every executed step. ``residual_scale``
    is a scalar or one value per action dimension (env units).
    """
    import torch

    from sac_residual import image_for_head
    from vla_core.policy import AGENTVIEW_KEY
    from vla_core.residual import apply_residual

    def choose(obs):
        base_action, env_obs = base.act(obs, task)
        if head is None:
            return base_action
        image = image_for_head(env_obs[AGENTVIEW_KEY], image_size)
        state = env_obs["observation.state"].detach().float().cpu()
        if zero_delta():
            delta = np.zeros_like(base_action)
        elif random_delta():
            delta = np.random.uniform(-1.0, 1.0, size=base_action.shape).astype(np.float32)
        else:
            with torch.no_grad():
                feats = head.features(image.to(device), state.to(device), torch.from_numpy(base_action)[None].to(device))
                delta = head.sample(feats, deterministic=deterministic)[0][0].float().cpu().numpy()
        if record is not None:
            record(image[0].numpy(), state[0].numpy(), base_action, delta)
        return apply_residual(base_action, delta, residual_scale)

    return choose


def next_inputs(base, task, obs, image_size):
    """Head inputs (image, state, base action) of an observation that no action is chosen for."""
    from sac_residual import image_for_head
    from vla_core.policy import AGENTVIEW_KEY

    base_action, env_obs = base.act(obs, task)
    return (image_for_head(env_obs[AGENTVIEW_KEY], image_size)[0].numpy(),
            env_obs["observation.state"].detach().float().cpu()[0].numpy(), base_action)


def evaluate(args) -> int:
    import torch

    sys.path.insert(0, str(REPO_ROOT / "baseline_performance"))
    from sac_residual import head_from_checkpoint
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode

    policies = [p.strip() for p in args.eval_policies.split(",") if p.strip()]
    if not policies or set(policies) - {"base", "residual"}:
        raise SystemExit(f"--eval_policies must name base and/or residual, got {args.eval_policies!r}")
    seeds, states = parse_index_ranges(args.eval_seeds), parse_index_ranges(args.eval_init_states)
    out = Path(args.eval_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = {p: load_episode_rows(out / f"per_episode_{p}.csv") for p in policies}
    jobs = [(p, s, i) for s in seeds for i in states for p in policies if (s, i) not in done[p]]
    print(f"[eval] {len(jobs)} episodes to run in {out}", flush=True)

    if jobs:
        ckpt = None
        if "residual" in policies:
            ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
            check_same_base(ckpt["run_config"], args)
        env, base, device = setup(args, hard_reset=True)
        head = head_from_checkpoint(ckpt["learner"], device) if ckpt else None
        rc = ckpt["run_config"] if ckpt else {}
        scale = np.asarray(rc.get("max_residual", rc.get("residual_scale")), dtype=np.float32) if ckpt else None
        image_size = ckpt["learner"]["head_config"]["image_size"] if ckpt else None
        try:
            for p, s, i in jobs[: args.max_episodes_per_process]:
                base.reset()
                choose = make_actor(base, env.task_description, head if p == "residual" else None, device,
                                    scale, image_size, deterministic=True)
                r = run_episode(env, i, s, choose, args.max_steps)
                append_episode_row(out / f"per_episode_{p}.csv", {"task_id": args.task_id, "seed": s, "init_state": i,
                                                                 "success": int(r["success"]), "steps": r["steps"]})
                print(f"[eval] {p} seed {s} init {i}: success={int(r['success'])} steps={r['steps']}", flush=True)
        finally:
            env.close()
        if len(jobs) > args.max_episodes_per_process:
            return EXIT_MORE_EPISODES

    done = {p: load_episode_rows(out / f"per_episode_{p}.csv") for p in policies}
    for p in policies:
        rows = [done[p][(s, i)] for s in seeds for i in states]
        n = sum(int(r["success"]) for r in rows)
        print(f"[eval] {p}: {n}/{len(rows)} = {100 * n / len(rows):.1f}%")
    if set(policies) == {"base", "residual"}:
        from compare_per_state import compare, format_report

        def eps(p):
            return {(args.task_id, s, i): int(done[p][(s, i)]["success"]) for s in seeds for i in states}

        report = format_report(compare(eps("base"), eps("residual")), "base", "residual")
        (out / "comparison_base_vs_residual.txt").write_text(report)
        print(report)
    return 0


def check_same_base(run_config, args):
    for key in ("policy_path", "n_action_steps", "rename_map", "task_id"):
        if run_config.get(key) != getattr(args, key):
            raise SystemExit(f"checkpoint was trained with {key}={run_config.get(key)!r}, evaluation has "
                             f"{getattr(args, key)!r}")


def train(args) -> int:
    import torch
    import wandb

    from sac_residual import HeadConfig, ReplayBuffer, SACConfig, SACLearner, SACResidualHead
    from vla_core.envs import parse_index_ranges, task_sim
    from vla_core.evaluation import run_episode, seed_everything
    from vla_core.provenance import append_jsonl, collect

    seed_everything(args.seed)
    train_states = parse_index_ranges(args.train_init_states)
    overlap = set(train_states) & set(parse_index_ranges(args.eval_init_states))
    if overlap:
        raise SystemExit(f"training and evaluation init states overlap: {sorted(overlap)}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_config = {k: getattr(args, k) for k in ("task_id", "seed", "policy_path", "n_action_steps", "rename_map",
                                                "residual_scale", "residual_scale_std", "warmup_episodes",
                                                "max_steps", "train_init_states")}

    env, base, device = setup(args, hard_reset=bool(args.train_hard_reset))
    task = env.task_description
    max_residual = max_residual_per_dim(args, base_action_std(base))
    run_config["max_residual"] = max_residual.tolist()
    print("[train] largest residual per action dimension (env units):", np.round(max_residual, 4).tolist())
    stats = state_stats(base)
    if stats is None:
        print("[warn] no observation.state statistics in the base preprocessor; the head sees raw state")
    head = SACResidualHead(HeadConfig(image_size=args.residual_image_size), *(stats or (None, None))).to(device)
    learner = SACLearner(head, SACConfig(gamma=args.gamma, tau=args.tau, alpha=args.alpha, auto_alpha=args.auto_alpha,
                                         target_entropy=args.target_entropy, min_alpha=args.min_alpha,
                                         max_alpha=args.max_alpha, action_l2_penalty_coef=args.action_l2_penalty_coef),
                         lr=args.learning_rate, alpha_lr=args.alpha_lr)
    buffer = ReplayBuffer(args.buffer_size, args.residual_image_size, seed=args.seed)
    replay_path = os.path.join(args.checkpoint_dir, "replay_buffer.npz")
    start_episode, global_step, history, run_id = 0, 0, [], args.wandb_run_id

    if args.resume_training and os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        check_same_base(ckpt["run_config"], args)
        learner.load_state_dict(ckpt["learner"])
        start_episode, global_step = ckpt["episode"] + 1, ckpt["global_step"]
        history, run_id = list(ckpt["success_history"]), run_id or ckpt.get("wandb_run_id")
        if os.path.exists(replay_path):
            buffer.load(replay_path)
        print(f"[train] resumed at episode {start_episode}, replay buffer {len(buffer)} transitions")
    else:
        rng = np.random.default_rng(args.seed)
        wins = 0
        for k in range(args.preflight_episodes):
            base.reset()
            r = run_episode(env, int(rng.choice(train_states)), 1_000_000 + k, make_actor(
                base, task, None, device, 0.0, 0, True), args.max_steps)
            wins += int(r["success"])
        rate = wins / max(1, args.preflight_episodes)
        print(f"[preflight] base success {wins}/{args.preflight_episodes} on training init states")
        if args.preflight_episodes and rate < args.preflight_min_success:
            raise SystemExit(f"preflight: base success {rate:.2f} < {args.preflight_min_success}; "
                             "the rollout stack or the task choice is wrong")

    append_jsonl(os.path.join(args.checkpoint_dir, "provenance.jsonl"), collect(str(REPO_ROOT), vars(args)))
    wandb.init(project=args.wandb_project, name=args.wandb_name or f"sac_residual_task{args.task_id}_seed{args.seed}",
               group=args.wandb_group or f"task_{args.task_id}", id=run_id or uuid.uuid4().hex[:8],
               resume="allow", config={**vars(args), "head_config": vars(head.cfg)})
    run_id = wandb.run.id

    stop = []

    def on_signal(sig, frame):
        if stop:  # second signal: exit now, without saving
            raise KeyboardInterrupt
        print("[signal] finishing the current episode, then saving; send again to exit now", flush=True)
        stop.append(True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def save(episode):
        if args.save_replay:
            buffer.save(replay_path)
        torch.save({"learner": learner.state_dict(), "run_config": run_config, "episode": episode,
                    "global_step": global_step, "success_history": history, "wandb_run_id": run_id},
                   args.checkpoint_path)

    rng = np.random.default_rng(args.seed * 7919 + start_episode)
    episode = start_episode - 1
    started, timed_out = time.monotonic(), False
    for episode in range(start_episode, args.num_episodes):
        if stop:
            break
        if args.time_limit_hours is not None and time.monotonic() - started > 3600 * args.time_limit_hours:
            print(f"[train] time limit of {args.time_limit_hours} h reached; stopping before episode {episode + 1}")
            timed_out = True
            break
        init_state = int(rng.choice(train_states))
        indices, losses, shaping, deltas = [], [], [], []
        warmup = episode < args.warmup_episodes
        pending = [None]  # (image, state, base_action, delta, reward, terminal) awaiting its next observation
        last = {}

        def record(image, state, base_action, delta):
            # The previous transition is complete once the inputs for the next observation are known.
            if pending[0] is not None:
                indices.append(buffer.add(*pending[0][:5], image, state, base_action, pending[0][5]))
                pending[0] = None
            last["inputs"] = (image, state, base_action, delta)
            deltas.append(np.asarray(delta, dtype=np.float32))

        def on_reset(obs):
            if args.use_dense_shaping:
                shaping.append(DistanceShaping(task_sim(env), task, args.gamma))

        def on_step(obs, action, reward, next_obs, terminated, info):
            nonlocal global_step
            success = bool(info.get("is_success", False))
            r = float(reward) + (args.success_bonus if success else -args.step_penalty)
            if shaping:
                r += shaping[0].step(terminal=success)
            pending[0] = (*last["inputs"], r, success)
            last["next_obs"] = next_obs
            global_step += 1
            if len(buffer) > args.batch_size and global_step > args.start_steps:
                for _ in range(args.updates_per_step):
                    losses.append(learner.update(buffer.sample(args.batch_size, args.success_replay_ratio, device),
                                                 update_actor=not warmup))

        base.reset()
        choose = make_actor(base, task, head, device, max_residual, args.residual_image_size,
                            deterministic=False, record=record, random_delta=lambda: global_step < args.start_steps,
                            zero_delta=lambda: warmup)
        result = run_episode(env, init_state, 1_000_000 + args.seed * 10_000 + episode, choose, args.max_steps,
                             on_step=on_step, on_reset=on_reset)
        if pending[0] is not None:
            image, state, base_action, delta, r, terminal = pending[0]
            if terminal:  # the next observation is never used for a terminal transition
                nxt = (image, state, base_action)
            else:  # time limit: bootstrap from the last observation
                nxt = next_inputs(base, task, last["next_obs"], args.residual_image_size)
            indices.append(buffer.add(image, state, base_action, delta, r, *nxt, terminal))
        if result["success"]:
            buffer.mark_success(indices)
        history.append(int(result["success"]))
        log = {"episode": episode, "init_state": init_state, "success": int(result["success"]), "steps": result["steps"],
               "success_mean20": float(np.mean(history[-20:])), "success_buffer_size": buffer.n_success()}
        log["warmup"] = int(warmup)
        if losses:
            for k in sorted({k for l in losses for k in l}):
                log[k] = float(np.mean([l[k] for l in losses if k in l]))
        if deltas:  # how far the residual moves each dimension, and how often tanh is saturated
            d = np.abs(np.stack(deltas))
            log.update({f"delta_abs_mean_{i}": float(v) for i, v in enumerate(d.mean(axis=0))})
            log["delta_saturated_frac"] = float((d > 0.95).mean())
        wandb.log(log)
        print(f"[train] ep {episode + 1}/{args.num_episodes} init {init_state}: success={log['success']} "
              f"steps={log['steps']} mean20={log['success_mean20']:.2f}" + (" (warm-up)" if warmup else "")
              + (f" |delta| {np.round(np.abs(np.stack(deltas)).mean(axis=0), 2).tolist()}" if deltas else ""), flush=True)
        if (episode + 1) % max(1, args.checkpoint_interval) == 0:
            save(episode)
    save(episode)
    env.close()
    wandb.finish(exit_code=130 if stop else 0)
    if stop:
        return 130
    return EXIT_MORE_EPISODES if timed_out else 0


def diagnose(args) -> int:
    """Episodes on distinct training init states in the training environment, without learning: the base policy
    alone, or with the untrained head's exploration noise at the configured residual size."""
    from sac_residual import HeadConfig, SACResidualHead
    from vla_core.envs import parse_index_ranges
    from vla_core.evaluation import append_episode_row, load_episode_rows, run_episode

    states = parse_index_ranges(args.train_init_states)
    chosen = [int(i) for i in np.random.default_rng(args.seed).permutation(states)[: args.diagnostic_episodes]]
    out = Path(args.checkpoint_dir) / f"diagnostic_{args.diagnostic}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_episode_rows(out)
    env, base, device = setup(args, hard_reset=bool(args.train_hard_reset))
    head, scale = None, None
    if args.diagnostic == "noise":
        scale = max_residual_per_dim(args, base_action_std(base))
        print("[diagnostic] largest residual per action dimension (env units):", np.round(scale, 4).tolist())
        head = SACResidualHead(HeadConfig(image_size=args.residual_image_size, pretrained_vision=False)).to(device)
    try:
        for i in chosen:
            if (args.seed, i) in done:
                continue
            base.reset()
            choose = make_actor(base, env.task_description, head, device, scale, args.residual_image_size,
                                deterministic=False)
            r = run_episode(env, i, 2_000_000 + args.seed, choose, args.max_steps)
            append_episode_row(out, {"task_id": args.task_id, "seed": args.seed, "init_state": i,
                                     "success": int(r["success"]), "steps": r["steps"]})
            print(f"[diagnostic] {args.diagnostic} init {i}: success={int(r['success'])} steps={r['steps']}", flush=True)
    finally:
        env.close()
    rows = load_episode_rows(out)
    n = sum(int(r["success"]) for r in rows.values())
    print(f"[diagnostic] {args.diagnostic}: {n}/{len(rows)} successes ({out})")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    require_mujoco_patch()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.eval:
        return evaluate(args)
    if args.diagnostic:
        return diagnose(args)
    try:
        from linux_inhibit import LinuxInhibit
    except ImportError:
        return train(args)
    with LinuxInhibit(reason="SAC residual training"):
        return train(args)


if __name__ == "__main__":
    sys.exit(main())
