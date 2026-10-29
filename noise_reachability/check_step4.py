#!/usr/bin/env python3
"""Step 4 checks on the real SmolVLA before the Kaggle run (README.md, step 4, "Checks before running").

1. Demonstration frames are presented as the policy expects: the base policy's flow-matching loss (fixed noise
   and times, self_filtered_bc's validation loss) on converted task-0 demonstrations is below DEMO_LOSS_MAX,
   and the same frames with both images rotated by 180 degrees give at least ROTATED_RATIO_MIN times that loss
   (so the check can see a wrong orientation). The loss on the base's own task-0 rollout frames is printed for
   reference only: rollout targets span several replans and are not comparable.
2. A freshly initialised LoRA adapter (arm A and C's settings) on the action expert gives the base policy's
   executed chunk bit for bit at --n_action_steps, on one observation of task 0 (init state 0, one reset), and
   a small change to its weights changes the chunk (the adapter is in the forward path).
   With --rollout_dir it also prints, for reference, the loss on the base's own rollout frames: on all frames,
   and on the frames where it replanned (step a multiple of n_action_steps) with only the first n_action_steps
   target actions counted, which are exactly the actions of the chunk it executed there.
3. With --b_policy_path (arm B, the continued-training checkpoint): it loads in this harness, queues
   n_action_steps - 1 actions after one inference on the same observation, and its chunk differs from the
   base's for the same seed (different weights are in use).

    python noise_reachability/check_step4.py --demo_dir <converted demos> --rollout_dir <collected episodes>

Exit code 1 if a check fails. Run from the repo root with the repo root on PYTHONPATH (FAIL-003).
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "self_filtered_bc"))

DEMO_LOSS_MAX = 0.3          # under 3x the checkpoint's loss on this data (0.112, continued_training at step 600)
ROTATED_RATIO_MIN = 2.0


class Rotated180:
    """A RolloutDataset with both camera images rotated by 180 degrees (the control of check 1)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import torch

        from rollout_data import IMAGE_KEYS

        s = dict(self.ds[i])
        for k in IMAGE_KEYS:
            s[k] = torch.flip(s[k], dims=(-2, -1))
        return s


class ReplanFrames:
    """Frames at steps that are multiples of ``n`` with the target masked after its first ``n`` actions."""

    def __init__(self, ds, n):
        self.ds, self.n = ds, n
        self.keep = [i for i, (e, f) in enumerate(ds.index) if int(ds.episodes[e]["frame_steps"][f]) % n == 0]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        s = dict(self.ds[self.keep[i]])
        pad = s["action_is_pad"].clone()
        pad[self.n:] = True
        s["action_is_pad"] = pad
        return s


def main(argv=None) -> int:
    import torch
    from lerobot.utils.constants import ACTION
    from torch.utils.data import DataLoader

    from rollout_data import RolloutDataset
    from self_filtered_bc import require_mujoco_patch, validation_loss
    from vla_core.envs import libero_env_config, make_libero_env, reset_to_init_state
    from vla_core.evaluation import seed_everything
    from vla_core.lora import LoraSettings, add_lora, resolve_targets
    from vla_core.policy import DEFAULT_RENAME_MAP, load_base_policy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo_dir", required=True, help="data dir of noise_reachability/demo_episodes.py")
    p.add_argument("--rollout_dir", default=None, help="data dir of self_filtered_bc.py --stage collect (optional)")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--policy_path", default="lerobot/smolvla_libero")
    p.add_argument("--n_action_steps", type=int, default=10)
    p.add_argument("--rename_map", type=json.loads, default=DEFAULT_RENAME_MAP)
    p.add_argument("--max_steps", type=int, default=520)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_targets", default="all")
    p.add_argument("--b_policy_path", default=None, help="arm B's checkpoint folder (optional)")
    args = p.parse_args(argv)
    require_mujoco_patch()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = libero_env_config(args.task_id, episode_length=args.max_steps, hard_reset=True)
    base = load_base_policy(args.policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                            rename_map=args.rename_map)
    policy = base.policy
    failures = []

    def check(name, ok, detail=""):
        print(f"[check] {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
        if not ok:
            failures.append(name)

    def loss(ds):
        loader = DataLoader(ds, batch_size=args.batch_size)
        return validation_loss(policy, base.preprocessor, loader, device, True, seed=0)

    demos = RolloutDataset(sorted((Path(args.demo_dir) / "episodes").glob("*.npz")), chunk_size=policy.config.chunk_size)
    demo_loss, rotated_loss = loss(demos), loss(Rotated180(demos))
    check("base loss on demonstration frames", demo_loss < DEMO_LOSS_MAX,
          f"{demo_loss:.4f} on {len(demos)} frames (limit {DEMO_LOSS_MAX})")
    check("rotated frames raise the loss", rotated_loss >= ROTATED_RATIO_MIN * demo_loss,
          f"{rotated_loss:.4f}, {rotated_loss / demo_loss:.1f}x (limit {ROTATED_RATIO_MIN}x)")
    if args.rollout_dir:
        rollouts = RolloutDataset(sorted((Path(args.rollout_dir) / "episodes").glob("*.npz")),
                                  chunk_size=policy.config.chunk_size)
        print(f"[check] reference: base loss on its own rollout frames {loss(rollouts):.4f} on {len(rollouts)} "
              "frames", flush=True)
        rr, rd = ReplanFrames(rollouts, args.n_action_steps), ReplanFrames(demos, args.n_action_steps)
        print(f"[check] reference: first {args.n_action_steps} target actions only, at replan frames: rollouts "
              f"{loss(rr):.4f} on {len(rr)} frames, demonstrations {loss(rd):.4f} on {len(rd)} frames", flush=True)

    env = make_libero_env(env_cfg)
    try:
        task = env.task_description
        obs, _ = reset_to_init_state(env, 0, seed=0)

        def chunk(seed=11, pol=base):
            seed_everything(seed)
            pol.reset()
            pol.act(obs, task)
            return torch.stack(list(pol.policy._queues[ACTION]))

        base_chunk = chunk()
        if args.b_policy_path:
            b = load_base_policy(args.b_policy_path, env_cfg, device=device, n_action_steps=args.n_action_steps,
                                 rename_map=args.rename_map)
            b_chunk = chunk(pol=b)
            check("arm B queues n_action_steps - 1 actions", len(b_chunk) == args.n_action_steps - 1,
                  f"{len(b_chunk)} (config n_action_steps {b.policy.config.n_action_steps})")
            diff = float((b_chunk - base_chunk).abs().max())
            check("arm B's chunk differs from the base's", diff > 1e-3, f"max |diff| {diff:.3g}")
            del b
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        expert = policy.model.vlm_with_expert
        settings = LoraSettings(r=args.lora_r, alpha=args.lora_alpha, target_modules=resolve_targets(args.lora_targets))
        expert.lm_expert = add_lora(expert.lm_expert, settings)
        lora_chunk = chunk()
        check(f"fresh LoRA reproduces the base chunk (n_action_steps={args.n_action_steps})",
              torch.equal(lora_chunk, base_chunk), f"max |diff| {float((lora_chunk - base_chunk).abs().max()):.3g}, "
              f"{len(base_chunk)} queued actions")
        with torch.no_grad():
            for name, prm in expert.lm_expert.named_parameters():
                if "lora_B" in name:
                    prm.add_(1e-2 * torch.randn_like(prm))
        diff = float((chunk() - base_chunk).abs().max())
        check("changed LoRA weights change the chunk", diff > 1e-4, f"max |diff| {diff:.3g}")
    finally:
        env.close()
    print("[check] all passed" if not failures else f"[check] FAILED: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
