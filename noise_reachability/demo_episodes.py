"""Step 4, arm C: one LIBERO-10 task's demonstrations as episodes in ``self_filtered_bc``'s format.

The demonstrations are those of ``lerobot/libero`` at the revision ``continued_training/`` used, the dataset
``lerobot/smolvla_libero`` was trained on. Each frame is taken from LeRobot's own dataset loader (the decoding
the checkpoint's training used) and stored with ``rollout_data.EpisodeRecorder``: the two images as uint8 (the
recorder refuses frames that are not exact uint8 values), the 8-dimensional state, and every action, so
``self_filtered_bc.py --stage train`` trains on demonstrations exactly as on rollouts (the next 50 actions of
a frame as its target). The dataset's task index is not LIBERO-10's task id (task 0 is the dataset's task 5),
so the task is found by its language string.

    python noise_reachability/demo_episodes.py --task_id 0 --root <dataset dir> --out_dir <data_dir>

Episode metadata carries ``source = "demonstration"``, the dataset episode index and ``demo_index`` (the
order among the task's demonstrations), which ``--val_demo_every`` in the train stage uses for the held-out
split. Demonstrations have no benchmark init state, seed or collecting policy (all stored as None).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "self_filtered_bc"))

from rollout_data import IMAGE_KEYS, STATE_KEY, EpisodeRecorder, save_episode  # noqa: E402

DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
LIBERO10_EPISODES = 379          # episodes 0-378 of the dataset are the ten LIBERO-10 tasks (continued_training/)
SOURCE = "demonstration"


def task_index_for(tasks: dict[str, int], language: str) -> int:
    """The dataset task index whose string is exactly ``language``."""
    hits = [i for t, i in tasks.items() if t == language]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} dataset tasks are {language!r}")
    return hits[0]


def episodes_of_task(task_of_episode: dict[int, int], task_index: int, n_episodes: int = LIBERO10_EPISODES) -> list[int]:
    """Dataset episodes below ``n_episodes`` whose frames all have ``task_index``, in index order."""
    return sorted(e for e, t in task_of_episode.items() if t == task_index and e < n_episodes)


def task_per_episode(episode_index, task_index) -> dict[int, int]:
    """{episode: task} from per-frame columns; refuses an episode whose frames name two tasks."""
    out = {}
    for e, t in zip(map(int, episode_index), map(int, task_index)):
        if out.setdefault(e, t) != t:
            raise ValueError(f"episode {e} has frames of tasks {out[e]} and {t}")
    return out


def record_demo(frames, frame_stride: int) -> EpisodeRecorder:
    """Record one episode's frames (dicts from LeRobot's dataset, in order) like a rollout.

    Checks that the frames are one episode of one task, with frame_index 0, 1, 2, ... in order.
    """
    recorder = EpisodeRecorder(frame_stride)
    episode = task = None
    for n, f in enumerate(frames):
        e, t, k = int(f["episode_index"]), int(f["task_index"]), int(f["frame_index"])
        episode, task = (e, t) if episode is None else (episode, task)
        if (e, t) != (episode, task):
            raise ValueError(f"frame {n} is from episode {e} (task {t}), expected episode {episode} (task {task})")
        if k != n:
            raise ValueError(f"episode {e}: frame {n} has frame_index {k}")
        recorder.add({key: f[key] for key in (*IMAGE_KEYS, STATE_KEY)}, f["action"])
    if len(recorder) == 0:
        raise ValueError("no frames")
    return recorder


def episode_groups(frames):
    """Yield (episode_index, frames) per run of consecutive frames of one episode, holding one episode at a time.
    An episode whose frames are not contiguous comes out twice and is refused by the caller's order check."""
    episode, buf = None, []
    for f in frames:
        e = int(f["episode_index"])
        if buf and e != episode:
            yield episode, buf
            buf = []
        episode = e
        buf.append(f)
    if buf:
        yield episode, buf


def demo_meta(task_id: int, language: str, episode_index: int, demo_index: int, n_steps: int,
              revision: str = DATASET_REVISION) -> dict:
    return {"task_id": task_id, "task": language, "source": SOURCE, "dataset": DATASET_REPO,
            "dataset_revision": revision, "dataset_episode_index": episode_index, "demo_index": demo_index,
            "success": True, "steps": n_steps, "init_state": None, "seed": None, "policy_path": None}


def provenance(settings: dict) -> dict:
    """Commit, dirty flags (whole default set, and this folder plus the episode format) and versions."""
    sys.path.insert(0, str(REPO_ROOT))
    from vla_core.provenance import collect, git_state

    record = collect(str(REPO_ROOT), settings, packages=("lerobot", "torch", "av", "huggingface_hub"))
    record["git_dirty_step4"] = git_state(str(REPO_ROOT), ("noise_reachability", "self_filtered_bc"))["git_dirty"]
    return record


def episode_file(demo_index: int, episode_index: int) -> str:
    return f"demo{demo_index:03d}_ep{episode_index:04d}.npz"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task_id", type=int, default=0, help="LIBERO-10 task id")
    p.add_argument("--out_dir", required=True, help="data dir; episodes go to <out_dir>/episodes/")
    p.add_argument("--frame_stride", type=int, default=2, help="store images every N frames (as the rollouts)")
    p.add_argument("--revision", default=DATASET_REVISION)
    p.add_argument("--root", required=True, help="local folder for the dataset (metadata and frame tables are "
                   "downloaded into it when missing; LeRobot adds the needed videos)")
    p.add_argument("--max_episodes", type=int, default=0, help="convert only the first N (smoke runs)")
    p.add_argument("--video_backend", default="pyav",
                   help="LeRobot video decoder, fixed so every machine decodes the same frames")
    args = p.parse_args(argv)

    import numpy as np
    import pandas as pd
    from huggingface_hub import snapshot_download
    from libero.libero import benchmark
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    language = benchmark.get_benchmark_dict()["libero_10"]().get_task(args.task_id).language
    root = Path(args.root)
    if not (root / "meta" / "info.json").exists():
        snapshot_download(DATASET_REPO, repo_type="dataset", revision=args.revision,
                          allow_patterns=["meta/**", "data/**"], local_dir=str(root))
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")["task_index"].to_dict()
    task_index = task_index_for(tasks, language)
    frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"])
                        for f in sorted((root / "data").glob("chunk-*/file-*.parquet"))])
    episodes = episodes_of_task(task_per_episode(frames["episode_index"], frames["task_index"]), task_index)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    print(f"task {args.task_id} ({language!r}) is dataset task {task_index}: {len(episodes)} episodes", flush=True)

    ds = LeRobotDataset(DATASET_REPO, root=root, revision=args.revision, episodes=episodes,
                        video_backend=args.video_backend)
    out = Path(args.out_dir)
    written = []
    groups = episode_groups(ds[i] for i in range(len(ds)))   # one episode's frames in memory at a time
    for demo_index, (e, frames_e) in enumerate(groups):
        if demo_index >= len(episodes) or e != episodes[demo_index]:
            raise ValueError(f"dataset episode {e} arrived as demonstration {demo_index}; expected {episodes}")
        recorder = record_demo(frames_e, args.frame_stride)
        meta = demo_meta(args.task_id, language, e, demo_index, len(recorder), args.revision)
        save_episode(out / "episodes" / episode_file(demo_index, e), recorder, meta)
        written.append({"file": episode_file(demo_index, e), "dataset_episode_index": e, "frames": len(recorder)})
        print(f"demo {demo_index}: dataset episode {e}, {len(recorder)} frames", flush=True)
    if len(written) != len(episodes):
        raise ValueError(f"{len(written)} of {len(episodes)} episodes in the loaded dataset")
    manifest = {"task_id": args.task_id, "task": language, "dataset": DATASET_REPO, "revision": args.revision,
                "task_index": task_index, "frame_stride": args.frame_stride, "video_backend": args.video_backend,
                "episodes": written, "total_frames": int(np.sum([w["frames"] for w in written])),
                "provenance": provenance(vars(args))}
    (out / "demo_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
