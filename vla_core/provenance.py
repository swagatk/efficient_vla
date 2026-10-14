"""Run provenance: git commit, dirty flag and the versions that produced a result."""
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib import metadata

# Paths whose uncommitted changes can affect evaluation numbers.
DEFAULT_TRACKED_PATHS = ("baseline_performance", "hybrid_diffusion", "vla_core", "robosuite_mujoco_patch.py", "sitecustomize.py")


def git_state(repo_root: str, paths=DEFAULT_TRACKED_PATHS) -> dict:
    """Commit hash and whether tracked files under ``paths`` differ from it (untracked files ignored)."""
    def git(*args):
        return subprocess.run(["git", "-C", repo_root, *args], capture_output=True, text=True)

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"git_commit": None, "git_dirty": None}
    status = git("status", "--porcelain", "--untracked-files=no", "--", *paths)
    return {"git_commit": head.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def package_version(name: str) -> dict:
    """Installed version of ``name`` and, for a git install, the commit it was built from."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"version": None, "commit": None}
    commit = None
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    return {"version": dist.version, "commit": commit}


def gpu_names() -> list:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    except FileNotFoundError:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else []


def collect(repo_root: str, settings: dict, packages=("lerobot", "torch", "mujoco", "robosuite"),
            env_vars=("MUJOCO_GL", "PYOPENGL_PLATFORM")) -> dict:
    """One provenance record: time, host, git state, package versions, GPUs, render backend and run settings."""
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        **git_state(repo_root),
        "packages": {name: package_version(name) for name in packages},
        "gpus": gpu_names(),
        "env": {name: os.environ.get(name) for name in env_vars},
        "settings": settings,
    }


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
