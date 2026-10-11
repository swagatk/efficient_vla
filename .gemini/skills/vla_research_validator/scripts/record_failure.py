#!/usr/bin/env python3
"""
record_failure.py
Structured failure taxonomy and regression memory tracker for efficient_vla.

Usage:
  python record_failure.py --list
  python record_failure.py --query "segfault"
  python record_failure.py --record --title "..." --symptom "..." --root_cause "..." --resolution "..." --invariant "..."
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_FILE = SKILL_DIR / "failures_taxonomy.json"
FAILURES_MD = SKILL_DIR / "FAILURES.md"

DEFAULT_FAILURES = [
    {
        "id": "FAIL-001",
        "date": "2026-10-06",
        "title": "VLM Semantic Conditioning Corruption (0% Success)",
        "symptom": "Evaluation success rate on LIBERO Task 0 suddenly dropped from 20% to 0% across all 10 episodes.",
        "root_cause": "A dummy expert token (1 token, dim 480) was injected into `inputs_embeds = [prefix_embs, dummy_expert_embs]` with `use_cache=False`. Inside SmolVLMWithExpertModel, `fill_kv_cache=False` forced `forward_cross_attn_layer` instead of `forward_attn_layer`. All visual and state prefix tokens cross-attended against zeroes across all layers, completely distorting the conditioning vector passed to the diffusion head.",
        "resolution": "Removed dummy_expert_embs completely. Restored `inputs_embeds=[prefix_embs, None]` with `use_cache=True` and `past_key_values=None` so `fill_kv_cache=True` runs clean prefix self-attention, matching the trained checkpoint. Delete past_key_values immediately after forward pass to prevent VRAM accumulation.",
        "invariant": "Never pass non-None expert tokens to the frozen VLM backbone. Always use `inputs_embeds=[prefix_embs, None]` and clean self-attention prefill.",
        "tags": ["conditioning", "vlm", "diffusion", "performance_drop"]
    },
    {
        "id": "FAIL-002",
        "date": "2026-10-06",
        "title": "PyTorch Submodule Traversal Recursion Segfault (Exit 139)",
        "symptom": "Evaluation crashed with `Segmentation fault (core dumped)` in `named_children` / `train` / `eval` on step 0 of episode 1.",
        "root_cause": "SmolVLAPolicy.select_action(batch) calls `self.eval()` internally on every step. Calling `.eval()` traverses all submodules in SmolVLMWithExpertModel recursively. In PyTorch with deep multimodal hierarchies, calling this 520 times per episode overflows the C-level call stack.",
        "resolution": "Freeze base_policy in eval mode at init: `self.base_policy.eval = lambda *args, **kwargs: self.base_policy`. This neutralizes repeated traversals while keeping the model deterministic.",
        "invariant": "Always override `self.base_policy.eval` with a lambda no-op after initial `self.base_policy.eval()` call in wrapper classes.",
        "tags": ["segfault", "eval", "pytorch", "recursion"]
    },
    {
        "id": "FAIL-003",
        "date": "2026-10-06",
        "title": "Missing robosuite_mujoco_patch in PYTHONPATH",
        "symptom": "Robosuite / MuJoCo 3.x memory corruption and joint property assertions during environment resets.",
        "root_cause": "eval_hybrid_diffusion.sh only added `hybrid_diffusion/` to PYTHONPATH, omitting the repository root. `sitecustomize.py` and `robosuite_mujoco_patch.py` were never loaded.",
        "resolution": "Added `REPO_ROOT` to PYTHONPATH in `eval_hybrid_diffusion.sh` and added fallback explicit import in `train_hybrid_diffusion.py`.",
        "invariant": "All runner scripts and Python entry points must ensure repository root is in PYTHONPATH so `sitecustomize.py` loads `robosuite_mujoco_patch`.",
        "tags": ["mujoco", "robosuite", "pythonpath", "driver"]
    },
    {
        "id": "FAIL-004",
        "date": "2026-10-06",
        "title": "Headless EGL Resource Accumulation Segfault on 5th Episode Reset",
        "symptom": "Evaluation crashed with `Segmentation fault` inside `robosuite/utils/mjcf_utils.py` `find_elements` on Episode 5 reset.",
        "root_cause": "Under headless EGL in WSL2, Robosuite accumulates internal C-level XML ElementTree handles and OpenGL contexts across resets. Running 5 consecutive resets in a single Python process exceeds the memory threshold.",
        "resolution": "Changed default batch size in `eval_hybrid_diffusion.sh` to 3 episodes per process. Each sub-process runs at most 3 episodes, exiting cleanly and recycling the driver context.",
        "invariant": "Never run more than 3 consecutive environment resets per Python process under headless EGL.",
        "tags": ["egl", "batch_size", "segfault", "xml", "robosuite"]
    },
    {
        "id": "FAIL-005",
        "date": "2026-10-04",
        "title": "Open-Loop Residual Execution Collapse (replan_each_step=False)",
        "symptom": "Residual evaluation dropped to 0.0% when replan_each_step was False.",
        "root_cause": "When `eval_replan_each_step=False`, the model predicts a 16-step residual chunk and executes it open-loop. In a contact-rich 520-step manipulation task, compounding drift without closed-loop replanning causes failure.",
        "resolution": "Set `--eval_replan_each_step 1` by default for closed-loop residual adjustment on every step.",
        "invariant": "Residual evaluations on LIBERO must use `--eval_replan_each_step 1` unless specifically testing open-loop horizon stability.",
        "tags": ["replan_each_step", "closed_loop", "drift"]
    }
]


def load_taxonomy():
    if not TAXONOMY_FILE.exists():
        TAXONOMY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TAXONOMY_FILE, "w") as f:
            json.dump(DEFAULT_FAILURES, f, indent=2)
        return DEFAULT_FAILURES
    try:
        with open(TAXONOMY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {TAXONOMY_FILE}: {e}", file=sys.stderr)
        return DEFAULT_FAILURES


def save_taxonomy(data):
    with open(TAXONOMY_FILE, "w") as f:
        json.dump(data, f, indent=2)
    export_markdown(data)


def export_markdown(data):
    with open(FAILURES_MD, "w") as f:
        f.write("# VLA Research Failure Taxonomy & Regression Guard\n\n")
        f.write("This file is automatically updated by `record_failure.py`. Before modifying agent code or debugging performance drops, check these documented failure modes to avoid repeating past mistakes.\n\n")
        f.write("| ID | Date | Title | Key Symptom | Primary Invariant |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for item in data:
            f.write(f"| **{item['id']}** | {item['date']} | {item['title']} | {item['symptom'][:60]}... | {item['invariant'][:60]}... |\n")
        f.write("\n---\n\n")
        for item in data:
            f.write(f"### [{item['id']}] {item['title']}\n\n")
            f.write(f"- **Date**: {item['date']}\n")
            f.write(f"- **Symptom**: {item['symptom']}\n")
            f.write(f"- **Root Cause**: {item['root_cause']}\n")
            f.write(f"- **Resolution**: {item['resolution']}\n")
            f.write(f"- **Mandatory Invariant**: `{item['invariant']}`\n\n")


def list_failures(data):
    print(f"\n{'='*75}")
    print(f"📋 Documented Failure Modes ({len(data)} total)")
    print(f"{'='*75}")
    for item in data:
        print(f"[{item['id']}] {item['title']} ({item['date']})")
        print(f"  Symptom   : {item['symptom']}")
        print(f"  Invariant : {item['invariant']}")
        print(f"  Tags      : {', '.join(item.get('tags', []))}\n")


def query_failures(data, term):
    term = term.lower()
    matches = [
        item for item in data
        if term in item['title'].lower()
        or term in item['symptom'].lower()
        or term in item['root_cause'].lower()
        or any(term in t.lower() for t in item.get('tags', []))
    ]
    print(f"\n🔍 Query '{term}' matched {len(matches)} failure record(s):")
    for item in matches:
        print(f"\n--- [{item['id']}] {item['title']} ---")
        print(f"  Symptom    : {item['symptom']}")
        print(f"  Root Cause : {item['root_cause']}")
        print(f"  Resolution : {item['resolution']}")
        print(f"  Invariant  : {item['invariant']}")


def main():
    parser = argparse.ArgumentParser(description="Failure Taxonomy and Regression Tracker")
    parser.add_argument("--list", action="store_true", help="List all known failure modes")
    parser.add_argument("--query", type=str, help="Search failure records by keyword")
    parser.add_argument("--record", action="store_true", help="Record a new failure mode")
    parser.add_argument("--title", type=str, help="Title of the failure")
    parser.add_argument("--symptom", type=str, help="Observed symptom")
    parser.add_argument("--root_cause", type=str, help="Diagnosed root cause")
    parser.add_argument("--resolution", type=str, help="Applied fix")
    parser.add_argument("--invariant", type=str, help="Invariant rule to prevent recurrence")
    parser.add_argument("--tags", type=str, help="Comma-separated tags")

    args = parser.parse_args()
    data = load_taxonomy()

    if args.list or len(sys.argv) == 1:
        list_failures(data)
    elif args.query:
        query_failures(data, args.query)
    elif args.record:
        if not (args.title and args.symptom and args.root_cause and args.resolution and args.invariant):
            print("Error: --record requires --title, --symptom, --root_cause, --resolution, and --invariant", file=sys.stderr)
            sys.exit(1)
        new_id = f"FAIL-{len(data)+1:03d}"
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
        new_entry = {
            "id": new_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "title": args.title,
            "symptom": args.symptom,
            "root_cause": args.root_cause,
            "resolution": args.resolution,
            "invariant": args.invariant,
            "tags": tags
        }
        data.append(new_entry)
        save_taxonomy(data)
        print(f"✅ Recorded failure {new_id}: '{args.title}'")


if __name__ == "__main__":
    main()
