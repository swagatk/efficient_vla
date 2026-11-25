#!/usr/bin/env python3
"""Step 18 analysis (mechanism M1, README "## Step 18"): the decision rule, the run's harness and its report. Standard
library only.

    python3 -S noise_reachability/analyze_step18.py --root <run folder> --gate harness [--out harness.txt]  # exit 1 stops
    python3 -S noise_reachability/analyze_step18.py --root <run folder> --out step18_report.txt [--table table.csv]

<run folder> holds
  harness/session<k>/task<t>_state<s>/   two committed `repeat` episodes of each state, rerun as ar_repeat rho 1 at
                                         the start of every session k;
  task_<t>/<condition>/state<s>_block<b>/  the base with ``<condition>`` (noise_processes.family_label) on ids block b
                                         of the state, on every state whose harness reproduced in session 0.

Conditions are named by ``noise_processes.family_label``: ``repeat`` (rho = 1, constant), ``ar_repeat_rho0.9``,
``scaled_sigma0.5``, and the reported ones ``ar_repeat_rho0.5``, ``ar_repeat_rho0``, ``scaled_sigma1`` and ``zero``.

- *Tests* (TESTED): pooled over the states, `repeat` above the condition, one-sided Fisher exact on equal episode
  counts, Holm over the two at 0.05.
- *Keeps half* (``ratio_interval``): with rare successes and equal episodes per state, the condition's share of the
  pair's pooled successes is Binomial(total, theta / (1 + theta)) given the total, theta = rate(condition) /
  rate(repeat) on every state. The two-sided 95% Clopper-Pearson interval on that share gives one on theta; the
  condition "keeps half" when its lower end is above KEEP (0.5). Approximate: exact for Poisson counts.
- *Gate*: `repeat` must succeed at least MIN_REFERENCE times pooled, or nothing is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_step5a import read_rows  # noqa: E402
from analyze_step7 import clopper_pearson, fisher_greater  # noqa: E402
from analyze_step8 import holm  # noqa: E402
from analyze_step10 import mcnemar_greater  # noqa: E402
from analyze_step12 import _rows  # noqa: E402
from analyze_step15 import ids_spec  # noqa: E402
from analyze_step16 import check_provenance  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
W6 = REPO_ROOT / "week6"
EPISODE_SEED_BASE = 1000                     # deep_noise_search: noise id k runs with episode seed 1000 + k
# calibration/step18_states.py's choice, in its order; i is the state's index for its ids
STATES = ((0, 8), (0, 6), (4, 26), (4, 11), (6, 12), (6, 8), (7, 33), (7, 47))
ID_BASE, ID_STRIDE = 20000, 200              # fresh: earlier steps used ids up to 10049
BLOCK = 10                                   # ids per process (deep_noise_search's --max_episodes_per_process)
N_TESTED, N_REPORTED, N_ZERO = 100, 20, 2    # ids per state: repeat and the tested conditions, reported ones, zero
# label: (family, rho, sigma, ids per state)
CONDITIONS = {"repeat": ("repeat", None, None, N_TESTED),
              "ar_repeat_rho0.9": ("ar_repeat", 0.9, None, N_TESTED),
              "scaled_sigma0.5": ("scaled", None, 0.5, N_TESTED),
              "ar_repeat_rho0.5": ("ar_repeat", 0.5, None, N_REPORTED),
              "ar_repeat_rho0": ("ar_repeat", 0.0, None, N_REPORTED),
              "scaled_sigma1": ("scaled", None, 1.0, N_REPORTED),
              "zero": ("zero", None, None, N_ZERO)}
HARNESS_LABEL = "ar_repeat_rho1"             # the new code path, bit-identical to repeat (check_noise_processes.py)
MIN_HARNESS_STATES = 7
STEP16_TABLE = W6 / "noise_reachability_step16_kaggle" / "run" / "step16_table_parts5.csv"
STEP8_733 = W6 / "noise_reachability_step8_kaggle" / "run" / "task_7" / "test" / "per_episode_repeat_state33.csv"
STEP9_747 = W6 / "noise_reachability_step9_kaggle" / "run" / "task_7" / "test" / "random" / "per_episode_repeat_state47.csv"
SHA256 = {STEP16_TABLE: "ed0b3e32ee140d639b57415b3c690230a8c84bb96939563a11eeb4d42c1e036b",
          STEP8_733: "4db6955e8f9d7089aeaaf259f41487af057f57b8b12a2779757cab007bb56dd8",
          STEP9_747: "a435e82b7c4e7e9ed2a911c2f02698c0475103b70664fcd9fdda2e9e0ea732d7"}
# (task, state): (committed success id, committed failure id, source)
HARNESS = {(0, 8): (10000, 10001, STEP16_TABLE), (0, 6): (10009, 10000, STEP16_TABLE),
           (4, 26): (10033, 10000, STEP16_TABLE), (4, 11): (10010, 10000, STEP16_TABLE),
           (6, 12): (10018, 10000, STEP16_TABLE), (6, 8): (10037, 10000, STEP16_TABLE),
           (7, 33): (1459, 1450, STEP8_733), (7, 47): (6007, 6000, STEP9_747)}

REFERENCE = "repeat"
TESTED = ("scaled_sigma0.5", "ar_repeat_rho0.9")
REPORTED = ("ar_repeat_rho0.5", "ar_repeat_rho0", "scaled_sigma1", "zero")
RHO_TREND = ("ar_repeat_rho0", "ar_repeat_rho0.5", "ar_repeat_rho0.9", "repeat")   # reported on the first N_REPORTED ids
KEEP = 0.5
MIN_REFERENCE = 10


def ratio_interval(x_cond: int, x_ref: int) -> tuple[float, float]:
    """Two-sided 95% interval on rate(condition) / rate(reference) from pooled successes on equal episode counts."""
    total = x_cond + x_ref
    if total == 0:
        return 0.0, float("inf")
    lo, hi = clopper_pearson(x_cond, total)
    return lo / (1 - lo), (float("inf") if hi >= 1 else hi / (1 - hi))


def condition_reading(below: bool, keeps_half: bool) -> str:
    if below and keeps_half:
        return "below constant noise, but keeps more than half its rate"
    if below:
        return "below constant noise"
    if keeps_half:
        return "matches constant noise within half its rate"
    return "not resolved"


def overall_reading(results: dict) -> str:
    """The step's reading from the two tested conditions (see the README's "Reading")."""
    s, r = results["scaled_sigma0.5"], results["ar_repeat_rho0.9"]
    if s["keeps_half"] and not s["below"]:
        return ("lower diversity: sampling at half the temperature keeps at least half of constant noise's rate, so "
                "'constancy' becomes 'lower noise diversity'")
    if s["keeps_half"]:
        return ("partly lower diversity: sampling at half the temperature succeeds less than constant noise but keeps "
                "more than half its rate")
    if not s["below"]:
        return "not resolved: lower-temperature sampling is neither shown below constant noise nor shown to keep half"
    if r["below"] and not r["keeps_half"]:
        return "constancy: both lower-temperature sampling and a slowly drifting noise succeed less than a constant noise"
    if r["keeps_half"]:
        return "persistence: lower-temperature sampling succeeds less, but a slowly drifting noise keeps half the rate"
    return "not lower diversity: lower-temperature sampling succeeds less; the drifting noise is not resolved"


def rule(pooled: dict, n: int) -> dict:
    """pooled: {condition: successes over the states}, each condition with n episodes in all (equal per state)."""
    x_ref = pooled[REFERENCE]
    if x_ref < MIN_REFERENCE:
        return {"read": False, "reading": f"not read: constant noise succeeded {x_ref} times (< {MIN_REFERENCE})"}
    ps = {c: fisher_greater(x_ref, pooled[c], n) for c in TESTED}
    below = holm(ps)
    out = {"read": True, "p": ps, "conditions": {}}
    for c in TESTED + tuple(c for c in REPORTED if c in pooled):
        lo, hi = ratio_interval(pooled[c], x_ref)
        out["conditions"][c] = {"successes": pooled[c], "ratio_interval": (lo, hi), "keeps_half": lo > KEEP,
                                "below": below.get(c)}                  # None for a reported condition: not tested
        if c in TESTED:
            out["conditions"][c]["reading"] = condition_reading(below[c], lo > KEEP)
    out["reading"] = overall_reading(out["conditions"])
    return out


# ---- the protocol's ids and folders -------------------------------------------------------------------------------

def state_index(task: int, state: int) -> int:
    return STATES.index((task, state))


def condition_ids(task: int, state: int, label: str) -> list[int]:
    i = state_index(task, state)
    return [ID_BASE + ID_STRIDE * i + j for j in range(CONDITIONS[label][3])]


def blocks(task: int, state: int, label: str) -> list[list[int]]:
    ids = condition_ids(task, state, label)
    return [ids[b:b + BLOCK] for b in range(0, len(ids), BLOCK)]


def block_dir(root, task: int, state: int, label: str, b: int) -> Path:
    """One folder per state, condition and block, so each provenance record and file belong to one process."""
    return Path(root, f"task_{task}", label, f"state{state}_block{b}")


def block_file(root, task: int, state: int, label: str, b: int) -> Path:
    return block_dir(root, task, state, label, b) / f"per_episode_{label}_state{state}.csv"


def harness_dir(root, session: int, task: int, state: int) -> Path:
    """Every session reruns the harness in its own folder (session 0 is the first)."""
    return Path(root, "harness", f"session{session}", f"task{task}_state{state}")


def harness_file(root, session: int, task: int, state: int) -> Path:
    return harness_dir(root, session, task, state) / f"per_episode_{HARNESS_LABEL}_state{state}.csv"


def sessions(root) -> list[int]:
    """The harness sessions present, which must be 0, 1, ... without a gap."""
    d = Path(root, "harness")
    found = sorted(int(p.name.removeprefix("session")) for p in d.glob("session*")) if d.exists() else []
    if found != list(range(len(found))):
        raise ValueError(f"harness sessions {found}: expected 0-{len(found) - 1}")
    return found


def harness_ids(task: int, state: int) -> list[int]:
    return sorted(HARNESS[(task, state)][:2])


# ---- provenance ---------------------------------------------------------------------------------------------------

def _require(cond: bool, f, what: str) -> None:
    if not cond:
        raise ValueError(f"provenance: {what} in {f}")


def _family(f, st, family, rho, sigma) -> None:
    _require(st.get("family") == family and st.get("rho") == rho and st.get("sigma") == sigma, f,
             f"family {st.get('family')} rho {st.get('rho')} sigma {st.get('sigma')}")
    _require(all(st.get(k) is None for k in ("jitter_seeds", "record_dir", "lora_dir", "feature_dir")), f,
             "not a plain search of the base")


def check_settings(root, f: Path, r: dict) -> None:
    """The protocol's settings for the folder ``f`` is in; refuses a folder the protocol does not have."""
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    harness_names = {f"task{t}_state{s}": (t, s) for t, s in STATES}
    if len(rel) == 3 and rel[0] == "harness" and rel[1].removeprefix("session").isdigit() and rel[2] in harness_names:
        t, s = harness_names[rel[2]]
        _family(f, st, "ar_repeat", 1.0, None)
        _require(st["task_id"] == t and st["init_state"] == s, f, f"task {st['task_id']} state {st['init_state']}")
        _require(st["noise_ids"] == ids_spec(harness_ids(t, s)), f, f"harness ids {st['noise_ids']}")
        return
    tasks = {f"task_{t}": t for t in {t for t, _ in STATES}}
    folders = {f"state{s}_block{b}": (s, b) for t, s in STATES for c in CONDITIONS for b in range(len(blocks(t, s, c)))}
    if len(rel) == 3 and rel[0] in tasks and rel[1] in CONDITIONS and rel[2] in folders:
        t, label = tasks[rel[0]], rel[1]
        s, b = folders[rel[2]]
        _family(f, st, *CONDITIONS[label][:3])
        _require(st["task_id"] == t and st["init_state"] == s and (t, s) in STATES, f,
                 f"task {st['task_id']} state {st['init_state']}")
        bl = blocks(t, s, label)
        _require(0 <= b < len(bl) and st["noise_ids"] == ids_spec(bl[b]), f, f"block {b} ids {st['noise_ids']}")
        return
    raise ValueError(f"provenance: an unexpected folder {f.parent}")


# ---- harness ------------------------------------------------------------------------------------------------------

def _checked(path: Path) -> Path:
    if path in SHA256:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != SHA256[path]:
            raise ValueError(f"{path}: sha256 {got} is not the committed file's")
    return path


def committed_harness() -> dict:
    """{(task, state): {id: (success, steps)}} of the committed episodes the harness reruns (sha256-checked)."""
    table = {}
    with open(_checked(STEP16_TABLE)) as f:
        for r in csv.DictReader(f):
            table[(int(r["task_id"]), int(r["init_state"]), int(r["noise_id"]))] = (int(r["success"]), int(r["steps"]))
    out = {}
    for (t, s), (good, bad, src) in HARNESS.items():
        if src == STEP16_TABLE:
            rows = {k: table[(t, s, k)] for k in (good, bad)}
        else:
            old = _rows(_checked(src))
            rows = {k: old[(EPISODE_SEED_BASE + k, s)] for k in (good, bad)}
        if rows[good][0] != 1 or rows[bad][0] != 0:
            raise ValueError(f"harness {t}/{s}: committed rows {rows} are not one success and one failure")
        out[(t, s)] = rows
    return out


def session_harness(root, session: int, old: dict) -> dict:
    """{"t/s": {"same", "run", "committed"}} of one session; refuses a missing episode."""
    per = {}
    for t, s in STATES:
        path = harness_file(root, session, t, s)
        new = {seed - EPISODE_SEED_BASE: v for (seed, st), v in (_rows(path) if path.exists() else {}).items()
               if st == s}
        if set(new) != set(old[(t, s)]):
            raise ValueError(f"harness session {session} {t}/{s}: has ids {sorted(new)}, expected "
                             f"{sorted(old[(t, s)])}; finish the run")
        per[f"{t}/{s}"] = {"same": all(new[k] == old[(t, s)][k] for k in new),
                           "run": {str(k): list(v) for k, v in sorted(new.items())},
                           "committed": {str(k): list(v) for k, v in sorted(old[(t, s)].items())}}
    return per


def harness(root) -> dict:
    """The states whose two committed episodes reproduce exactly (success and steps) in the first session are kept;
    every later session must reproduce every kept state again, or the run is stopped (the stack changed)."""
    old = committed_harness()
    found = sessions(root)
    if not found:
        raise ValueError("harness: no session has run; finish the run")
    per = [session_harness(root, k, old) for k in found]
    kept = [[t, s] for t, s in STATES if per[0][f"{t}/{s}"]["same"]]
    later = [f"session {k}: {t}/{s}" for k in found[1:] for t, s in kept if not per[k][f"{t}/{s}"]["same"]]
    return {"states": per[0], "sessions": per, "kept": kept, "later_differ": later,
            "passes": len(kept) >= MIN_HARNESS_STATES and not later}


def gate(root) -> dict:
    check_provenance(root, check_settings)
    return harness(root)


# ---- the episodes -------------------------------------------------------------------------------------------------

def block_rows(root, task: int, state: int, label: str, b: int) -> dict[int, tuple[int, int]]:
    """{id: (success, steps)} of one block; refuses a file without exactly that block's ids."""
    path = block_file(root, task, state, label, b)
    ids = blocks(task, state, label)[b]
    rows = _rows(path) if path.exists() else {}
    if path.exists():
        read_rows(path)                                   # refuses a success other than 0/1
    if any(st != state for _, st in rows):
        raise ValueError(f"{path}: rows of another state")
    got = {seed - EPISODE_SEED_BASE: v for (seed, _), v in rows.items()}
    if set(got) != set(ids):
        raise ValueError(f"{label} {task}/{state} block {b}: {len(set(got) & set(ids))} of {len(ids)} episodes and "
                         f"{len(set(got) - set(ids))} others; finish the run")
    return got


def episodes(root, kept) -> dict[tuple[str, int, int, int], tuple[int, int]]:
    """{(condition, task, state, id): (success, steps)} over every condition and kept state; all must be complete."""
    out = {}
    for t, s in kept:
        for label in CONDITIONS:
            for b in range(len(blocks(t, s, label))):
                for k, v in block_rows(root, t, s, label, b).items():
                    out[(label, t, s, k)] = v
    return out


def counts(eps: dict, label: str, kept, n_ids: int | None = None) -> tuple[int, int]:
    """(successes, episodes) of ``label`` over the kept states, on each state's first ``n_ids`` ids (all if None)."""
    y = n = 0
    for t, s in kept:
        for k in condition_ids(t, s, label)[:n_ids]:
            y += eps[(label, t, s, k)][0]
            n += 1
    return y, n


def analyse(root) -> tuple[dict, dict | None]:
    """(report, episodes); episodes is None when the harness stopped the run."""
    prov = check_provenance(root, check_settings)
    h = harness(root)
    out = {"provenance": prov, "harness": h}
    if not h["passes"]:
        out["reading"] = (f"not read: a later session did not reproduce {h['later_differ']}" if h["later_differ"] else
                          f"not read: the harness reproduced {len(h['kept'])} of {len(STATES)} states")
        return out, None
    kept = [tuple(k) for k in h["kept"]]
    eps = episodes(root, kept)
    pooled = {c: counts(eps, c, kept) for c in CONDITIONS}
    n = pooled[REFERENCE][1]
    assert all(pooled[c][1] == n for c in TESTED), "tested conditions with unequal episodes"
    res = rule({c: pooled[c][0] for c in (REFERENCE,) + TESTED}, n)
    out.update({"kept": [list(k) for k in kept], "episodes": len(eps), "pooled": {c: list(v) for c, v in pooled.items()},
                "rule": res, "reading": res["reading"]})
    per_state = {}
    for t, s in kept:
        row = {}
        for c in CONDITIONS:
            y, m = counts(eps, c, [(t, s)])
            row[c] = {"successes": y, "episodes": m, "interval": list(clopper_pearson(y, m))}
        per_state[f"{t}/{s}"] = row
    out["per_state"] = per_state
    # reported, not part of the rule
    rep = {}
    for c in CONDITIONS:
        if c == REFERENCE:
            continue
        ref = counts(eps, REFERENCE, kept, CONDITIONS[c][3])
        rep[c] = {"successes": pooled[c][0], "episodes": pooled[c][1], "repeat_same_ids": list(ref),
                  "ratio_interval": list(ratio_interval(pooled[c][0], ref[0]))}
    keys = [(t, s, k) for t, s in kept for k in condition_ids(t, s, "ar_repeat_rho0.9")]
    only_ref = sum(1 for t, s, k in keys if eps[("repeat", t, s, k)][0] and not eps[("ar_repeat_rho0.9", t, s, k)][0])
    only_r = sum(1 for t, s, k in keys if eps[("ar_repeat_rho0.9", t, s, k)][0] and not eps[("repeat", t, s, k)][0])
    rep["mcnemar_repeat_vs_rho0.9"] = {"repeat_only": only_ref, "rho0.9_only": only_r,
                                       "p_one_sided": mcnemar_greater(only_ref, only_r)}
    rep["rho0_vs_sigma1"] = {"ar_repeat_rho0": pooled["ar_repeat_rho0"], "scaled_sigma1": pooled["scaled_sigma1"]}
    rep["rho_trend"] = {c: list(counts(eps, c, kept, N_REPORTED)) for c in RHO_TREND}
    rep["zero_outcomes"] = {f"{t}/{s}": {"outcomes": [list(eps[("zero", t, s, k)]) for k in condition_ids(t, s, "zero")],
                                "agree": len({eps[("zero", t, s, k)] for k in condition_ids(t, s, "zero")}) == 1}
                   for t, s in kept}
    out["reported"] = rep
    return out, eps


def _rate(y, n):
    return f"{y}/{n} ({100 * y / n:.1f}%)" if n else f"{y}/{n}"


def format_report(r: dict) -> str:
    h = r["harness"]
    lines = [f"step 18 (M1): constancy or lower noise diversity, commit {r['provenance']['commit']}",
             f"harness: {len(h['kept'])} of {len(STATES)} states reproduce both committed episodes "
             f"({'ok' if h['passes'] else 'STOPPED'})"]
    differ = [k for k, v in h["states"].items() if not v["same"]]
    if differ:
        lines.append(f"  dropped (did not reproduce): {differ}")
    lines.append(f"  harness sessions: {len(h['sessions'])}")
    if h["later_differ"]:
        lines.append(f"  a later session did not reproduce a kept state (stack changed): {h['later_differ']}")
    if "rule" not in r:
        lines.append(f"reading: {r['reading']}")
        return "\n".join(lines) + "\n"
    res = r["rule"]
    lines.append("pooled over the kept states: " + ", ".join(f"{c} {_rate(*r['pooled'][c])}" for c in CONDITIONS))
    if res["read"]:
        for c in TESTED:
            v = res["conditions"][c]
            lo, hi = v["ratio_interval"]
            lines.append(f"  {c}: one-sided Fisher p {res['p'][c]:.3g} ({'below' if v['below'] else 'not below'}); "
                         f"rate ratio to repeat 95% {lo:.2f}-{hi:.2f} ({'keeps half' if v['keeps_half'] else 'not shown to keep half'}): "
                         f"{v['reading']}")
    lines.append(f"reading: {r['reading']}")
    lines.append("reported, not part of the rule:")
    for k, row in r["per_state"].items():
        lines.append(f"  {k}: " + ", ".join(f"{c} {row[c]['successes']}/{row[c]['episodes']}" for c in CONDITIONS))
    rep = r["reported"]
    for c in CONDITIONS:
        if c in rep:
            v = rep[c]
            lo, hi = v["ratio_interval"]
            lines.append(f"  {c}: {_rate(v['successes'], v['episodes'])} against repeat {_rate(*v['repeat_same_ids'])} "
                         f"on the same ids; ratio 95% {lo:.2f}-{hi:.2f}")
    m = rep["mcnemar_repeat_vs_rho0.9"]
    lines.append(f"  paired repeat vs rho 0.9: repeat only {m['repeat_only']}, rho 0.9 only {m['rho0.9_only']}, "
                 f"one-sided McNemar p {m['p_one_sided']:.3g}")
    lines.append("  rho trend (first %d ids): " % N_REPORTED + ", ".join(f"{c} {_rate(*v)}" for c, v in rep["rho_trend"].items()))
    lines.append("  zero: " + ", ".join(f"{k} {v['outcomes']}{'' if v['agree'] else ' DISAGREE'}" for k, v in rep["zero_outcomes"].items()))
    return "\n".join(lines) + "\n"


def write_table(eps: dict, out) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "task_id", "init_state", "noise_id", "success", "steps"])
        for (c, t, s, k), (y, steps) in sorted(eps.items()):
            w.writerow([c, t, s, k, y, steps])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", choices=["harness"])
    ap.add_argument("--out")
    ap.add_argument("--table")
    a = ap.parse_args(argv)
    if a.gate:
        h = gate(a.root)
        print(f"gate harness {'passed' if h['passes'] else 'FAILED'}: {len(h['kept'])} of {len(STATES)} states "
              f"reproduce in session 0; kept {h['kept']}; sessions {len(h['sessions'])}"
              + (f"; a later session differs: {h['later_differ']}" if h["later_differ"] else ""))
        if a.out:
            Path(a.out).write_text(json.dumps(h, indent=1) + "\n")
        return 0 if h["passes"] else 1
    r, eps = analyse(a.root)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    if a.table and eps is not None:
        write_table(eps, a.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
