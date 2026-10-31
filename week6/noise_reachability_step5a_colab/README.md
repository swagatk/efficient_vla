# Step 5a: deep noise search (Colab A100, 2026-10-30/31)

Protocol: `noise_reachability/README.md`, step 5a, fixed before the run. Commit `522c181` (clean tree in all 136
provenance records), `lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps, task 0. Source:
`step5a_results.zip`, sha256 `c6103270c5e9ae4ed7750bdf20f2173898183ce92f34d1582c0f31f2db5149b7`.

**Result: "mostly reachable" under the rule fixed in advance.** All 5 systematic states were reached, so the
hypothesis (at least one) holds. Under the rule, "systematic" now means rare. But no state was reached by `sampled`,
the way the base policy actually runs.

Harness: base 5/30 against 7 (gap -2, limit 6). Control, state 9, of 35: sampled 8, full 12, repeat 11; both gates passed.

| State | sampled /70 | full /70 | repeat /70 | Jittered successes per read hit, of 4 | Class |
|---|---|---|---|---|---|
| 2 | 0 | 0 | 1 | repeat: 1 | fragile |
| 5 | 0 | 0 | 1 | repeat: 1 | fragile |
| 6 | 0 | 2 | 7 | full: 0, 3; repeat: 1, 0, 2 | robust |
| 7 | 0 | 1 | 3 | full: 0; repeat: 1, 1, 1 | fragile |
| 8 | 0 | 2 | 8 | full: 2, 4; repeat: 4, 0, 3 | robust |
| Total /350 | 0 | 5 | 20 | 24/64 | |

Exact 95% intervals on the pooled rates: sampled 0-1.05%, full 0.47-3.30%, repeat 3.53-8.69%. Exploratory, not part of
the rule: Fisher exact on pooled episodes gives repeat vs sampled p = 1.4e-6, full vs sampled p = 0.062, repeat vs full
p = 0.0035. Episodes on one state are not independent draws across states, and 15 of repeat's 20 hits are on states 6
and 8, so these p-values describe this set of five states only.

Checks: `analyze_step5a.py` rerun locally reproduces the Colab report byte for byte, and `check_step5a_rows.py` (written
separately after the run) recomputes the counts from the rows (`step5a_checks.txt`). All 1,155 search episodes and 64
confirmation episodes are present exactly once, no log has a traceback, and every failed episode ran 520 steps.

Reading limits:

- Hits are picked from the same episodes they are counted on. Jitter is the only held-out check, and it changes the
  noise too; no fixed noise was replayed unchanged, because the episodes are deterministic.
- States 2 and 5 rest on one hit and one jittered success each.
- A fixed noise was chosen after seeing that state's outcome. This is existence of a noise, not a policy that can find
  one. The mechanism for repeat's advantage was not tested.
- Task 0 only. Whether a hit noise helps other states was not tested beyond the id overlaps: repeat ids 30 and 15 hit
  3 and 2 states, which a permutation test puts at chance (p = 0.60 for the number of ids hitting 2 or more states).
