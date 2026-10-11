---
name: vla_research_validator
description: >-
  Validates VLA code invariants, checks for known regression patterns,
  prevents repeated debugging loops (segfaults, 0% performance drops),
  and validates codebase health before committing to git.
---

# VLA Research Validator Skill

This skill enforces critical engineering invariants for the `efficient_vla` codebase. It prevents agents from hallucinating diagnoses, breaking working components while attempting ad-hoc fixes, or committing unverified changes to version control.

---

## When to Apply This Skill

Activate and follow this skill whenever:
1. **Modifying core agent architectures**: editing `hybrid_diffusion_agent.py`, `rl_residual_agent.py`, or policy forward passes.
2. **Modifying evaluation or training loops**: editing `train_hybrid_diffusion.py`, `eval_hybrid_diffusion.sh`, or RL loops.
3. **Investigating unexpected performance drops**: e.g., success rate dropping to 0% after code changes.
4. **Encountering crashes or segmentation faults**: e.g., C-level stack overflow, EGL driver crashes, XML errors.
5. **Before committing or pushing code to Git**: running the invariant suite to guarantee zero regressions.

---

## The 5 Cardinal Invariants (Must NEVER Be Violated)

| # | Invariant | Why It Exists | Violation Consequence |
| :- | :--- | :--- | :--- |
| **1** | **Pure Prefix Self-Attention**<br>`inputs_embeds=[prefix_embs, None]` with `use_cache=True` | The checkpoint was trained with prefix self-attention only. | Injecting dummy expert tokens triggers cross-attention against zeroes, corrupting condition embeddings and causing **0% success rate**. |
| **2** | **Eval Recursion Guard**<br>`self.base_policy.eval = lambda *a, **kw: self.base_policy` | SmolVLA `select_action()` calls `.eval()` internally on every step. | Recursively traversing 500M parameters across 520 steps causes **Segmentation fault (Exit 139)** in `named_children`. |
| **3** | **Repo Root in PYTHONPATH**<br>`PYTHONPATH="$REPO_ROOT:..."` | `sitecustomize.py` must load `robosuite_mujoco_patch.py`. | Missing patch causes MuJoCo 3.10 joint assertions and memory corruption on episode resets. |
| **4** | **Safe Batching**<br>`BATCH_SIZE <= 3` in evaluation runner | Headless EGL in WSL2 leaks OpenGL/XML handles across resets. | 5 consecutive resets in a single process triggers **fatal segfault in `find_elements`** on Episode 5. |
| **5** | **Closed-Loop Replanning**<br>`--eval_replan_each_step 1` | Multi-step contact-rich tasks require constant closed-loop error correction. | Open-loop execution (`eval_replan_each_step=0`) drifts, dropping success to **0%**. |

---

## Automated Invariant Check (Pre-Commit Gate)

Before staging, committing, or pushing code to git, execute the automated invariant checker:

```bash
bash .gemini/skills/vla_research_validator/scripts/check_invariants.sh
```

If any check fails, **do not commit**. Fix the regression first.

---

## Investigation Protocol: Debugging Performance Drops

If evaluation performance drops significantly (e.g., from 20% to 0%):

1. **Check Failure History First**:
   ```bash
   python .gemini/skills/vla_research_validator/scripts/record_failure.py --list
   ```
   Do not guess or assume. Check if the exact symptom has already been diagnosed.

2. **Verify Against Pure Base Policy Baseline**:
   Always run `--policy_mode base` (or `-a 0.0`) on the exact same task and seeds:
   ```bash
   bash eval_hybrid_diffusion.sh -t <TASK_ID> -m base -e 10 -b 3 -o outputs/verify_base
   ```
   If the base policy also achieves 0% or low score, the issue is task difficulty / environment seed variance, not a broken residual model!

3. **Check Condition Embeddings**:
   Verify that `extract_semantic_features(batch)` outputs vectors with:
   - Condition dimension matching checkpoint (e.g., `cond_dim = 960`).
   - Non-zero variance (`torch.std(semantics) > 1e-4`).
   - No NaNs or infinities.

4. **Verify Seed Batching Alignment**:
   Ensure `eval_seed_offset` correctly matches episode indices so results are comparable across runs.

---

## Recording New Failures

When you diagnose a new non-obvious bug or regression:
```bash
python .gemini/skills/vla_research_validator/scripts/record_failure.py \
  --record \
  --title "Brief Title" \
  --symptom "What was observed (log, metric, exit code)" \
  --root_cause "Technical root cause explanation" \
  --resolution "Exact fix applied" \
  --invariant "Rule added to check_invariants.sh to prevent recurrence" \
  --tags "keyword1,keyword2"
```
This updates both `failures_taxonomy.json` and `FAILURES.md`.

---

## Pre-Push Checklist

- [ ] Ran `bash .gemini/skills/vla_research_validator/scripts/check_invariants.sh` (100% PASS).
- [ ] Base policy comparison confirmed if altering residual weights.
- [ ] No uncommitted scratch files or temporary debug prints left in production paths.
- [ ] Verified `git diff` shows only intended, minimal changes.