I’ll use your academic-researcher framework and now convert the selected idea into a concrete 12-week execution plan plus a paper-ready ablation table template you can run directly.

Read skill [academic-researcher](file:///home/swagat/GIT/github_skill_expts/.github/skills/academic-researcher/SKILL.md?vscodeLinkType%3Dskill)

### Week-by-Week Experimental Plan (12 Weeks)
Project: Self-Correcting Edge VLA (SmolVLA + Visual Prompting + Lightweight Online RL)

#### Week 1: Reproducible Baseline Lock
1. Freeze a clean SmolVLA baseline pipeline on LIBERO-10.
2. Re-run and verify your baseline success rate target (around 46%) with 3 seeds.
3. Log fixed protocol:
    1. task split,
    2. episode horizon,
    3. camera settings,
    4. evaluation seeds,
    5. hardware stats.
4. Add profiling hooks:
    1. peak VRAM,
    2. average inference latency per step,
    3. end-to-end control loop rate.

Deliverable: Baseline report and exact reproducibility config.

#### Week 2: Visual Prompting Integration (No RL Yet)
1. Integrate FastSAM or GroundingDINO object grounding.
2. Implement two prompt encodings:
1. box-overlay channel on image,
2. tokenized spatial hint in text prompt.
3. Add fallback mode when grounding fails.

Deliverable: Stage-1 code with deterministic prompt generation.

#### Week 3: Stage-1 Performance Sweep
1. Sweep grounding frequency (every step vs every N steps).
2. Sweep box style (filled, edge-only, mask).
3. Evaluate with 3 seeds.

Go/No-Go Gate A:
    1. Must improve baseline by at least +3 to +5 points.
    2. Latency increase should stay under 20%.

Deliverable: Best Stage-1 config selected.

#### Week 4: Error Taxonomy and RL Interface
1. Build failure taxonomy from rollouts:
    1. wrong object attention,
    2. grasp pose error,
    3. placement precision error,
    4. recovery failure.
2. Add online RL interface in simulator only.
3. Decide RL update target:
    1. last action head only, or
    2. small residual correction head.

Deliverable: RL-ready training loop and failure buckets.

#### Week 5: Lightweight Online RL Prototype
1. Start with conservative RL:
1. frozen backbone,
2. trainable last layers only.
2. Add behavior-cloning anchor loss to prevent drift.
3. Add KL regularization to baseline policy outputs.

Deliverable: Stable RL run without catastrophic collapse.

#### Week 6: Reward and Stability Tuning
1. Tune reward:
1. sparse task success,
2. dense shaping for approach/grasp/place,
3. penalties for unsafe contacts.
2. Tune update frequency and batch size for 16 GB VRAM.
3. Run 2-3 short training schedules.

Go/No-Go Gate B:
1. RL model must beat Stage-1 by at least +2 points in held-out eval seeds.
2. No major regression in easy tasks.

Deliverable: Stable RL recipe.

#### Week 7: Core Ablation Batch 1
1. Run full ablations for visual prompting variants.
2. Run ablations for RL target module choice.
3. Run ablations for reward composition.

Deliverable: First complete ablation table (draft).

#### Week 8: Core Ablation Batch 2 (Efficiency + Robustness)
1. Measure:
1. latency,
2. VRAM,
3. training wall-clock,
4. throughput.
2. Add perturbation robustness:
1. distractor objects,
2. lighting variation,
3. minor camera jitter.

Deliverable: Efficiency and robustness section ready.

#### Week 9: Generalization and Overfitting Check
1. Evaluate on held-out LIBERO tasks or modified initial states.
2. Check if gains are concentrated in narrow task families.
3. Run cross-seed variance analysis and confidence intervals.

Go/No-Go Gate C:
1. Gains must hold beyond one seed and one task subset.

Deliverable: Final model checkpoint candidate.

#### Week 10: Real-Time Edge Profile and Demo Runs
1. Benchmark real-time loop behavior:
1. policy-only,
2. visual prompting enabled,
3. visual prompting + RL-corrected policy.
2. Record qualitative success/failure videos for representative tasks.

Deliverable: Edge deployment profile and demo evidence.

#### Week 11: Paper Figures and Claims Validation
1. Finalize all plots:
1. success bars with confidence intervals,
2. efficiency Pareto (success vs latency),
3. failure-category reduction chart.
2. Red-team claims: verify each claim has at least one quantitative table.

Deliverable: Camera-ready experimental artifacts.

#### Week 12: Writing and Submission Package
1. Write method and experiment sections first.
2. Freeze code commit and config file bundle.
3. Prepare reproducibility appendix:
1. seeds,
2. hyperparameters,
3. hardware profile,
4. failure cases.

Deliverable: Submission-ready manuscript and reproducibility package.

---

### Exact Ablation Table Template
Use this as your primary table schema (one row per run).

| Run ID | Model Variant | Visual Prompting | Prompt Encoder | Grounding Model | Grounding Freq | RL Used | RL Algo | Trainable Params | BC Anchor | KL Reg | Reward Setting | LIBERO-10 Success (%) | Delta vs Baseline | Recovery Success (%) | Avg Step Latency (ms) | Peak VRAM (GB) | Train Time (hrs) | Notes |
|---|---|---|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | SmolVLA baseline | No | None | None | None | No | None | 0 | No | No | None | 46.0 | 0.0 | 0.0 | TBD | TBD | 0 | Baseline |
| V1 | Stage-1 | Yes | Box overlay | FastSAM | Every step | No | None | 0 | No | No | None | TBD | TBD | TBD | TBD | TBD | 0 | Visual only |
| V2 | Stage-1 | Yes | Box overlay | FastSAM | Every 3 steps | No | None | 0 | No | No | None | TBD | TBD | TBD | TBD | TBD | 0 | Lower latency |
| V3 | Stage-1 | Yes | Spatial tokens | GroundingDINO | Every step | No | None | 0 | No | No | None | TBD | TBD | TBD | TBD | TBD | 0 | Token prompt |
| R1 | Stage-2 | Yes | Best from V* | Best from V* | Best from V* | Yes | PPO/AWAC/SAC-lite | Last action head | Yes | Yes | Dense+sparse | TBD | TBD | TBD | TBD | TBD | TBD | Main method |
| R2 | Stage-2 | Yes | Best from V* | Best from V* | Best from V* | Yes | Same as R1 | Residual head | Yes | Yes | Dense+sparse | TBD | TBD | TBD | TBD | TBD | TBD | Module ablation |
| R3 | Stage-2 | Yes | Best from V* | Best from V* | Best from V* | Yes | Same as R1 | Last action head | No | Yes | Dense+sparse | TBD | TBD | TBD | TBD | TBD | TBD | No BC anchor |
| R4 | Stage-2 | Yes | Best from V* | Best from V* | Best from V* | Yes | Same as R1 | Last action head | Yes | No | Dense+sparse | TBD | TBD | TBD | TBD | TBD | TBD | No KL |
| R5 | Stage-2 | Yes | Best from V* | Best from V* | Best from V* | Yes | Same as R1 | Last action head | Yes | Yes | Sparse only | TBD | TBD | TBD | TBD | TBD | TBD | Reward ablation |
| E1 | Efficiency | Yes | Best | Best | Every 5 steps | Yes | Best | Best | Yes | Yes | Best | TBD | TBD | TBD | TBD | TBD | TBD | Edge budget mode |

---

### Minimum Ablations Required for a Strong Paper
1. Visual prompting on/off.
2. Grounding frequency sweep.
3. Prompt representation sweep (overlay vs token hint).
4. RL on/off after Stage-1.
5. Trainable module sweep (last head vs residual head).
6. Stability controls (with/without BC anchor, with/without KL).
7. Reward design (sparse vs mixed dense+sparse).
8. Efficiency tradeoff (success vs latency vs VRAM).

---

### Statistical Protocol (important)
1. Run every key model with at least 3 seeds (preferably 5 for final claims).
2. Report mean plus standard deviation and 95% confidence interval.
3. Use paired comparison against baseline per task when possible.
4. Predefine primary metric:
1. LIBERO-10 task success rate.
5. Predefine secondary metrics:
1. recovery success,
2. latency,
3. peak VRAM.

---

### Risk-Control Checklist (operational)
1. If Stage-1 gain is less than +3 points by Week 3, simplify grounding and move quickly to RL.
2. If RL destabilizes by Week 6, freeze more layers and switch to residual-only correction.
3. If latency exceeds budget, reduce grounding frequency and action chunk recalculation rate.
4. If generalization is weak, shift claims to edge-efficiency and reliability improvements rather than SOTA success.

If you want, I can now give you a ready-to-use experiment tracking sheet format (CSV headers and run naming convention) so you can start logging runs immediately.