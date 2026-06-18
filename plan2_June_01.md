## A high-probability novel direction for your goal:

* Shift from online RL-heavy training to selective correction with confidence gating.
* Core idea:
    * Keep frozen SmolVLA as primary controller.
    * Train a lightweight residual corrector offline.
    * Activate residual only when a learned failure-risk head says intervention is needed.
* Why this is novel and practical:
    * It targets reliability and recovery, not replacing the base policy.
    * It directly connects to your failure taxonomy work and avoids unstable full online RL loops.
    * It is computationally cheaper and easier to ablate cleanly.

## Concrete 4-week contribution path:

1. Week A: Build clean benchmark harness
* Full LIBERO-10 eval parity, 3 seeds, consistent success extraction and preprocessing.
* Keep this as your paper backbone.
2. Week B: Failure-risk gating module
* Train a binary risk predictor from rollout states (success/failure horizons).
* Report calibration and trigger precision/recall.
3. Week C: Gated residual corrector
* Offline train residual head on failure-heavy segments.
* At inference: action = base + gate * residual.
* Gate can be hard or soft with threshold sweep.
4. Week D: Ablations and claim package
* Base vs always-on residual vs gated residual.
* Success, recovery-success, latency, VRAM.
* Per-task gains and confidence intervals.

## Practical go/no-go threshold:

* Go if gated residual gives at least +2 absolute points on full LIBERO-10 mean across 3 seeds with no major latency hit.
* No-go if gains are only on a small subset or variance overlaps baseline.
