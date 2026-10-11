# VLA Research Failure Taxonomy & Regression Guard

This file is automatically updated by `record_failure.py`. Before modifying agent code or debugging performance drops, check these documented failure modes to avoid repeating past mistakes.

| ID | Date | Title | Key Symptom | Primary Invariant |
| :--- | :--- | :--- | :--- | :--- |
| **FAIL-001** | 2026-10-06 | VLM Semantic Conditioning Corruption (0% Success) | Evaluation success rate on LIBERO Task 0 suddenly dropped fr... | Never pass non-None expert tokens to the frozen VLM backbone... |
| **FAIL-002** | 2026-10-06 | PyTorch Submodule Traversal Recursion Segfault (Exit 139) | Evaluation crashed with `Segmentation fault (core dumped)` i... | Always override `self.base_policy.eval` with a lambda no-op ... |
| **FAIL-003** | 2026-10-06 | Missing robosuite_mujoco_patch in PYTHONPATH | Robosuite / MuJoCo 3.x memory corruption and joint property ... | All runner scripts and Python entry points must ensure repos... |
| **FAIL-004** | 2026-10-06 | Headless EGL Resource Accumulation Segfault on 5th Episode Reset | Evaluation crashed with `Segmentation fault` inside `robosui... | Never run more than 3 consecutive environment resets per Pyt... |
| **FAIL-005** | 2026-10-04 | Open-Loop Residual Execution Collapse (replan_each_step=False) | Residual evaluation dropped to 0.0% when replan_each_step wa... | Residual evaluations on LIBERO must use `--eval_replan_each_... |
| **FAIL-006** | 2026-10-07 | LeRobot make_env Wrapper Degradation & Batch 2 Segfault | Task 3 baseline evaluation resulted in 0.00% success rate (i... | Always evaluate LIBERO policies using direct OffScreenRender... |
| **FAIL-007** | 2026-10-10 | Stacked mj_fullM Patches Corrupt OSC Mass Matrix (0% Base Success in Hybrid Harness) | Frozen base SmolVLA scored 0/10 on Task 3 through train_hybr... | Only robosuite_mujoco_patch.py may patch mujoco.mj_fullM. No... |
| **FAIL-008** | 2026-10-11 | Runner Aborted by Segfault in a Helper Python Call (sitecustomize imports torch) | run_task3_residual.sh died with 'Fatal Python error: Segment... | Helper Python calls in runner scripts that do not need torch... |

---

### [FAIL-001] VLM Semantic Conditioning Corruption (0% Success)

- **Date**: 2026-10-06
- **Symptom**: Evaluation success rate on LIBERO Task 0 suddenly dropped from 20% to 0% across all 10 episodes.
- **Root Cause**: A dummy expert token (1 token, dim 480) was injected into `inputs_embeds = [prefix_embs, dummy_expert_embs]` with `use_cache=False`. Inside SmolVLMWithExpertModel, `fill_kv_cache=False` forced `forward_cross_attn_layer` instead of `forward_attn_layer`. All visual and state prefix tokens cross-attended against zeroes across all layers, completely distorting the conditioning vector passed to the diffusion head.
- **Resolution**: Removed dummy_expert_embs completely. Restored `inputs_embeds=[prefix_embs, None]` with `use_cache=True` and `past_key_values=None` so `fill_kv_cache=True` runs clean prefix self-attention, matching the trained checkpoint. Delete past_key_values immediately after forward pass to prevent VRAM accumulation.
- **Mandatory Invariant**: `Never pass non-None expert tokens to the frozen VLM backbone. Always use `inputs_embeds=[prefix_embs, None]` and clean self-attention prefill.`

### [FAIL-002] PyTorch Submodule Traversal Recursion Segfault (Exit 139)

- **Date**: 2026-10-06
- **Symptom**: Evaluation crashed with `Segmentation fault (core dumped)` in `named_children` / `train` / `eval` on step 0 of episode 1.
- **Root Cause**: SmolVLAPolicy.select_action(batch) calls `self.eval()` internally on every step. Calling `.eval()` traverses all submodules in SmolVLMWithExpertModel recursively. In PyTorch with deep multimodal hierarchies, calling this 520 times per episode overflows the C-level call stack.
- **Resolution**: Freeze base_policy in eval mode at init: `self.base_policy.eval = lambda *args, **kwargs: self.base_policy`. This neutralizes repeated traversals while keeping the model deterministic.
- **Mandatory Invariant**: `Always override `self.base_policy.eval` with a lambda no-op after initial `self.base_policy.eval()` call in wrapper classes.`

### [FAIL-003] Missing robosuite_mujoco_patch in PYTHONPATH

- **Date**: 2026-10-06
- **Symptom**: Robosuite / MuJoCo 3.x memory corruption and joint property assertions during environment resets.
- **Root Cause**: eval_hybrid_diffusion.sh only added `hybrid_diffusion/` to PYTHONPATH, omitting the repository root. `sitecustomize.py` and `robosuite_mujoco_patch.py` were never loaded.
- **Resolution**: Added `REPO_ROOT` to PYTHONPATH in `eval_hybrid_diffusion.sh` and added fallback explicit import in `train_hybrid_diffusion.py`.
- **Mandatory Invariant**: `All runner scripts and Python entry points must ensure repository root is in PYTHONPATH so `sitecustomize.py` loads `robosuite_mujoco_patch`.`

### [FAIL-004] Headless EGL Resource Accumulation Segfault on 5th Episode Reset

- **Date**: 2026-10-06
- **Symptom**: Evaluation crashed with `Segmentation fault` inside `robosuite/utils/mjcf_utils.py` `find_elements` on Episode 5 reset.
- **Root Cause**: Under headless EGL in WSL2, Robosuite accumulates internal C-level XML ElementTree handles and OpenGL contexts across resets. Running 5 consecutive resets in a single Python process exceeds the memory threshold.
- **Resolution**: Changed default batch size in `eval_hybrid_diffusion.sh` to 3 episodes per process. Each sub-process runs at most 3 episodes, exiting cleanly and recycling the driver context.
- **Mandatory Invariant**: `Never run more than 3 consecutive environment resets per Python process under headless EGL.`

### [FAIL-005] Open-Loop Residual Execution Collapse (replan_each_step=False)

- **Date**: 2026-10-04
- **Symptom**: Residual evaluation dropped to 0.0% when replan_each_step was False.
- **Root Cause**: When `eval_replan_each_step=False`, the model predicts a 16-step residual chunk and executes it open-loop. In a contact-rich 520-step manipulation task, compounding drift without closed-loop replanning causes failure.
- **Resolution**: Set `--eval_replan_each_step 1` by default for closed-loop residual adjustment on every step.
- **Mandatory Invariant**: `Residual evaluations on LIBERO must use `--eval_replan_each_step 1` unless specifically testing open-loop horizon stability.`

### [FAIL-006] LeRobot make_env Wrapper Degradation & Batch 2 Segfault

- **Date**: 2026-10-07
- **Symptom**: Task 3 baseline evaluation resulted in 0.00% success rate (instead of ~83% expected baseline), and the evaluation script crashed on launching Batch 2 with Segmentation fault (core dumped).
- **Root Cause**: evaluate_in_environment in train_hybrid_diffusion.py introduced a conditional use_lerobot_env block using make_env which omitted benchmark initial state setup (env.set_init_state), skipped 10-step dummy action physics stabilization, did not enable delta robot control (use_delta = True), ignored image_flip_mode (feeding inverted OpenGL camera images to SmolVLA), and applied duplicate preprocessing and postprocessing. Batch 2 crashed with Segfault due to zero inter-batch delay under WSL2 EGL.
- **Resolution**: Replaced make_env with direct OffScreenRenderEnv pipeline matching test_base.py, guaranteeing benchmark init_states, dummy action stabilization, delta control, correct image flipping, and single-stage LeRobot preprocessing. Added 2-second inter-batch delay in eval_hybrid_diffusion.sh.
- **Mandatory Invariant**: `Always evaluate LIBERO policies using direct OffScreenRenderEnv with benchmark initial states, 10 dummy warmup steps, delta controller, vertical_horizontal image flips, and a 2-second delay between batched runner processes.`

### [FAIL-007] Stacked mj_fullM Patches Corrupt OSC Mass Matrix (0% Base Success in Hybrid Harness)

- **Date**: 2026-10-10
- **Symptom**: Frozen base SmolVLA scored 0/10 on Task 3 through train_hybrid_diffusion.py --eval_only (expected ~83%), while lerobot_eval on the same task was fine. Robot moved but never completed the task; no crash.
- **Root Cause**: train_hybrid_diffusion.py carried its own copy of the robosuite/MuJoCo compatibility patch and applied it on top of robosuite_mujoco_patch (already loaded via sitecustomize and the explicit import). The second mujoco.mj_fullM wrapper forwarded robosuite's (model, dst, qM) call to the first wrapper as (model, data, dst), which then unpacked MuJoCo 3.13's data.M with the legacy dof_Madr layout. The OSC controller received a wrong mass matrix (max abs error 4.67 vs mj_mulM ground truth, negative diagonal terms), so every commanded delta pose was executed with wrong torques.
- **Resolution**: Deleted the duplicated patch block from train_hybrid_diffusion.py; the script now only imports robosuite_mujoco_patch (hard import, no silent fallback). Controller mass matrix error vs ground truth is 3.5e-5 afterwards and base Task 3 succeeds 3/3 in the smoke check.
- **Mandatory Invariant**: `Only robosuite_mujoco_patch.py may patch mujoco.mj_fullM. No other file may assign mujoco.mj_fullM; verify any eval harness by reproducing the base-policy success rate before measuring a new method.`

### [FAIL-008] Runner Aborted by Segfault in a Helper Python Call (sitecustomize imports torch)

- **Date**: 2026-10-11
- **Symptom**: run_task3_residual.sh died with 'Fatal Python error: Segmentation fault' after base-eval seeds 12..14 on Task 0. Stack shows sitecustomize -> robosuite_mujoco_patch -> import torch. The 15 finished episodes were intact; no eval batch had failed.
- **Root Cause**: The crash was not in an evaluation process. Between batches the runner calls a one-line 'python -c' to count finished seeds in eval_results.json. With the repo root on PYTHONPATH, every interpreter start runs sitecustomize, which imports torch and patches MuJoCo; that import segfaulted once (transient, right after a GPU/EGL process exited under WSL2). The call sat in a command substitution under 'set -e', so one helper crash aborted the whole multi-hour run.
- **Resolution**: The resume check now runs with 'python -S' (no sitecustomize, stdlib only) and falls back to 0 on any failure; each eval batch is retried up to 3 times until its seeds are present in eval_results.json. The run was resumed from the saved episode records.
- **Mandatory Invariant**: `Helper Python calls in runner scripts that do not need torch/MuJoCo must use 'python -S', and no single helper or batch failure may abort a resumable run under set -e.`

