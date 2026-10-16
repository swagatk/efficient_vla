#!/usr/bin/env bash
# check_invariants.sh
# Automated regression and invariant checker for efficient_vla.
# Run this script before committing or pushing changes to git to prevent
# repeating past mistakes and breaking working baselines.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"

echo "============================================================"
echo "🔍 Running VLA Research Validator Invariant Checks..."
echo "Repo Root: $REPO_ROOT"
echo "============================================================"

FAILURES=0
WARNINGS=0

pass() {
  echo "  ✅ [PASS] $1"
}

fail() {
  echo "  ❌ [FAIL] $1"
  FAILURES=$((FAILURES + 1))
}

warn() {
  echo "  ⚠️  [WARN] $1"
  WARNINGS=$((WARNINGS + 1))
}

# --- Check 1: Python Syntax Check ---
echo ""
echo "--- 1. Python Syntax & Static Validation ---"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "/home/swagat/miniconda3/envs/lerobot_env/bin/python" ]]; then
    PYTHON_BIN="/home/swagat/miniconda3/envs/lerobot_env/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

# A listed file that no longer exists fails the check, so a move cannot silently drop it from checking.
for py_file in "$REPO_ROOT/hybrid_diffusion/hybrid_diffusion_agent.py" \
               "$REPO_ROOT/hybrid_diffusion/train_hybrid_diffusion.py" \
               "$REPO_ROOT/rl_based_tuning/train_online_ppo.py" \
               "$REPO_ROOT/rl_based_tuning/train_online_sac.py" \
               "$REPO_ROOT/rl_based_tuning/sac_residual.py" \
               "$REPO_ROOT/vla_core/envs.py" \
               "$REPO_ROOT/vla_core/policy.py" \
               "$REPO_ROOT/vla_core/residual.py" \
               "$REPO_ROOT/vla_core/evaluation.py"; do
  if [[ ! -f "$py_file" ]]; then
    fail "Checked file missing (moved or deleted? update this list): $py_file"
  elif "$PYTHON_BIN" -m py_compile "$py_file" 2>/dev/null; then
    pass "Syntax valid: ${py_file#$REPO_ROOT/}"
  else
    fail "Syntax error in: $py_file"
  fi
done

# Residual RL must run the base policy through vla_core (lerobot-eval's processing) and add the residual in
# env action units; the pre-fix SAC clamped normalised actions and re-reset after set_init_state.
SAC_FILE="$REPO_ROOT/rl_based_tuning/train_online_sac.py"
if [[ -f "$SAC_FILE" ]]; then
  if grep -q "from vla_core.policy import load_base_policy" "$SAC_FILE" && grep -q "from vla_core.residual import apply_residual" "$SAC_FILE"; then
    pass "train_online_sac.py uses vla_core's base policy and apply_residual"
  else
    fail "train_online_sac.py does not use vla_core.policy.load_base_policy and vla_core.residual.apply_residual"
  fi
  if grep -qE "set_init_state|env\.reset\(" "$SAC_FILE"; then
    fail "train_online_sac.py resets or sets init states itself; use vla_core.evaluation.run_episode"
  else
    pass "train_online_sac.py leaves resets and init states to vla_core"
  fi
fi

# --- Check 2: robosuite_mujoco_patch Presence & Loading ---
echo ""
echo "--- 2. Robosuite / MuJoCo Patch Invariants ---"
if [[ -f "$REPO_ROOT/robosuite_mujoco_patch.py" ]]; then
  pass "robosuite_mujoco_patch.py exists in repo root"
else
  fail "robosuite_mujoco_patch.py missing from repo root! Essential for MuJoCo 3.x/PyTorch 2.6+"
fi

if [[ -f "$REPO_ROOT/sitecustomize.py" ]]; then
  if grep -q "robosuite_mujoco_patch" "$REPO_ROOT/sitecustomize.py"; then
    pass "sitecustomize.py auto-loads robosuite_mujoco_patch"
  else
    fail "sitecustomize.py exists but does not load robosuite_mujoco_patch"
  fi
else
  warn "sitecustomize.py missing from repo root"
fi

# Check that eval_hybrid_diffusion.sh includes REPO_ROOT in PYTHONPATH
if [[ -f "$REPO_ROOT/hybrid_diffusion/eval_hybrid_diffusion.sh" ]]; then
  if grep -q "REPO_ROOT" "$REPO_ROOT/hybrid_diffusion/eval_hybrid_diffusion.sh"; then
    pass "eval_hybrid_diffusion.sh includes REPO_ROOT in PYTHONPATH"
  else
    fail "eval_hybrid_diffusion.sh does not include REPO_ROOT in PYTHONPATH (sitecustomize will not load!)"
  fi
fi

# Check that train_hybrid_diffusion.py imports robosuite_mujoco_patch
if [[ -f "$REPO_ROOT/hybrid_diffusion/train_hybrid_diffusion.py" ]]; then
  if grep -q "robosuite_mujoco_patch" "$REPO_ROOT/hybrid_diffusion/train_hybrid_diffusion.py"; then
    pass "train_hybrid_diffusion.py explicitly imports robosuite_mujoco_patch"
  else
    warn "train_hybrid_diffusion.py does not explicitly import robosuite_mujoco_patch"
  fi
fi

# Only robosuite_mujoco_patch.py may wrap mujoco.mj_fullM; stacked wrappers corrupt the OSC mass matrix (FAIL-007)
DUP_FULLM=$(grep -lE "^[[:space:]]*mujoco\.mj_fullM[[:space:]]*=" "$REPO_ROOT"/*.py "$REPO_ROOT"/hybrid_diffusion/*.py 2>/dev/null | grep -v "/robosuite_mujoco_patch.py$" || true)
if [[ -z "$DUP_FULLM" ]]; then
  pass "mujoco.mj_fullM is patched only in robosuite_mujoco_patch.py"
else
  fail "mujoco.mj_fullM is re-patched outside robosuite_mujoco_patch.py: $DUP_FULLM"
fi

# --- Check 3: Architecture & Conditioning Invariants ---
echo ""
echo "--- 3. Hybrid Diffusion Architecture Invariants ---"
AGENT_FILE="$REPO_ROOT/hybrid_diffusion/hybrid_diffusion_agent.py"
if [[ -f "$AGENT_FILE" ]]; then
  # Invariant: No dummy expert tokens injected into prefix attention
  if grep -q "dummy_expert_embs" "$AGENT_FILE"; then
    fail "dummy_expert_embs found in hybrid_diffusion_agent.py! This corrupts prefix conditioning with cross-attention against zeros."
  else
    pass "No spurious dummy expert token in extract_semantic_features"
  fi

  # Invariant: Base policy eval recursion guard
  if grep -q "base_policy.eval = lambda" "$AGENT_FILE"; then
    pass "base_policy.eval recursion guard is active (prevents select_action named_children segfault)"
  else
    fail "base_policy.eval = lambda guard missing in hybrid_diffusion_agent.py! select_action will segfault."
  fi

  # Invariant: Clean inputs_embeds in vlm_with_expert.forward
  if grep -q "inputs_embeds=\[prefix_embs, None\]" "$AGENT_FILE"; then
    pass "inputs_embeds properly formatted as [prefix_embs, None] for pure prefix self-attention"
  else
    warn "Verify inputs_embeds format in hybrid_diffusion_agent.py"
  fi
fi

# --- Check 4: Rollout Inference Mode ---
echo ""
echo "--- 4. Rollout Inference Context Invariants ---"
TRAIN_FILE="$REPO_ROOT/hybrid_diffusion/train_hybrid_diffusion.py"
if [[ -f "$TRAIN_FILE" ]]; then
  # We should use torch.inference_mode() in evaluate_in_environment, not torch.no_grad()
  if grep -q "with torch.inference_mode():" "$TRAIN_FILE"; then
    pass "evaluate_in_environment uses torch.inference_mode() for rollouts"
  else
    warn "evaluate_in_environment does not use torch.inference_mode()"
  fi
fi

# --- Check 5: Evaluation Safe Batching ---
echo ""
echo "--- 5. Headless EGL / Driver Safety Invariants ---"
EVAL_SH="$REPO_ROOT/hybrid_diffusion/eval_hybrid_diffusion.sh"
if [[ -f "$EVAL_SH" ]]; then
  # Default batch size must not exceed 4 (5 crashes on WSL2 EGL reset)
  DEFAULT_BATCH=$(grep -E 'BATCH_SIZE="\$\{BATCH_SIZE:-[0-9]+\}"' "$EVAL_SH" | grep -o -E '[0-9]+' | tail -1 || echo "0")
  if [[ "$DEFAULT_BATCH" -le 3 && "$DEFAULT_BATCH" -gt 0 ]]; then
    pass "Default evaluation batch size ($DEFAULT_BATCH) <= 3 to prevent EGL buffer overflow"
  elif [[ "$DEFAULT_BATCH" -gt 3 ]]; then
    fail "Default evaluation batch size is $DEFAULT_BATCH > 3! Consecutive resets will trigger robosuite XML segfault on episode 5."
  else
    warn "Unable to verify default BATCH_SIZE in eval_hybrid_diffusion.sh"
  fi
fi

# --- Summary ---
echo ""
echo "============================================================"
if [[ "$FAILURES" -eq 0 ]]; then
  echo "🎉 All critical invariant checks PASSED! ($WARNINGS warning(s))"
  echo "Code is safe to commit."
  exit 0
else
  echo "🚨 Invariant checks FAILED with $FAILURES failure(s) and $WARNINGS warning(s)!"
  echo "Please fix the failing invariant(s) before committing or evaluating."
  exit 1
fi