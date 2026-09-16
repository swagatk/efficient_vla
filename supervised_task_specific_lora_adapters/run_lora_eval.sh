#!/usr/bin/env bash
# run_lora_eval.sh
# Evaluates SmolVLA policy with task-specific LoRA adapters across tasks and seeds.
#
# Usage:
#   bash run_lora_eval.sh --task_id 0 --lora_dir outputs/run_lora_training_20260829_215932 --num_episodes 10 --output_dir lora_eval_results_task0_3seeds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Environment & Interpreter Detection ---
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/swagat/miniconda3/envs/lerobot_env/bin/python" ]]; then
    PYTHON_BIN="/home/swagat/miniconda3/envs/lerobot_env/bin/python"
  elif [[ -x "/home/swagat/anaconda3/envs/lerobot_v040/bin/python" ]]; then
    PYTHON_BIN="/home/swagat/anaconda3/envs/lerobot_v040/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="python3"
  fi
fi
export PYTHON_BIN

# Headless rendering backend for MuJoCo and PyOpenGL
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Add local LIBERO clone to PYTHONPATH if present
if [[ -d "$HOME/LIBERO" ]]; then
  export PYTHONPATH="$HOME/LIBERO:${PYTHONPATH:-}"
fi

# Pre-flight check: verify simulation environment dependencies
if ! "$PYTHON_BIN" -c "import libero, robosuite" >/dev/null 2>&1; then
  echo "================================================================================"
  echo "⚠️ ERROR: Simulation packages 'libero' and/or 'robosuite' failed to import:"
  "$PYTHON_BIN" -c "import libero, robosuite" || true
  echo ""
  echo "Evaluation requires the LIBERO simulation benchmark."
  echo "================================================================================"
  exit 1
fi

# Ensure LIBERO config exists and points to valid directories
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
mkdir -p "$LIBERO_CONFIG_PATH"
"$PYTHON_BIN" - <<'PY'
import os
import importlib.util
from pathlib import Path
import yaml
try:
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    cfg_path = cfg_dir / "config.yaml"
    dataset_root = os.environ.get("DATASET_ROOT", os.path.expanduser("~/lerobot_datasets/libero_10"))

    candidates = [
        Path(os.path.expanduser("~/LIBERO/libero/libero")),
        Path(os.path.expanduser("~/LIBERO/libero")),
    ]
    for mod in ["libero.libero", "libero"]:
        try:
            spec = importlib.util.find_spec(mod)
            if spec and spec.origin:
                p = Path(spec.origin).resolve().parent
                candidates.extend([p, p / "libero"])
        except Exception:
            pass

    benchmark_root = None
    for c in candidates:
        if (c / "init_files").is_dir() and (c / "bddl_files").is_dir():
            benchmark_root = c
            break

    if benchmark_root is None:
        for c in candidates:
            if (c / "init_files").is_dir():
                benchmark_root = c
                break

    if benchmark_root is None:
        benchmark_root = Path(os.path.expanduser("~/LIBERO/libero/libero"))

    needs_write = True
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                curr = yaml.safe_load(f)
            if isinstance(curr, dict) and Path(curr.get("init_states", "")).is_dir():
                needs_write = False
        except Exception:
            pass

    if needs_write:
        config = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(Path(dataset_root).resolve()),
            "assets": str(benchmark_root / "assets"),
        }
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"[setup] Wrote validated LIBERO config: {cfg_path}")
        print(f"[setup]   init_states: {benchmark_root / 'init_files'}")
except Exception as e:
    print(f"[setup] Warning: Could not generate LIBERO config: {e}")
PY

# --- Windows WSL & Linux Power Profile & Sleep/Hibernation Prevention ---
IS_WSL=0
if grep -qi "microsoft" /proc/version 2>/dev/null || [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  IS_WSL=1
fi

POWERCFG_BIN=""
ORIG_POWER_SCHEME=""
ORIG_STANDBY_AC=""
ORIG_HIBERNATE_AC=""
ORIG_LINUX_PROFILE=""

if [[ "$IS_WSL" == "1" ]]; then
  if command -v powercfg.exe >/dev/null 2>&1; then
    POWERCFG_BIN="powercfg.exe"
  elif [[ -x "/mnt/c/Windows/System32/powercfg.exe" ]]; then
    POWERCFG_BIN="/mnt/c/Windows/System32/powercfg.exe"
  fi
fi

setup_power() {
  if [[ "$IS_WSL" == "1" && -n "$POWERCFG_BIN" ]]; then
    echo "==========================================================="
    echo "⚡ [WSL] Configuring Windows 11 Host Power Profile"
    echo "==========================================================="
    # Query current active scheme GUID
    ORIG_POWER_SCHEME=$($POWERCFG_BIN /getactivescheme 2>/dev/null | awk '{print $4}' | tr -d '\r')
    echo "  • Original Active Scheme: ${ORIG_POWER_SCHEME:-Unknown}"

    # Query current AC standby & hibernate setting indices
    ORIG_STANDBY_AC=$($POWERCFG_BIN /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 2>/dev/null | grep -i "Current AC Power Setting Index" | awk '{print $NF}' | tr -d '\r' || true)
    ORIG_HIBERNATE_AC=$($POWERCFG_BIN /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE 2>/dev/null | grep -i "Current AC Power Setting Index" | awk '{print $NF}' | tr -d '\r' || true)

    # Find or duplicate High Performance scheme
    local high_perf_guid=""
    high_perf_guid=$($POWERCFG_BIN /list 2>/dev/null | grep -iE "High performance|Ultimate Performance" | awk '{print $4}' | head -n 1 | tr -d '\r' || true)
    if [[ -z "$high_perf_guid" ]]; then
      high_perf_guid=$($POWERCFG_BIN -duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>/dev/null | awk '{print $4}' | head -n 1 | tr -d '\r' || true)
    fi

    if [[ -n "$high_perf_guid" ]]; then
      $POWERCFG_BIN /setactive "$high_perf_guid" 2>/dev/null || true
      echo "  • Activated Windows High Performance scheme: $high_perf_guid"
    fi

    # Disable sleep and hibernation timeouts while evaluating (0 = never sleep / hibernate)
    $POWERCFG_BIN /change standby-timeout-ac 0 2>/dev/null || true
    $POWERCFG_BIN /change standby-timeout-dc 0 2>/dev/null || true
    $POWERCFG_BIN /change hibernate-timeout-ac 0 2>/dev/null || true
    $POWERCFG_BIN /change hibernate-timeout-dc 0 2>/dev/null || true

    $POWERCFG_BIN /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0 2>/dev/null || true
    $POWERCFG_BIN /setdcvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0 2>/dev/null || true
    $POWERCFG_BIN /setacvalueindex SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE 0 2>/dev/null || true
    $POWERCFG_BIN /setdcvalueindex SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE 0 2>/dev/null || true

    # Unattended sleep timeout (guid: 7bc4a2f9-d8fc-4469-a07b-33eb785aaca0)
    $POWERCFG_BIN /setacvalueindex SCHEME_CURRENT SUB_SLEEP 7bc4a2f9-d8fc-4469-a07b-33eb785aaca0 0 2>/dev/null || true
    $POWERCFG_BIN /setdcvalueindex SCHEME_CURRENT SUB_SLEEP 7bc4a2f9-d8fc-4469-a07b-33eb785aaca0 0 2>/dev/null || true

    $POWERCFG_BIN /setactive SCHEME_CURRENT 2>/dev/null || true
    echo "  • Disabled Windows Standby Sleep (timeout: 0 / disabled)"
    echo "  • Disabled Windows Hibernation (timeout: 0 / disabled)"
    echo "==========================================================="
  fi

  if [[ "$IS_WSL" == "0" ]] && command -v powerprofilesctl >/dev/null 2>&1; then
    ORIG_LINUX_PROFILE=$(powerprofilesctl get 2>/dev/null || true)
    powerprofilesctl set performance 2>/dev/null || true
    echo "⚡ [Linux] Set system power profile to 'performance'"
  fi
}

restore_power() {
  local exit_code=$?
  echo ""
  echo "==========================================================="
  echo "⚡ Restoring Host Power Profile & Sleep/Hibernation Settings"
  echo "==========================================================="
  if [[ "$IS_WSL" == "1" && -n "$POWERCFG_BIN" ]]; then
    if [[ -n "$ORIG_POWER_SCHEME" ]]; then
      $POWERCFG_BIN /setactive "$ORIG_POWER_SCHEME" 2>/dev/null || true
      echo "  • Restored Windows Power Scheme: $ORIG_POWER_SCHEME"
    fi
    if [[ -n "$ORIG_STANDBY_AC" ]]; then
      $POWERCFG_BIN /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE "$ORIG_STANDBY_AC" 2>/dev/null || true
      echo "  • Restored Standby AC setting index: $ORIG_STANDBY_AC"
    fi
    if [[ -n "$ORIG_HIBERNATE_AC" ]]; then
      $POWERCFG_BIN /setacvalueindex SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE "$ORIG_HIBERNATE_AC" 2>/dev/null || true
      echo "  • Restored Hibernate AC setting index: $ORIG_HIBERNATE_AC"
    fi
    $POWERCFG_BIN /setactive SCHEME_CURRENT 2>/dev/null || true
  fi

  if [[ -n "${ORIG_LINUX_PROFILE:-}" ]] && command -v powerprofilesctl >/dev/null 2>&1; then
    powerprofilesctl set "$ORIG_LINUX_PROFILE" 2>/dev/null || true
    echo "  • Restored Linux power profile: $ORIG_LINUX_PROFILE"
  fi
  echo "==========================================================="
  exit "$exit_code"
}

trap restore_power EXIT INT TERM HUP

setup_power

echo "Executing LoRA Policy Evaluation with MUJOCO_GL=$MUJOCO_GL ..."

CMD=("$PYTHON_BIN" "$SCRIPT_DIR/eval_lora_policy.py" "$@")

if [[ "$IS_WSL" == "1" ]]; then
  # On WSL, host power management is handled directly by powercfg.exe
  "${CMD[@]}"
elif command -v systemd-inhibit >/dev/null 2>&1; then
  systemd-inhibit --what=idle:sleep:shutdown --why="SmolVLA LoRA Evaluation" "${CMD[@]}"
elif command -v gnome-session-inhibit >/dev/null 2>&1; then
  gnome-session-inhibit --inhibit suspend:idle --reason "SmolVLA LoRA Evaluation" "${CMD[@]}"
else
  "${CMD[@]}"
fi

