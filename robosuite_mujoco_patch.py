"""
Compatibility patch for Robosuite, MuJoCo 3.x, NumPy 2.x, and PyTorch 2.6+ in lerobot_env.
Fixes:
1. PyTorch 2.6+ torch.load weights_only=False default for LIBERO init states.
2. Robosuite binding_utils joint_type assertion error under MuJoCo 3.x / NumPy 2.x.
3. Robosuite MjData qM property renamed to M in MuJoCo 3.10+.
4. MuJoCo mj_fullM API argument ordering between Robosuite and MuJoCo 3.x.
5. PyAV missing Option namespace.
"""

import types

# 1. PyTorch 2.6+ weights_only patch
try:
    import torch
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
except ImportError:
    pass

# 2. PyAV option patch
try:
    import av
    if not hasattr(av, "option"):
        av.option = types.SimpleNamespace(Option=object)
except ImportError:
    pass

# 3. Robosuite / NumPy 2.x & MuJoCo 3.x compatibility patch
try:
    import mujoco
    import numpy as np
    import robosuite.utils.binding_utils as bu

    def _create_patched_qpos(orig_fn):
        def _patched(self, name):
            joint_id = self.joint_name2id(name)
            if joint_id < 0:
                for alt_name in [f"robot0_{name}", name.replace("robot0_", "")]:
                    alt_id = self.joint_name2id(alt_name)
                    if alt_id >= 0:
                        joint_id = alt_id
                        break
            if joint_id < 0:
                raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
            joint_type = int(self.jnt_type[joint_id])
            joint_addr = int(self.jnt_qposadr[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                ndim = 7
            elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                ndim = 4
            else:
                assert joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)), \
                    f"Joint '{name}' (id {joint_id}) has unsupported type {joint_type}"
                ndim = 1

            if ndim == 1:
                return joint_addr
            return (joint_addr, joint_addr + ndim)
        return _patched

    def _create_patched_qvel(orig_fn):
        def _patched(self, name):
            joint_id = self.joint_name2id(name)
            if joint_id < 0:
                for alt_name in [f"robot0_{name}", name.replace("robot0_", "")]:
                    alt_id = self.joint_name2id(alt_name)
                    if alt_id >= 0:
                        joint_id = alt_id
                        break
            if joint_id < 0:
                raise KeyError(f"Joint '{name}' not found in MuJoCo model.")
            joint_type = int(self.jnt_type[joint_id])
            joint_addr = int(self.jnt_dofadr[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                ndim = 6
            elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                ndim = 3
            else:
                assert joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)), \
                    f"Joint '{name}' (id {joint_id}) has unsupported type {joint_type}"
                ndim = 1

            if ndim == 1:
                return joint_addr
            return (joint_addr, joint_addr + ndim)
        return _patched

    for target in [bu, getattr(bu, "MjModel", None), getattr(bu, "MjModelWrapper", None)]:
        if target is not None:
            if hasattr(target, "get_joint_qpos_addr"):
                target.get_joint_qpos_addr = _create_patched_qpos(target.get_joint_qpos_addr)
            if hasattr(target, "get_joint_qvel_addr"):
                target.get_joint_qvel_addr = _create_patched_qvel(target.get_joint_qvel_addr)

    # Fix for MuJoCo 3.10+ where qM was renamed to M in MjData
    class _MArray(np.ndarray):
        """NumPy array view that retains a reference to the MjData struct."""
        _mj_data = None

    def _get_qM(self):
        d = getattr(self, "_data", self)
        raw = getattr(d, "M", None)
        if raw is None:
            raw = getattr(d, "qM", None)
        if raw is None:
            raise AttributeError("'MjData' object has no attribute 'qM' or 'M'")
        arr = np.asarray(raw).view(_MArray)
        arr._mj_data = d
        return arr

    for data_cls in [getattr(bu, "MjData", None), getattr(bu, "MjDataWrapper", None), getattr(mujoco, "MjData", None)]:
        if data_cls is not None:
            try:
                setattr(data_cls, "qM", property(_get_qM))
            except Exception:
                pass

    # Safety wrapper for mj_fullM compatible with robosuite 1.4 + MuJoCo 3.x
    _orig_mj_fullM = mujoco.mj_fullM

    def _unpack_M_fallback(m, dst, M):
        nv = int(m.nv)
        dst.fill(0.0)
        for i in range(nv):
            adr = int(m.dof_Madr[i])
            j = i
            while j >= 0:
                val = float(M[adr])
                dst[i, j] = val
                dst[j, i] = val
                adr += 1
                j = int(m.dof_parentid[j])

    def _patched_mj_fullM(*args, **kwargs):
        if len(args) == 3:
            m, a1, a2 = args
            model = getattr(m, "_model", m)

            # Case 1: Called as mj_fullM(model, data, dst) [Standard MuJoCo 3.x]
            if not isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
                dst = a2
                d = getattr(a1, "_data", a1)
                raw_M = getattr(d, "M", getattr(d, "qM", None))
                if raw_M is not None:
                    return _unpack_M_fallback(model, dst, raw_M)
                try:
                    return _orig_mj_fullM(model, d, dst)
                except Exception:
                    pass

            # Case 2: Called as mj_fullM(model, dst, qM_or_data) [Robosuite 1.4 API]
            if isinstance(a1, np.ndarray):
                dst = a1
                qM_or_data = a2
                raw_M = getattr(qM_or_data, "M", getattr(qM_or_data, "qM", qM_or_data))
                if isinstance(raw_M, np.ndarray):
                    return _unpack_M_fallback(model, dst, raw_M)
                d = getattr(qM_or_data, "_mj_data", getattr(qM_or_data, "_data", None))
                if d is not None:
                    raw_M = getattr(d, "M", getattr(d, "qM", None))
                    if raw_M is not None:
                        return _unpack_M_fallback(model, dst, raw_M)

        elif len(args) == 2:
            m, a1 = args
            model = getattr(m, "_model", m)
            if hasattr(a1, "_mj_data") or hasattr(a1, "_data") or hasattr(a1, "qpos"):
                d = getattr(a1, "_mj_data", getattr(a1, "_data", a1))
                return _orig_mj_fullM(model, d)

        return _orig_mj_fullM(*args, **kwargs)

    mujoco.mj_fullM = _patched_mj_fullM
except Exception:
    pass
