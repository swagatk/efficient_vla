"""Which body poses change between resets to one init state? One process per task (FAIL-004: at most 3 resets per process, here 3)."""
import sys
sys.path.insert(0, ".")
import numpy as np
from vla_core.envs import libero_env_config, make_libero_env, reset_to_init_state
from vla_core.evaluation import seed_everything


def inner(env):
    e = env
    for _ in range(6):
        if hasattr(e, "sim"):
            return e
        e = getattr(e, "_env", None) or getattr(e, "env", None)
    raise AttributeError("no sim")


def poses(env):
    e = inner(env)
    m = e.sim.model
    return {m.body_id2name(i): np.array(m.body_pos[i]) for i in range(m.nbody)}


task, state = int(sys.argv[1]), int(sys.argv[2])
env = make_libero_env(libero_env_config(task, episode_length=520, hard_reset=True))
print("task", task, env.task_description)
seed_everything(7); reset_to_init_state(env, state, seed=7); a = poses(env)
seed_everything(7); reset_to_init_state(env, state, seed=7); b = poses(env)
seed_everything(8); reset_to_init_state(env, state, seed=8); c = poses(env)
for name, other in (("seeded same", b), ("other seed", c)):
    moved = {k: round(float(np.abs(a[k] - other[k]).max()), 4) for k in a if k in other and not np.allclose(a[k], other[k], atol=1e-6)}
    print(f" task {task} state {state} {name}: {len(moved)} of {len(a)} bodies moved, max {max(moved.values(), default=0)}", dict(list(moved.items())[:6]))
env.close()
