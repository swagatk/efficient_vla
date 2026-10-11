"""Seed derivation shared by evaluation entry points."""

# Larger than any init-state index, so different noise seeds never share a torch seed.
NOISE_SEED_STRIDE = 1_000_003


def episode_noise_seed(init_state_seed: int, noise_seed: int = 0) -> int:
    """Torch seed for the policy's sampling noise in one evaluation episode.

    ``init_state_seed`` picks the LIBERO init state; ``noise_seed`` picks an independent
    repeat of the evaluation on that same init state. ``noise_seed=0`` returns
    ``init_state_seed`` unchanged, which is what evaluations ran with before the flag existed.
    """
    if init_state_seed < 0 or noise_seed < 0:
        raise ValueError(f"seeds must be non-negative, got init_state_seed={init_state_seed}, noise_seed={noise_seed}")
    if init_state_seed >= NOISE_SEED_STRIDE:
        raise ValueError(f"init_state_seed must be below {NOISE_SEED_STRIDE}, got {init_state_seed}")
    return init_state_seed + noise_seed * NOISE_SEED_STRIDE
