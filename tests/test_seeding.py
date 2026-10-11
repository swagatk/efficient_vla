import pytest
import torch

from vla_core.seeding import episode_noise_seed


def test_default_noise_seed_keeps_legacy_seed():
    # Results recorded before --eval_noise_seed existed used the init-state index directly.
    for init_seed in (0, 1, 37, 49):
        assert episode_noise_seed(init_seed) == init_seed
        assert episode_noise_seed(init_seed, 0) == init_seed


def test_seeds_are_unique_across_init_states_and_noise_seeds():
    seeds = {episode_noise_seed(i, n) for i in range(50) for n in range(4)}
    assert len(seeds) == 50 * 4


def test_noise_seed_changes_the_sampled_noise_and_is_reproducible():
    def draw(noise_seed):
        torch.manual_seed(episode_noise_seed(7, noise_seed))
        return torch.randn(16, 7)

    assert torch.equal(draw(1), draw(1))
    assert not torch.equal(draw(0), draw(1))
    assert not torch.equal(draw(1), draw(2))


@pytest.mark.parametrize("init_seed,noise_seed", [(-1, 0), (0, -1), (1_000_003, 0)])
def test_invalid_seeds_are_rejected(init_seed, noise_seed):
    with pytest.raises(ValueError):
        episode_noise_seed(init_seed, noise_seed)
