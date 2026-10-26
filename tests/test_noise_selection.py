import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

from noise_selection import (  # noqa: E402
    acceleration_score,
    install,
    medoid_score,
    parse_policy_label,
    select_index,
    uninstall,
)

CHUNK, MAX_DIM, ACT_DIM, STEPS = 6, 5, 3, 4


class FakeModel(torch.nn.Module):
    """Same call structure as SmolVLA's VLAFlowMatching.sample_actions: noise from sample_noise, Euler steps
    through self.denoise_step(x_t=..., ...), the velocity depending on the observation."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=CHUNK, max_action_dim=MAX_DIM, num_steps=STEPS)
        self.w = torch.nn.Linear(MAX_DIM, MAX_DIM)

    def sample_noise(self, shape, device):
        return torch.normal(0.0, 1.0, size=shape, device=device)

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        if noise is None:
            noise = self.sample_noise((state.shape[0], CHUNK, MAX_DIM), state.device)
        x = noise
        for k in range(STEPS):
            t = torch.full((x.shape[0],), 1.0 - k / STEPS)
            x = x - self.denoise_step(x_t=x, prefix_pad_masks=img_masks[0], past_key_values=None, timestep=t) / STEPS
        return x

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        obs = (self._images[0].mean(dim=(1, 2, 3)) + self._state.sum(1))[:, None, None]
        return torch.tanh(self.w(x_t)) * timestep[:, None, None] + obs


def fake_policy():
    torch.manual_seed(0)
    model = FakeModel()
    cfg = SimpleNamespace(n_action_steps=2, chunk_size=CHUNK, action_feature=SimpleNamespace(shape=(ACT_DIM,)))
    return SimpleNamespace(model=model, config=cfg)


def inputs():
    images, state = [torch.rand(1, 3, 4, 4)], torch.rand(1, 2)
    return images, [torch.ones(1, dtype=torch.bool)], torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3), state


def call(policy, args):
    # The fake denoise step reads the observation from the model, repeated as the wrapper repeats it.
    images, _, _, _, state = args
    n = policy.model._n = getattr(policy.model, "_n", 1)
    policy.model._images, policy.model._state = [images[0].expand(n, -1, -1, -1)], state.expand(n, -1)
    return policy.model.sample_actions(*args)


def test_acceleration_score_is_zero_for_a_straight_flow_and_ignores_unexecuted_parts():
    const = torch.randn(2, CHUNK, MAX_DIM)
    vel = [const.clone() for _ in range(STEPS)]
    assert torch.equal(acceleration_score(vel, 2, ACT_DIM), torch.zeros(2))
    vel[1][1, 0, 0] += 1.0                                # sample 1 bends once, inside the executed part
    vel[2][0, 3, 0] += 5.0                                # sample 0 bends only after n_exec
    vel[2][0, 0, ACT_DIM] += 5.0                          # ... and in a padded dimension
    s = acceleration_score(vel, 2, ACT_DIM)
    assert s[0] == 0 and s[1] == pytest.approx(2.0)      # +1 then -1 at the next step
    assert select_index("accel", None, vel, 2, ACT_DIM) == 0


def test_medoid_picks_the_central_chunk_on_executed_actions():
    chunks = torch.zeros(3, CHUNK, MAX_DIM)
    chunks[0, :2, :ACT_DIM] = -1.0
    chunks[2, :2, :ACT_DIM] = 1.0
    chunks[1, 2:, :] = 100.0                              # far away only where nothing is executed
    assert int(medoid_score(chunks, 2, ACT_DIM).argmin()) == 1
    assert select_index("medoid", chunks, [], 2, ACT_DIM) == 1
    assert select_index("first", chunks, [], 2, ACT_DIM) == 0


def test_one_sample_reproduces_the_base_policy_exactly():
    policy, args = fake_policy(), inputs()
    torch.manual_seed(7)
    base = call(policy, args)
    sel = install(policy, 1, "medoid")
    torch.manual_seed(7)
    out = call(policy, args)
    assert torch.equal(out, base) and sel.chosen == [0]


def test_n_samples_return_the_chosen_row_of_one_batched_call_and_uninstall_restores():
    policy, args = fake_policy(), inputs()
    original = policy.model.sample_actions
    policy.model._n = 4
    sel = install(policy, 4, "medoid")
    torch.manual_seed(3)
    out = call(policy, args)
    assert out.shape == (1, CHUNK, MAX_DIM) and len(sel.chosen) == 1
    assert "denoise_step" not in vars(policy.model)      # the recording hook is removed after each call
    with pytest.raises(RuntimeError):
        install(policy, 4, "accel")
    uninstall(policy)
    assert policy.model.sample_actions == original
    # The same noise through the unwrapped model, batched: the selector returned the medoid row.
    torch.manual_seed(3)
    noise = policy.model.sample_noise((4, CHUNK, MAX_DIM), "cpu")
    images, masks, tokens, lmask, state = args
    full = policy.model.sample_actions([images[0].expand(4, -1, -1, -1)], [masks[0].expand(4)],
                                       tokens.expand(4, -1), lmask.expand(4, -1), state.expand(4, -1), noise=noise)
    i = int(medoid_score(full, 2, ACT_DIM).argmin())
    assert sel.chosen == [i] and torch.allclose(out[0], full[i])


def test_accel_selection_matches_recorded_velocities():
    policy, args = fake_policy(), inputs()
    policy.model._n = 5
    sel = install(policy, 5, "accel")
    torch.manual_seed(11)
    call(policy, args)
    uninstall(policy)
    torch.manual_seed(11)
    noise = policy.model.sample_noise((5, CHUNK, MAX_DIM), "cpu")
    vel, step = [], policy.model.denoise_step
    policy.model.denoise_step = lambda **kw: vel.append(step(**kw)) or vel[-1]
    images, masks, tokens, lmask, state = args
    policy.model.sample_actions([images[0].expand(5, -1, -1, -1)], [masks[0].expand(5)], tokens.expand(5, -1),
                                lmask.expand(5, -1), state.expand(5, -1), noise=noise)
    assert sel.chosen == [int(acceleration_score(vel, 2, ACT_DIM).argmin())]


def test_labels_and_bad_arguments():
    assert parse_policy_label("base") is None and parse_policy_label("accel_n8") == (8, "accel")
    for bad in ("medoid", "foo_n8", "medoid_n0", "medoid_nx"):
        with pytest.raises(ValueError):
            parse_policy_label(bad)
    with pytest.raises(ValueError):
        install(fake_policy(), 0, "medoid")
    with pytest.raises(ValueError):
        install(fake_policy(), 2, "best")
