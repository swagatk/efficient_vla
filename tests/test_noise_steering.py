import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import noise_steering  # noqa: E402
from noise_steering import executed_shape, install, pooled_features, uninstall  # noqa: E402

CHUNK, MAX_DIM, ACT_DIM, N_EXEC = 6, 5, 3, 2
REAL_PREFIX_FEATURES = noise_steering.prefix_features


class FakeModel:
    """sample_actions draws noise with sample_noise when given none, like SmolVLA, and returns a chunk that
    depends on every noise entry and on the observation."""

    def __init__(self):
        self.config = SimpleNamespace(chunk_size=CHUNK, max_action_dim=MAX_DIM, use_cache=True)
        self.received = []

    def sample_noise(self, shape, device):
        return torch.normal(0.0, 1.0, size=shape, device=device)

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        if noise is None:
            noise = self.sample_noise((state.shape[0], CHUNK, MAX_DIM), state.device)
        self.received.append(noise.clone())
        return noise.cumsum(dim=1) + state.sum() + images[0].mean()


def fake_policy(n_action_steps=N_EXEC):
    cfg = SimpleNamespace(n_action_steps=n_action_steps, chunk_size=CHUNK, action_feature=SimpleNamespace(shape=(ACT_DIM,)))
    return SimpleNamespace(model=FakeModel(), config=cfg)


def inputs():
    return [torch.rand(1, 3, 4, 4)], [torch.ones(1, dtype=torch.bool)], torch.zeros(1, 3, dtype=torch.long), \
        torch.ones(1, 3), torch.rand(1, 32)


@pytest.fixture(autouse=True)
def fake_prefix(monkeypatch):
    # The real prefix pass needs SmolVLA; here it returns the state and must not use random numbers.
    monkeypatch.setattr(noise_steering, "prefix_features", lambda model, i, m, t, lm, state, d: state[:, :d].float())


def test_callback_returning_none_reproduces_the_base_chunk_exactly():
    policy, args = fake_policy(), inputs()
    torch.manual_seed(3)
    base = policy.model.sample_actions(*args)
    seen = []
    install(policy, lambda feats, drawn: seen.append((feats, drawn.clone())))
    torch.manual_seed(3)
    steered = policy.model.sample_actions(*args)
    assert torch.equal(steered, base)
    feats, drawn = seen[0]
    assert torch.equal(feats, args[4][:, :8]) and drawn.shape == (N_EXEC, ACT_DIM)
    assert torch.equal(drawn, policy.model.received[0][0, :N_EXEC, :ACT_DIM])


def test_chosen_noise_replaces_only_the_executed_real_part():
    policy, args = fake_policy(), inputs()
    torch.manual_seed(5)
    policy.model.sample_actions(*args)
    drawn_base = policy.model.received[-1]
    chosen = torch.full((N_EXEC, ACT_DIM), 7.0)
    st = install(policy, lambda feats, drawn: chosen)
    torch.manual_seed(5)
    policy.model.sample_actions(*args)
    used = policy.model.received[-1]
    assert torch.equal(used[0, :N_EXEC, :ACT_DIM], chosen)
    mask = torch.ones_like(used, dtype=torch.bool)
    mask[0, :N_EXEC, :ACT_DIM] = False
    assert torch.equal(used[mask], drawn_base[mask])         # everything else is the base draw
    assert st.n_inferences == 1 and (st.n_exec, st.action_dim) == (N_EXEC, ACT_DIM)


def test_wrong_shape_extra_noise_batch_and_double_install_raise_and_uninstall_restores():
    policy, args = fake_policy(), inputs()
    install(policy, lambda feats, drawn: torch.zeros(N_EXEC + 1, ACT_DIM))
    with pytest.raises(ValueError):
        policy.model.sample_actions(*args)
    with pytest.raises(ValueError):
        policy.model.sample_actions(*args, noise=torch.zeros(1, CHUNK, MAX_DIM))
    with pytest.raises(ValueError):
        images, m, t, lm, state = args
        policy.model.sample_actions(images, m, t, lm, state.expand(2, -1))
    with pytest.raises(RuntimeError):
        install(policy, lambda f, d: None)
    uninstall(policy)
    assert "sample_actions" not in vars(policy.model)
    uninstall(policy)  # no-op


def test_executed_shape_caps_at_the_chunk():
    assert executed_shape(fake_policy(2)) == (2, ACT_DIM)
    assert executed_shape(fake_policy(CHUNK + 4)) == (CHUNK, ACT_DIM)


def test_pooled_features_average_only_valid_tokens_then_append_the_state():
    out = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    state = torch.tensor([[5.0, 6.0, 7.0]])
    f = pooled_features(out.to(torch.bfloat16), mask, state, state_dim=2)
    assert f.dtype == torch.float32 and torch.equal(f, torch.tensor([[2.0, 3.0, 5.0, 6.0]]))
    one = pooled_features(out, torch.tensor([[False, True, False]]), state, state_dim=1)
    assert torch.equal(one, torch.tensor([[3.0, 4.0, 5.0]]))       # a single valid token is its own mean


class FakeVLM:
    def __init__(self):
        self.calls = []

    def forward(self, **kw):
        self.calls.append(kw)
        return [kw["inputs_embeds"][0] * 2, "expert"], "cache"


def test_prefix_pass_uses_the_arguments_of_sample_actions():
    # SmolVLA's sample_actions: position ids = cumsum(pad_masks) - 1, inputs_embeds=[prefix, None],
    # past_key_values=None, use_cache from the config; the features pool its first output.
    embs = torch.arange(12.0).reshape(1, 3, 4)
    pad = torch.tensor([[True, False, True]])
    att = torch.tensor([[False, False, True]])
    model = SimpleNamespace(embed_prefix=lambda i, m, t, lm, state: (embs, pad, att), vlm_with_expert=FakeVLM(),
                            config=SimpleNamespace(use_cache=True))
    state = torch.tensor([[9.0, 8.0, 7.0]])
    f = REAL_PREFIX_FEATURES(model, None, None, None, None, state, 2)
    kw = model.vlm_with_expert.calls[0]
    assert torch.equal(kw["position_ids"], torch.tensor([[0, 0, 1]]))
    assert kw["past_key_values"] is None and kw["use_cache"] is True and kw["inputs_embeds"][1] is None
    assert kw["attention_mask"].shape == (1, 3, 3) and not kw["attention_mask"][0, 0, 2]   # earlier token cannot see a later block
    assert torch.equal(f, torch.tensor([[8.0, 10.0, 12.0, 14.0, 9.0, 8.0]]))   # 2 * mean of tokens 0 and 2
