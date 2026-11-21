"""M1's noise processes: rho = 1 is exactly `repeat`, rho = 0 redraws it, the AR(1) is stationary with the set
correlation, sigma scales fresh noise, zero is zero; determinism per id, refusals, the wrapper, and deep_noise_search's
flags, file names and unchanged fixed families."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))
import deep_noise_search as dns  # noqa: E402
import noise_processes as npr  # noqa: E402


class FakeModel:
    def __init__(self, chunk=6, dim=5):
        self.config = SimpleNamespace(chunk_size=chunk, max_action_dim=dim)
        self.received = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        self.received.append(noise)
        return noise


def fake_policy(chunk=6, dim=5):
    m = FakeModel(chunk, dim)
    return SimpleNamespace(model=m, config=m.config)


def seq(proc, n):
    return [proc.next() for _ in range(n)]


def test_rho_one_is_exactly_repeat_at_every_inference():
    want = dns.make_noise("repeat", 7, 50, 32)
    got = seq(npr.NoiseProcess("ar_repeat", 7, 50, 32, rho=1.0), 6)
    assert all(torch.equal(z, want) for z in got)


def test_rho_zero_starts_at_repeat_then_redraws_along_the_chunk():
    got = seq(npr.NoiseProcess("ar_repeat", 7, 50, 32, rho=0.0), 4)
    assert torch.equal(got[0], dns.make_noise("repeat", 7, 50, 32))
    for z in got:
        assert z.shape == (1, 50, 32) and z.dtype == torch.float32
        assert all(torch.equal(z[0, 0], z[0, k]) for k in range(50))        # repeated along the chunk
    assert not any(torch.equal(got[i], got[i + 1]) for i in range(3))
    # rho = 0: u_j is the innovation itself
    g = torch.Generator().manual_seed(npr.AR_SEED_BASE + 7)
    assert torch.equal(got[1][0, 0], torch.randn(32, generator=g))


def test_ar_is_stationary_with_the_set_correlation():
    rho, ids, steps = 0.6, range(400), 8
    xs = torch.stack([torch.stack([z[0, 0] for z in seq(npr.NoiseProcess("ar_repeat", k, 4, 32, rho=rho), steps)])
                      for k in ids])                                         # (ids, steps, 32)
    for j in (0, 3, 7):
        assert 0.9 < float(xs[:, j].var()) < 1.1
    lag1 = float((xs[:, 1:] * xs[:, :-1]).mean())
    lag2 = float((xs[:, 2:] * xs[:, :-2]).mean())
    assert abs(lag1 - rho) < 0.05 and abs(lag2 - rho ** 2) < 0.05


def test_processes_are_deterministic_per_id_and_differ_between_ids():
    for fam, kw in (("ar_repeat", {"rho": 0.5}), ("scaled", {"sigma": 0.5})):
        a = seq(npr.NoiseProcess(fam, 3, 6, 5, **kw), 3)
        b = seq(npr.NoiseProcess(fam, 3, 6, 5, **kw), 3)
        c = seq(npr.NoiseProcess(fam, 4, 6, 5, **kw), 3)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert not torch.equal(a[1], c[1])


def test_scaled_is_fresh_full_noise_times_sigma():
    got = seq(npr.NoiseProcess("scaled", 2, 50, 32, sigma=0.5), 3)
    g = torch.Generator().manual_seed(npr.SCALED_SEED_BASE + 2)
    assert torch.equal(got[0], 0.5 * torch.randn(1, 50, 32, generator=g))
    assert not torch.equal(got[0], got[1]) and not torch.equal(got[0][0, 0], got[0][0, 1])   # full, fresh
    big = torch.cat([npr.NoiseProcess("scaled", k, 50, 32, sigma=0.5).next() for k in range(40)])
    assert 0.48 < float(big.std()) < 0.52


def test_zero_is_zero_and_calls_are_counted():
    p = npr.NoiseProcess("zero", 0, 6, 5)
    got = seq(p, 3)
    assert all(torch.equal(z, torch.zeros(1, 6, 5)) for z in got) and p.calls == 3


@pytest.mark.parametrize("family, rho, sigma", [
    ("ar_repeat", None, None), ("ar_repeat", 1.2, None), ("ar_repeat", -0.1, None), ("ar_repeat", 0.5, 1.0),
    ("scaled", None, None), ("scaled", None, 0.0), ("scaled", None, -1.0), ("scaled", None, float("inf")),
    ("scaled", 0.5, 0.5), ("zero", 0.5, None), ("zero", None, 0.5), ("repeat", 0.5, None), ("sampled", None, 0.5)])
def test_parameters_are_refused_where_they_do_not_belong(family, rho, sigma):
    with pytest.raises(ValueError):
        npr.check_params(family, rho, sigma)


def test_process_refuses_a_fixed_family():
    with pytest.raises(ValueError):
        npr.NoiseProcess("repeat", 0, 6, 5)


def test_labels():
    assert npr.family_label("repeat") == "repeat" and npr.family_label("sampled") == "sampled"
    assert npr.family_label("ar_repeat", rho=0.5) == "ar_repeat_rho0.5"
    assert npr.family_label("ar_repeat", rho=1.0) == "ar_repeat_rho1"
    assert npr.family_label("ar_repeat", rho=0.0) == "ar_repeat_rho0"
    assert npr.family_label("scaled", sigma=0.5) == "scaled_sigma0.5" and npr.family_label("zero") == "zero"


def test_wrapper_feeds_the_sequence_and_draws_no_global_random_numbers():
    p = fake_policy()
    proc = npr.NoiseProcess("ar_repeat", 1, 6, 5, rho=0.3)
    want = seq(npr.NoiseProcess("ar_repeat", 1, 6, 5, rho=0.3), 3)
    npr.install_noise_process(p, proc)
    torch.manual_seed(0)
    before = torch.rand(1)
    torch.manual_seed(0)
    for _ in range(3):
        p.model.sample_actions(None, None, None, None, torch.zeros(1, 8))
    assert torch.equal(torch.rand(1), before)
    assert all(torch.equal(a, b) for a, b in zip(p.model.received, want)) and proc.calls == 3


def test_wrapper_refusals_and_uninstall():
    p = fake_policy()
    with pytest.raises(ValueError, match="shape"):
        npr.install_noise_process(p, npr.NoiseProcess("zero", 0, 5, 5))
    npr.install_noise_process(p, npr.NoiseProcess("zero", 0, 6, 5))
    with pytest.raises(RuntimeError):
        npr.install_noise_process(p, npr.NoiseProcess("zero", 0, 6, 5))
    with pytest.raises(RuntimeError):
        dns.install_fixed_noise(p, torch.zeros(1, 6, 5))
    with pytest.raises(ValueError, match="supplies"):
        p.model.sample_actions(None, None, None, None, torch.zeros(1, 8), noise=torch.zeros(1, 6, 5))
    with pytest.raises(ValueError, match="one observation"):
        p.model.sample_actions(None, None, None, None, torch.zeros(2, 8))
    dns.uninstall_fixed_noise(p)
    assert "sample_actions" not in vars(p.model)


# ---- deep_noise_search ----------------------------------------------------------------------------------------

BASE = ["--init_state", "3", "--noise_ids", "0-9", "--output_dir", "/tmp/x"]


def test_cli_accepts_the_new_families_and_keeps_the_old_ones():
    assert dns.FAMILIES == ("sampled", "full", "repeat")
    a = dns.parse_args(["--family", "ar_repeat", "--rho", "0.5"] + BASE)
    assert (a.rho, a.sigma) == (0.5, None)
    assert dns.parse_args(["--family", "scaled", "--sigma", "0.5"] + BASE).sigma == 0.5
    assert dns.parse_args(["--family", "zero"] + BASE).rho is None
    old = dns.parse_args(["--family", "repeat"] + BASE)
    assert (old.rho, old.sigma) == (None, None)


@pytest.mark.parametrize("extra", [["--family", "ar_repeat"], ["--family", "repeat", "--rho", "0.5"],
                                   ["--family", "scaled", "--rho", "0.5"],
                                   ["--family", "ar_repeat", "--rho", "0.5", "--jitter_seeds", "5000-5003"]])
def test_cli_refuses_wrong_parameters(extra):
    with pytest.raises(SystemExit):
        dns.parse_args(extra + BASE)


def test_install_family_dispatch():
    for fam, kw, want in (("repeat", {}, dns.make_noise("repeat", 4, 6, 5)),
                          ("ar_repeat", {"rho": 1.0}, dns.make_noise("repeat", 4, 6, 5)),
                          ("full", {}, dns.make_noise("full", 4, 6, 5)),
                          ("zero", {}, torch.zeros(1, 6, 5))):
        p = fake_policy()
        args = SimpleNamespace(family=fam, rho=kw.get("rho"), sigma=None)
        dns.install_family(p, args, 4, 1004, False)
        p.model.sample_actions(None, None, None, None, torch.zeros(1, 8))
        p.model.sample_actions(None, None, None, None, torch.zeros(1, 8))
        assert all(torch.equal(z, want) for z in p.model.received)
    p = fake_policy()
    dns.install_family(p, SimpleNamespace(family="repeat", rho=None, sigma=None), 4, 5000, True)   # jitter seed
    p.model.sample_actions(None, None, None, None, torch.zeros(1, 8))
    assert torch.equal(p.model.received[0], dns.make_noise("repeat", 4, 6, 5, 5000))
    p = fake_policy()
    dns.install_family(p, SimpleNamespace(family="sampled", rho=None, sigma=None), 4, 1004, False)
    assert "sample_actions" not in vars(p.model)


def test_result_files_are_named_by_label(tmp_path):
    assert dns.result_file(tmp_path, npr.family_label("ar_repeat", 0.5), 3, [0], None).name == \
        "per_episode_ar_repeat_rho0.5_state3.csv"
    assert dns.result_file(tmp_path, "repeat", 3, [0], None).name == "per_episode_repeat_state3.csv"


def test_record_meta_carries_rho_and_sigma():
    args = SimpleNamespace(task_id=0, init_state=3, policy_path="p", n_action_steps=10, rename_map={}, family="ar_repeat",
                           lora_dir=None, rho=0.5, sigma=None)
    m = dns.record_meta(args, "t", 1, None, 1001, {"success": True, "steps": 9})
    assert (m["noise_family"], m["noise_rho"], m["noise_sigma"]) == ("ar_repeat", 0.5, None)
