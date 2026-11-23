"""Step 17: record SmolVLA's prefix features at an episode's first inference (README.md, step 17).

``install_first_features(policy, store)`` wraps the sampler wrapper already installed on that policy instance
(deep_noise_search's fixed-noise wrapper or a noise process): at the first call it stores
``noise_steering.prefix_features`` (the VLM prefix output averaged over valid tokens, plus the first ``state_dim``
entries of the normalised state) as float32 under ``store["features"]``, then hands every call, the first included, to
the inner wrapper unchanged. The prefix pass draws no random numbers and does not change the model, so the episode is
the one it would be without the recorder; it costs one extra prefix pass per episode. Uninstall with
``deep_noise_search.uninstall_fixed_noise``, which removes the recorder and the inner wrapper together.
"""
from __future__ import annotations

import json
from pathlib import Path

STATE_DIM = 8   # the normalised state entries kept, as in steps 3 and 5b


def install_first_features(policy, store: dict, state_dim: int = STATE_DIM) -> None:
    model = policy.model
    inner = vars(model).get("sample_actions")
    if inner is None:
        raise RuntimeError("install the noise wrapper first: the recorder wraps it")
    if getattr(inner, "_records_first_features", False):
        raise RuntimeError("a feature recorder is already installed on this policy")

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if "features" not in store:
            import torch

            from noise_steering import prefix_features

            with torch.no_grad():
                f = prefix_features(model, images, img_masks, lang_tokens, lang_masks, state, state_dim)
            if f.shape[0] != 1:
                raise ValueError(f"the recorder expects one observation per call, got batch {f.shape[0]}")
            store["features"] = f[0].detach().to("cpu", torch.float32).numpy()
        store["calls"] = store.get("calls", 0) + 1
        return inner(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)

    sample_actions._records_first_features = True
    model.sample_actions = sample_actions


def feature_file(feature_dir, state: int, noise_id: int) -> Path:
    return Path(feature_dir) / f"features_state{state}_id{noise_id}.json"


def write_features(path, store: dict, meta: dict) -> None:
    """One episode's first-inference features as JSON (float32 values written exactly; the results zip leaves npz
    out). Refuses a store whose first inference never ran."""
    if "features" not in store:
        raise ValueError("no features recorded: the episode made no inference")
    f = store["features"]
    rec = {**meta, "dim": int(f.shape[0]), "dtype": "float32", "inferences": int(store["calls"]),
           "features": [float(x) for x in f]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(rec) + "\n")
    tmp.replace(path)


def read_features(path) -> tuple[dict, list[float]]:
    rec = json.loads(Path(path).read_text())
    feats = rec.pop("features")
    if len(feats) != rec["dim"]:
        raise ValueError(f"{path}: {len(feats)} values, dim {rec['dim']}")
    return rec, feats
