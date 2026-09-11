"""Load / save trained PPO parameters and build a plain-numpy policy.

The numpy policy mirrors what the browser viewer runs: observation
normalisation -> MLP (SiLU) -> tanh-squashed mean action.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import numpy as np


def save_params(params, path: Path) -> None:
    path.write_bytes(pickle.dumps(jax.device_get(params)))


def load_params(path: Path):
    return pickle.loads(Path(path).read_bytes())


def export_numpy(params, action_size: int) -> dict:
    """Flatten Brax PPO params into a JSON-friendly dict of plain arrays."""
    normalizer, policy = params[0], params[1]
    layers = policy["params"]
    keys = sorted(layers.keys(), key=lambda k: int(k.split("_")[-1]))
    out = {
        "obs_mean": np.asarray(normalizer.mean).tolist(),
        "obs_std": np.asarray(normalizer.std).tolist(),
        "action_size": action_size,
        "layers": [{"w": np.asarray(layers[k]["kernel"]).tolist(),
                    "b": np.asarray(layers[k]["bias"]).tolist()} for k in keys],
    }
    return out


class NumpyPolicy:
    """Brax tanh-normal policy evaluated with numpy.

    ``stochastic=False`` returns the distribution mode tanh(loc); ``True``
    samples tanh(loc + std * N(0, 1)) exactly like Brax does during training
    (std = softplus(raw) + min_std with min_std = 0.001).
    """

    def __init__(self, spec: dict, stochastic: bool = False, seed: int = 0):
        self.mean = np.asarray(spec["obs_mean"], np.float32)
        self.std = np.asarray(spec["obs_std"], np.float32)
        self.layers = [(np.asarray(l["w"], np.float32), np.asarray(l["b"], np.float32)) for l in spec["layers"]]
        self.action_size = spec["action_size"]
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / (self.std + 1e-8)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = x / (1.0 + np.exp(-x))  # SiLU
        loc = x[: self.action_size]
        if not self.stochastic:
            return np.tanh(loc)
        std = np.log1p(np.exp(x[self.action_size:])) + 0.001
        return np.tanh(loc + std * self.rng.normal(size=loc.shape))

