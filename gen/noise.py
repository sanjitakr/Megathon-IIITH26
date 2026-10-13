"""Realism noise and missing data (spec 6.5).

observed = simulated x exp(eps), eps ~ N(0, sigma^2); 20-30% of homes lose 0-4 random
months; occasional sensor spikes. The clean value is kept as the reference label.
"""
from __future__ import annotations

import numpy as np

from sim.parameters import sampling_priors


def observe(clean_monthly: np.ndarray, rng: np.random.Generator, drop_months: list[int]) -> list[float | None]:
    obs = sampling_priors()["observation"]
    noisy = clean_monthly * np.exp(rng.normal(0, obs["noise_sigma"], size=12))
    spikes = rng.random(12) < obs["spike_probability"]
    noisy = np.where(spikes, noisy * rng.uniform(2.0, 4.0, size=12), noisy)
    out: list[float | None] = [round(float(x), 1) for x in noisy]
    for m in drop_months:
        out[m] = None
    return out


def missing_months(rng: np.random.Generator) -> list[int]:
    obs = sampling_priors()["observation"]
    if rng.random() >= obs["missing_share"]:
        return []
    k = int(rng.integers(0, obs["max_missing_months"] + 1))
    return sorted(rng.choice(12, size=k, replace=False).tolist())
