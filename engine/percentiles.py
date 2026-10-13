"""Reference distributions for the suitability index (spec 4.1.1).

P_saving and P_value are percentiles of a retrofit's standalone % saving in purchased
energy, and of saving per GBP, among simulated *training* homes where it is feasible.
gen/build_dataset.py writes data/processed/retrofit_reference.json from the training
split; if it is absent a small reference sample is simulated once and cached.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from sim.parameters import PROCESSED_DIR, retrofit_ids

REFERENCE_PATH = PROCESSED_DIR / "retrofit_reference.json"


@lru_cache(maxsize=1)
def reference() -> dict:
    if REFERENCE_PATH.exists():
        with open(REFERENCE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    from gen.labels import build_retrofit_reference

    ref = build_retrofit_reference(n_homes=600, seed=7)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_PATH, "w", encoding="utf-8") as fh:
        json.dump(ref, fh)
    return ref


def percentile(retrofit_id: str, metric: str, value: float) -> float:
    """Share of reference homes with a value at or below this one (0..1)."""
    ref = reference().get(retrofit_id, {}).get(metric)
    if not ref:
        return 0.5
    arr = np.asarray(ref)
    return float(np.searchsorted(arr, value, side="right") / len(arr))


def known_retrofits() -> list[str]:
    return [r for r in retrofit_ids() if r in reference()]
