"""Property value impact (spec 10): indicative, observational, not causal.

Uplift = predicted value at the post-retrofit band minus at the current band, everything
else fixed, from the hedonic model trained by ml/value_model.py on Land Registry sales
joined to EPC. If no model exists, the engine abstains.
"""
from __future__ import annotations

from functools import lru_cache

import joblib

from sim.parameters import MODELS_DIR

VALUE_MODEL_PATH = MODELS_DIR / "value_model.joblib"


@lru_cache(maxsize=1)
def _bundle():
    if not VALUE_MODEL_PATH.exists():
        return None
    return joblib.load(VALUE_MODEL_PATH)


def reset_cache() -> None:
    _bundle.cache_clear()


def value_impact(state, current_band: str, after_band: str, current_value: float | None, sale_year: int = 2026) -> dict:
    bundle = _bundle()
    if bundle is None:
        return {
            "status": "unavailable",
            "reason": "No value model is trained (needs HM Land Registry Price Paid joined to EPC). No value estimate is given.",
        }
    from ml.value_model import predict_uplift

    res = predict_uplift(bundle, state, current_band, after_band, sale_year)
    out = {
        "status": "ok",
        "basis": bundle["meta"]["basis"],
        "uplift_pct_range": [round(res["p10"], 1), round(res["p50"], 1), round(res["p90"], 1)],
        "current_band": current_band,
        "after_band": after_band,
        "label": "Indicative, observational association between EPC band and sale price; not a causal estimate.",
        "evidence": bundle["meta"].get("evidence", {}),
    }
    if bundle["meta"]["basis"] != "land_registry_epc":
        out["warning"] = "DEMO ONLY: value model trained on synthetic sales, not real Land Registry evidence."
    if bundle["meta"].get("thin_join"):
        out["warning_thin"] = "The EPC to Land Registry join is thin for this kind of property; the range is widened."
    if current_value:
        out["uplift_gbp_range"] = [round(current_value * x / 100, -2) for x in out["uplift_pct_range"]]
    if current_band == after_band:
        out["note"] = "The package does not change the EPC-style band, so no band-based uplift is expected."
    return out
