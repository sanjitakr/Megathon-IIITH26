"""Load trained models for prediction-time use. Every caller has a simulator fallback."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from engine.inputs import CustomerInput
from ml.features import features_from_input, to_frame
from sim.parameters import MODELS_DIR


class ModelsUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _baseline_bundle():
    path = MODELS_DIR / "baseline_models.joblib"
    if not path.exists():
        raise ModelsUnavailable("baseline models not trained (run `make train`)")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _surrogates():
    path = MODELS_DIR / "saving_surrogates.joblib"
    if not path.exists():
        raise ModelsUnavailable("saving surrogates not trained")
    return joblib.load(path)


def reset_cache() -> None:
    _baseline_bundle.cache_clear()
    _surrogates.cache_clear()


def predict_baseline(inp: CustomerInput) -> dict[str, tuple[float, float, float]]:
    """Purchased kWh/yr per fuel as (P50, P10, P90) with a conformal 80% interval."""
    bundle = _baseline_bundle()
    m = bundle["models"]
    X = to_frame([features_from_input(inp)])
    phys_e = np.log(max(X.phys_elec.iloc[0], 50.0))
    e = m["electricity"]
    lo, hi = e["cqr"].predict(X)
    out = {"electricity": (float(np.exp(phys_e + e["model"].predict(X)[0])), float(np.exp(phys_e + lo[0])), float(np.exp(phys_e + hi[0])))}
    g = m["gas"]
    prob = float(g["classifier"].predict_proba(X)[0, 1])
    if prob >= 0.5 or inp.main_heating == "gas_boiler":
        phys_g = np.log1p(X.phys_gas.iloc[0])
        glo, ghi = g["cqr"].predict(X)
        out["gas"] = (float(np.expm1(phys_g + g["model"].predict(X)[0])), float(np.expm1(phys_g + glo[0])), float(np.expm1(phys_g + ghi[0])))
    else:
        out["gas"] = (0.0, 0.0, 0.0)
    return out


def preview_savings(inp: CustomerInput, retrofits: list[str]) -> dict:
    """Fast standalone saving preview for sliders. The simulator result is authoritative."""
    sur = _surrogates()
    X = to_frame([features_from_input(inp)])
    out = {}
    for rid in retrofits:
        if rid in sur:
            out[rid] = {k.replace("d_", "") + "_kwh": float(sur[rid][k].predict(X)[0]) for k in ("d_elec", "d_gas")}
    return {"preview": out, "note": "Surrogate preview only; figures shown in the plan come from the simulator."}
