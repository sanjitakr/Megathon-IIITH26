"""Green Home Score (spec 8): S = sum(w_k x s_k), deterministic, no ML.

The current and projected scores use this same function; only the inputs differ.
"""
from __future__ import annotations

import numpy as np

from sim.geometry import compute_geometry
from sim.home import HomeState
from sim.parameters import score_config
from sim.simulator import u_values

LABELS = {
    "efficiency": "Energy efficiency",
    "intensity": "Energy intensity",
    "insulation": "Insulation",
    "heating": "Heating system",
    "glazing": "Glazing",
    "renewable": "Renewable generation",
    "smart": "Smart controls",
}


def piecewise(x: float, xs, ys) -> float:
    """Piecewise-linear between anchors, clamped at the ends."""
    return float(np.interp(x, xs, ys))


def classify(score: float) -> str:
    for lower, label in score_config()["classes"]:
        if score >= lower:
            return label
    return "Very poor"


def roof_key(state: HomeState) -> str:
    if state.loft == "flat_roof" or state.roof_type == "flat":
        return "flat_roof_insulated" if u_values(state)["roof"] <= 0.5 else "flat_roof_uninsulated"
    return state.loft


def insulation_score(state: HomeState) -> tuple[float, dict]:
    cfg = score_config()
    geom = compute_geometry(state)
    parts = {
        "roof": (cfg["roof_score"][roof_key(state)], cfg["insulation_weights"]["roof"] * geom.roof_exposure),
        "wall": (cfg["wall_score"][state.wall_type], cfg["insulation_weights"]["wall"]),
        "floor": (cfg["floor_score"][state.floor_type], cfg["insulation_weights"]["floor"] * geom.floor_exposure),
    }
    total_w = sum(w for _, w in parts.values())
    score = sum(s * w for s, w in parts.values()) / total_w
    return score, {k: {"score": s, "weight": round(w / total_w, 3)} for k, (s, w) in parts.items()}


def heating_score(state: HomeState) -> float:
    table = score_config()["heating_score"][state.heating]
    if isinstance(table, dict):
        return float(table[state.boiler_age])
    return float(table)


def smart_score(state: HomeState) -> float:
    cfg = score_config()["smart_score"]
    return float(cfg["tariff_aware"] if state.tariff_aware else cfg[state.controls])


def efficiency_score(sap_points: float | None, band: str | None) -> float:
    cfg = score_config()
    if sap_points is not None:
        a = cfg["efficiency_sap_anchors"]
        return piecewise(sap_points, a["x"], a["y"])
    return float(cfg["efficiency_band_score"][band])


ISSUE_TEXT = {
    "efficiency": "EPC-style rating is low",
    "intensity": "High energy use per m2",
    "heating": "Heating system is inefficient or high-carbon",
    "glazing": "Glazing loses a lot of heat",
    "renewable": "No or little renewable generation",
    "smart": "Heating controls are basic",
}

INSULATION_ISSUE = {
    "roof": "Loft or roof is poorly insulated",
    "wall": "Walls are uninsulated",
    "floor": "Floor is uninsulated",
}


def compute_score(
    state: HomeState,
    *,
    sap_points: float | None,
    band: str | None,
    efficiency_basis: str,
    intensity_kwh_m2: float,
    intensity_basis: str,
    generation_kwh: float,
    gross_electricity_kwh: float,
    estimated_components: set[str] | None = None,
) -> dict:
    cfg = score_config()
    w = cfg["weights"]
    estimated_components = set(estimated_components or ())

    ins, ins_parts = insulation_score(state)
    renewable_share = generation_kwh / gross_electricity_kwh if gross_electricity_kwh > 0 else 0.0
    ia = cfg["intensity_anchors"]
    ra = cfg["renewable_anchors"]

    sub = {
        "efficiency": efficiency_score(sap_points, band),
        "intensity": piecewise(intensity_kwh_m2, ia["x"], ia["y"]),
        "insulation": ins,
        "heating": heating_score(state),
        "glazing": float(cfg["glazing_score"][state.glazing]),
        "renewable": piecewise(renewable_share, ra["x"], ra["y"]),
        "smart": smart_score(state),
    }
    inputs = {
        "efficiency": f"SAP {sap_points:.0f} (band {band})" if sap_points is not None else f"band {band}",
        "intensity": f"{intensity_kwh_m2:.0f} kWh/m2/yr gross",
        "insulation": f"roof {roof_key(state)}, wall {state.wall_type}, floor {state.floor_type}",
        "heating": state.heating + (f" ({state.boiler_age.replace('_', '-')} yrs)" if "boiler" in state.heating else ""),
        "glazing": state.glazing,
        "renewable": f"{100 * renewable_share:.0f}% of electricity demand",
        "smart": "tariff-aware" if state.tariff_aware else state.controls,
    }
    basis = {k: "recorded" for k in sub}
    basis["efficiency"] = efficiency_basis
    basis["intensity"] = intensity_basis
    for k in estimated_components:
        basis[k] = "estimated" if basis[k] == "recorded" else basis[k]

    components = []
    for k in w:
        components.append({
            "key": k,
            "label": LABELS[k],
            "weight_pct": round(100 * w[k], 1),
            "sub_score": round(sub[k], 2),
            "contribution": round(w[k] * sub[k], 2),
            "points_lost": round(w[k] * (100 - sub[k]), 2),
            "input": inputs[k],
            "basis": basis[k],
        })
    total = sum(c["contribution"] for c in components)
    rounded = int(round(total))
    provisional = any(b in ("estimated", "default") for b in basis.values()) or intensity_basis == "estimated"

    issues = []
    for c in sorted(components, key=lambda c: -c["points_lost"]):
        if c["points_lost"] < 2.0 or len(issues) >= 3:
            continue
        text = ISSUE_TEXT.get(c["key"])
        if c["key"] == "insulation":
            worst = max(ins_parts.items(), key=lambda kv: kv[1]["weight"] * (100 - kv[1]["score"]))
            text = INSULATION_ISSUE[worst[0]]
        issues.append({"component": c["key"], "issue": text, "points_lost": c["points_lost"], "detail": c["input"]})

    return {
        "score": round(total, 2),
        "score_rounded": rounded,
        "class": classify(rounded),
        "components": components,
        "insulation_parts": ins_parts,
        "renewable_share": round(renewable_share, 3),
        "provisional": provisional,
        "key_issues": issues,
    }
