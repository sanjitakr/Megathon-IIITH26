"""Retrofit cost ranges from the catalogue: cost = fixed + per_unit x quantity."""
from __future__ import annotations

from sim.geometry import compute_geometry
from sim.home import HomeState
from sim.parameters import catalogue
from sim.retrofits import recommended_kwp

DISRUPTION_LEVEL = {"low": 1, "medium": 2, "high": 3}


def quantity(retrofit_id: str, state: HomeState) -> float:
    unit = catalogue()["retrofits"][retrofit_id]["cost"]["unit"]
    geom = compute_geometry(state)
    return {
        "roof_m2": geom.footprint * geom.roof_exposure,
        "wall_m2": geom.wall_area_net,
        "floor_m2": geom.floor_ground_area * geom.floor_exposure,
        "window_m2": geom.window_area,
        "kwp": recommended_kwp(state),
        "home": 1.0,
    }[unit]


def cost_range(retrofit_id: str, state: HomeState, apply_grants: bool = False) -> list[float]:
    """[low, mid, high] GBP for one retrofit on this home."""
    item = catalogue()["retrofits"][retrofit_id]
    q = quantity(retrofit_id, state)
    out = [item["cost"]["fixed"][i] + item["cost"]["per_unit"][i] * q for i in range(3)]
    if apply_grants and item.get("grant_gbp"):
        out = [max(c - item["grant_gbp"], 0.0) for c in out]
    return [round(c, -1) for c in out]


def disruption(retrofit_id: str) -> int:
    return DISRUPTION_LEVEL[catalogue()["retrofits"][retrofit_id]["disruption"]]
