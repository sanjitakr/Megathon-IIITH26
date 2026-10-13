"""EPC-style rating estimate (spec 6.7). An approximation of SAP, not an official EPC.

ECF = deflator x cost / (floor area + 45)
SAP = 117 - 121 log10(ECF) if ECF >= 3.5 else 100 - 13.95 ECF
Fixed standard prices and standard occupancy are used, never customer prices or metered use.
Constants are unconfirmed for SAP 10.2 / RdSAP 10 (see data/reference/prices.json).
"""
from __future__ import annotations

import math

from sim.home import HomeState
from sim.parameters import prices
from sim.simulator import simulate


def band_edges() -> dict[str, float]:
    return prices()["rating_basis"]["band_edges"]


def band_from_points(points: float) -> str:
    for band, lower in sorted(band_edges().items(), key=lambda kv: -kv[1]):
        if points >= lower:
            return band
    return "G"


def band_lower_edge(band: str) -> float:
    return band_edges()[band]


def sap_from_cost(cost: float, floor_area: float) -> tuple[float, float]:
    rb = prices()["rating_basis"]
    ecf = rb["deflator"] * cost / (floor_area + 45.0)
    if ecf >= rb["ecf_kink"]:
        sap = 117.0 - 121.0 * math.log10(ecf)
    else:
        sap = 100.0 - 13.95 * ecf
    return max(sap, 1.0), ecf


def rating(state: HomeState) -> dict:
    rb = prices()["rating_basis"]
    res = simulate(state, rating_mode=True)
    gross = res.gross
    rates = rb["unit_rate_gbp_per_kwh"]
    cost = sum(gross[f] * rates[f] for f in gross)
    cost += rb["standing_charge_gbp_per_year"]["electricity"]
    if gross["gas"] > 0:
        cost += rb["standing_charge_gbp_per_year"]["gas"]
    cost -= res.annual("g") * rb["pv_credit_gbp_per_kwh"]
    sap, ecf = sap_from_cost(cost, state.floor_area)
    return {
        "sap_points": round(sap, 1),
        "band": band_from_points(sap),
        "ecf": round(ecf, 3),
        "standard_cost_gbp": round(cost, 0),
        "basis": "EPC-style estimate (SAP-like formula, standard prices and occupancy; not an official EPC)",
    }
