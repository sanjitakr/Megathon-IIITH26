"""Monte Carlo ranges (spec 9): P10 to P90 over catalogue ranges for retrofit effects
(U-values, SCOP, self-consumption, ...), costs and per-home calibration uncertainty."""
from __future__ import annotations

import numpy as np

from engine.carbon import EmissionSet, PriceSet, annual_bill, annual_carbon, payback_years
from engine.costs import cost_range
from sim.home import FUELS, HomeState
from sim.parameters import catalogue
from sim.retrofits import apply_package
from sim.simulator import Calibration, simulate

TIER_KAPPA_SIGMA = {"A": 0.05, "B": 0.10, "C": 0.20}


def _tri(rng: np.random.Generator, lo: float, mid: float, hi: float) -> float:
    if hi <= lo:
        return mid
    mid = min(max(mid, lo), hi)
    return float(rng.triangular(lo, mid, hi))


def draw_effects(ids, rng: np.random.Generator) -> dict:
    cat = catalogue()["retrofits"]
    return {rid: {k: _tri(rng, *v) for k, v in cat[rid]["effect"].items()} for rid in ids}


def draw_cost(ids, state: HomeState, rng: np.random.Generator, apply_grants: bool = False) -> float:
    return sum(_tri(rng, *cost_range(rid, state, apply_grants)) for rid in ids)


def quantiles(values, qs=(10, 50, 90)) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(np.percentile(arr, q)) for q in qs]


def mc_pair(
    state: HomeState,
    calib: Calibration,
    base_ids,
    after_ids,
    ps: PriceSet,
    es: EmissionSet,
    tier: str,
    n: int = 60,
    seed: int = 0,
    widen: float = 1.0,
    apply_grants: bool = False,
) -> dict:
    """Distribution of the change from package base_ids to package after_ids (same draws for both)."""
    rng = np.random.default_rng(seed)
    added = [r for r in after_ids if r not in base_ids]
    sigma = TIER_KAPPA_SIGMA.get(tier, 0.1) * widen
    out = {f: [] for f in FUELS}
    bills, carbon, costs, paybacks = [], [], [], []
    all_ids = sorted(set(base_ids) | set(after_ids))
    for _ in range(n):
        draws = draw_effects(all_ids, rng)
        c = Calibration(kappa_home=calib.kappa_home * float(np.exp(rng.normal(0, sigma))), alpha=calib.alpha)
        sb = simulate(apply_package(state, base_ids, draws), c) if base_ids else simulate(state, c)
        sa = simulate(apply_package(state, after_ids, draws), c)
        pb, pa = sb.purchased, sa.purchased
        for f in FUELS:
            out[f].append(pb[f] - pa[f])
        bill = annual_bill(pb, sb.annual("X"), ps) - annual_bill(pa, sa.annual("X"), ps)
        bills.append(bill)
        carbon.append(annual_carbon(pb, es) - annual_carbon(pa, es))
        cost = draw_cost(added, state, rng, apply_grants)
        costs.append(cost)
        pbk = payback_years(cost, bill)
        paybacks.append(1e9 if pbk is None else pbk)
    res = {"delta_kwh": {f: quantiles(v) for f, v in out.items()}}
    res["bill_saving_gbp"] = quantiles(bills)
    res["carbon_saving_kg"] = quantiles(carbon)
    res["cost_gbp"] = quantiles(costs)
    pb_q = quantiles(paybacks)
    res["payback_years"] = [None if x >= 1e8 else x for x in pb_q]
    res["n_draws"] = n
    return res
