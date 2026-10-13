"""Labels produced by the simulator (spec 6.9).

Per home: clean baseline kWh per fuel, rating, feasibility, standalone per-retrofit deltas,
full-package delta and improvement potential. Hidden factors (behaviour, build quality,
appliance and hot-water use) enter the simulation but are never model inputs.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
from multiprocessing import get_context

import numpy as np

from engine.carbon import PriceSet, annual_bill
from engine.costs import cost_range
from engine.feasibility import assess
from sim.home import FUELS, HomeState
from sim.parameters import params, retrofit_ids
from sim.rating import rating
from sim.retrofits import apply_package
from sim.simulator import Calibration, simulate

HIDDEN = ("behaviour", "build_quality", "appliance_hidden", "dhw_hidden")


def true_state(row: dict) -> HomeState:
    state = HomeState.from_dict(json.loads(row["state_json"]))
    state.overrides["build_quality"] = row["hidden_build_quality"]
    return state


def hidden_calibration(row: dict) -> Calibration:
    return Calibration(
        behaviour=row["hidden_behaviour"],
        appliance_hidden=row["hidden_appliance_hidden"],
        dhw_hidden=row["hidden_dhw_hidden"],
    )


def _purchased_total(p: dict) -> float:
    return sum(p[f] for f in FUELS)


def label_home(row: dict) -> dict:
    state = true_state(row)
    calib = hidden_calibration(row)
    base = simulate(state, calib)
    rat = rating(state)
    feas = assess(state)
    F = feas.feasible_ids()
    ps = PriceSet.from_options()
    bill_base = annual_bill(base.purchased, base.annual("X"), ps)

    out = {
        "property_id": row["property_id"],
        "clean_electricity_kwh": base.purchased["electricity"],
        "clean_gas_kwh": base.purchased["gas"],
        "clean_oil_kwh": base.purchased["oil"],
        "clean_lpg_kwh": base.purchased["lpg"],
        "clean_solid_kwh": base.purchased["solid"],
        "clean_gross_electricity_kwh": base.gross["electricity"],
        "clean_generation_kwh": base.annual("g"),
        "clean_export_kwh": base.annual("X"),
        "hlc": base.hlc,
        "hlp": base.hlp,
        "sap_points": rat["sap_points"],
        "epc_band_true": rat["band"],
        "monthly_electricity_clean": json.dumps([round(float(x), 2) for x in base.monthly["I"]]),
        "monthly_gas_clean": json.dumps([round(float(x), 2) for x in base.monthly["gas"]]),
        "n_feasible": len(F),
    }
    base_total = _purchased_total(base.purchased)
    for rid in retrofit_ids():
        feasible = rid in feas.feasible
        out[f"feasible_{rid}"] = int(feasible)
        for key in ("d_elec", "d_gas", "d_other", "pct_saving", "saving_per_gbp"):
            out[f"{key}_{rid}"] = np.nan
        if not feasible:
            continue
        prereq = [p for p in feas.feasible[rid].get("prerequisites", []) if rid == "battery"]
        ref = simulate(apply_package(state, prereq), calib) if prereq else base
        after = simulate(apply_package(state, prereq + [rid]), calib)
        d = {f: ref.purchased[f] - after.purchased[f] for f in FUELS}
        out[f"d_elec_{rid}"] = d["electricity"]
        out[f"d_gas_{rid}"] = d["gas"]
        out[f"d_other_{rid}"] = d["oil"] + d["lpg"] + d["solid"]
        ref_total = _purchased_total(ref.purchased)
        out[f"pct_saving_{rid}"] = 100 * (ref_total - _purchased_total(after.purchased)) / ref_total if ref_total > 0 else 0.0
        bill = annual_bill(ref.purchased, ref.annual("X"), ps) - annual_bill(after.purchased, after.annual("X"), ps)
        out[f"saving_per_gbp_{rid}"] = bill / max(cost_range(rid, state)[1], 1.0)
        out[f"bill_saving_{rid}"] = bill

    if F:
        pkg_state = apply_package(state, F)
        pkg = simulate(pkg_state, calib)
        out["pkg_d_elec"] = base.purchased["electricity"] - pkg.purchased["electricity"]
        out["pkg_d_gas"] = base.purchased["gas"] - pkg.purchased["gas"]
        out["pkg_d_other"] = sum(base.purchased[f] - pkg.purchased[f] for f in ("oil", "lpg", "solid"))
        out["pkg_bill_saving"] = bill_base - annual_bill(pkg.purchased, pkg.annual("X"), ps)
        out["pkg_sap_after"] = rating(pkg_state)["sap_points"]
    else:
        out.update({"pkg_d_elec": 0.0, "pkg_d_gas": 0.0, "pkg_d_other": 0.0, "pkg_bill_saving": 0.0, "pkg_sap_after": rat["sap_points"]})
    E = base.purchased["electricity"] + base.purchased["gas"]
    out["improvement_potential_pct"] = 100 * (out["pkg_d_elec"] + out["pkg_d_gas"]) / E if E > 0 else np.nan
    out["pct_saving_total_base"] = base_total
    return out


def label_many(rows: list[dict], workers: int | None = None) -> list[dict]:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(rows) < 200:
        return [label_home(r) for r in rows]
    with get_context("fork").Pool(workers) as pool:
        return pool.map(label_home, rows, chunksize=100)


def build_retrofit_reference(n_homes: int = 600, seed: int = 7, rows: list[dict] | None = None, labels: list[dict] | None = None) -> dict:
    """Sorted standalone % saving and saving-per-GBP per retrofit among homes where feasible."""
    if labels is None:
        if rows is None:
            from gen.sampling import sample_population

            rows = sample_population(n_homes, seed).to_dict("records")
        labels = label_many(rows)
    ref = {}
    for rid in retrofit_ids():
        pct = [l[f"pct_saving_{rid}"] for l in labels if l.get(f"feasible_{rid}") and np.isfinite(l[f"pct_saving_{rid}"])]
        val = [l[f"saving_per_gbp_{rid}"] for l in labels if l.get(f"feasible_{rid}") and np.isfinite(l[f"saving_per_gbp_{rid}"])]
        if pct:
            ref[rid] = {"pct_saving": sorted(round(x, 3) for x in pct), "saving_per_gbp": sorted(round(x, 5) for x in val), "n": len(pct)}
    return ref


@contextlib.contextmanager
def perturbed_parameters(magnitude: float = 0.2, seed: int = 0):
    """Simulator 'version B' for the shift test: U-values, boiler efficiency, SCOP and
    occupancy-driven loads moved by up to +/- magnitude."""
    p = params()
    saved = copy.deepcopy(p)
    rng = np.random.default_rng(seed)

    def f():
        return float(1 + rng.choice([-1, 1]) * magnitude * rng.uniform(0.5, 1.0))

    try:
        for table in ("u_wall", "u_roof", "u_floor", "u_window"):
            k = f()
            p[table] = {key: v * k for key, v in p[table].items()}
        k = f()
        for sysname, ages in p["boiler_efficiency"].items():
            p["boiler_efficiency"][sysname] = {a: min(0.98, v * (1 / k)) for a, v in ages.items()}
        p["heat_pump"]["scop_base"] *= f()
        p["hot_water"]["a1"] *= f()
        p["appliances"]["a1"] *= f()
        yield p
    finally:
        p.clear()
        p.update(saved)
