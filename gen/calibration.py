"""Global calibration (spec 6.6).

Choose theta = (kappa, air change scale, appliance scale, boiler efficiency offset) to
minimise sum over (type, age band) cells of (log median_sim - log median_NEED)^2 for gas
and electricity, plus a small pull toward the starting values (the problem is weakly
identified: kappa and air change scale both move heating). Writes
data/processed/calibration.json and docs/calibration_report.md (before/after tables).
This is plausibility evidence, not validation of individual homes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gen.labels import hidden_calibration, true_state
from gen.sampling import sample_population
from sim.parameters import CALIBRATED_KEYS, PROCESSED_DIR, ROOT, need_targets, params, reload
from sim.rating import rating
from sim.simulator import simulate

THETA0 = {"kappa_global": 0.65, "air_change_scale_global": 1.0, "appliance_scale_global": 1.0, "boiler_efficiency_offset_global": 0.0}
SCALE = {"kappa_global": 0.3, "air_change_scale_global": 0.3, "appliance_scale_global": 0.3, "boiler_efficiency_offset_global": 0.05}
BOUNDS = {"kappa_global": (0.3, 1.5), "air_change_scale_global": (0.5, 1.5), "appliance_scale_global": (0.5, 1.6), "boiler_efficiency_offset_global": (-0.1, 0.08)}
EPC_REFERENCE_SHARE = {"A": 0.002, "B": 0.03, "C": 0.38, "D": 0.42, "E": 0.13, "F": 0.03, "G": 0.008}  # placeholder, verify


def need_cell_target(fuel: str, ptype: str, age: str) -> float:
    t = need_targets()
    return t[f"{fuel}_overall_median"] * t[f"{fuel}_type_factor"][ptype] * t[f"{fuel}_age_factor"][age]


def _simulate_all(prepared) -> pd.DataFrame:
    out = []
    for state, calib, ptype, age in prepared:
        r = simulate(state, calib).purchased
        out.append({"type": ptype, "age": age, "heating": state.heating, "solar": state.solar_kwp > 0,
                    "gas": r["gas"], "electricity": r["electricity"]})
    return pd.DataFrame(out)


def cell_table(df: pd.DataFrame, min_n: int = 12) -> pd.DataFrame:
    rows = []
    gas = df[df.heating == "gas_boiler"]
    elec = df[(df.heating.isin(["gas_boiler", "oil_boiler", "lpg_boiler"])) & (~df.solar)]
    for (t, a), g in gas.groupby(["type", "age"]):
        if len(g) >= min_n:
            rows.append({"fuel": "gas", "type": t, "age": a, "n": len(g), "median_sim": g.gas.median(), "median_need": need_cell_target("gas", t, a)})
    for (t, a), g in elec.groupby(["type", "age"]):
        if len(g) >= min_n:
            rows.append({"fuel": "electricity", "type": t, "age": a, "n": len(g), "median_sim": g.electricity.median(), "median_need": need_cell_target("electricity", t, a)})
    tab = pd.DataFrame(rows)
    tab["log_ratio"] = np.log(tab.median_sim / tab.median_need)
    return tab


def _set_theta(theta: dict) -> None:
    p = params()
    for k, v in theta.items():
        p[k] = float(v)


def run_calibration(n_homes: int = 1500, seed: int = 11, maxiter: int = 120) -> dict:
    reload()
    # start from uncalibrated values even if a calibration file exists
    _set_theta(THETA0)
    pop = sample_population(n_homes, seed).to_dict("records")
    prepared = []
    for row in pop:
        s = true_state(row)
        prepared.append((s, hidden_calibration(row), s.property_type, s.age_band))

    keys = list(CALIBRATED_KEYS)
    x0 = np.array([THETA0[k] for k in keys])

    def unpack(x):
        return {k: float(np.clip(x[i], *BOUNDS[k])) for i, k in enumerate(keys)}

    def objective(x):
        theta = unpack(x)
        _set_theta(theta)
        tab = cell_table(_simulate_all(prepared))
        fit = float((tab.log_ratio ** 2 * tab.n).sum() / tab.n.sum())
        reg = 0.02 * sum(((theta[k] - THETA0[k]) / SCALE[k]) ** 2 for k in keys)
        return fit + reg

    before_tab = cell_table(_simulate_all(prepared))
    obj_before = objective(x0)
    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})
    theta = unpack(res.x)
    _set_theta(theta)
    after_df = _simulate_all(prepared)
    after_tab = cell_table(after_df)

    ratings = pd.Series([rating(s)["band"] for s, *_ in prepared[:800]]).value_counts(normalize=True)
    rating_tab = pd.DataFrame({"simulated_share": ratings, "epc_reference_share": pd.Series(EPC_REFERENCE_SHARE)}).fillna(0.0)

    out = {
        "version": "calib-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M"),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "theta": theta,
        "theta0": THETA0,
        "objective_before": round(obj_before, 5),
        "objective_after": round(float(res.fun), 5),
        "n_homes": n_homes,
        "targets": "data/reference/need_targets.json (PLACEHOLDER medians - replace with NEED tables)",
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    write_report(out, before_tab, after_tab, rating_tab)
    reload()
    return out


def write_report(out: dict, before: pd.DataFrame, after: pd.DataFrame, rating_tab: pd.DataFrame) -> None:
    merged = before.merge(after, on=["fuel", "type", "age", "n", "median_need"], suffixes=("_before", "_after"))
    lines = [
        "# Calibration report (plausibility evidence only)",
        "",
        f"Created {out['created']} - version `{out['version']}` - {out['n_homes']} sampled homes.",
        "",
        "NEED targets are **placeholders** until replaced with the published tables. Matching cell medians shows the simulator sits in a realistic range; it does not show home-level accuracy.",
        "",
        "## Parameters",
        "",
        "| parameter | start | calibrated |",
        "|---|---|---|",
    ]
    for k, v in out["theta"].items():
        lines.append(f"| {k} | {out['theta0'][k]} | {v:.4f} |")
    lines += ["", f"Objective (weighted mean squared log ratio + regulariser): before {out['objective_before']}, after {out['objective_after']}.", ""]
    lines += ["## Cell medians (kWh/yr)", "", "| fuel | type | age | n | NEED | sim before | sim after |", "|---|---|---|---|---|---|---|"]
    for _, r in merged.sort_values(["fuel", "type", "age"]).iterrows():
        lines.append(f"| {r.fuel} | {r.type} | {r.age} | {r.n} | {r.median_need:,.0f} | {r.median_sim_before:,.0f} | {r.median_sim_after:,.0f} |")
    lines += ["", "## EPC-style rating distribution vs EPC reference (placeholder shares)", "", "| band | simulated | reference |", "|---|---|---|"]
    for band in "ABCDEFG":
        if band in rating_tab.index:
            r = rating_tab.loc[band]
            lines.append(f"| {band} | {r.simulated_share:.3f} | {r.epc_reference_share:.3f} |")
    lines += ["", "The monthly electricity shape uses the reference smart-meter shape directly (data/reference/parameters.json), so it matches by construction until a smart-meter subset is loaded."]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--maxiter", type=int, default=120)
    a = ap.parse_args()
    print(json.dumps(run_calibration(a.n, a.seed, a.maxiter), indent=2))
