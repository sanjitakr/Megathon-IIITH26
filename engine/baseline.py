"""Canonical baseline (spec 9.1): one baseline feeds the score, savings and dashboard.

Purchased energy (grid import + fuels) drives bills, savings, carbon and improvement
potential. Gross consumption (L = I + g - X - Z, plus fuels) drives the intensity
component of the score. For homes with metered use, the simulator is calibrated to the
home (kappa_home on heating, alpha on appliances) before any retrofit is simulated, so
after = baseline - savings reconciles by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.inputs import CustomerInput, Usage
from engine.validation import Validation
from sim.energy_balance import gross_from_meters
from sim.home import FUELS, HomeState
from sim.parameters import electricity_shape, gas_shape, params, prices, region_hdd
from sim.simulator import Calibration, SimResult, simulate

TIER_RANGE = {"A": 0.05, "B": 0.12, "C": 0.30}  # fallback half-widths when no model interval exists
TIER_KAPPA_SIGMA = {"A": 0.05, "B": 0.10, "C": 0.20}


@dataclass
class Baseline:
    tier: str
    purchased: dict[str, float]
    gross: dict[str, float]
    basis: dict[str, str]
    gross_basis: str
    raw_metered: dict[str, float | None]
    weather_factor: float
    range: dict[str, list[float]]
    calibration: Calibration
    calibration_diag: dict
    sim: SimResult
    generation_kwh: float
    export_kwh: float
    flags: list[str] = field(default_factory=list)
    reconciliation: list[dict] = field(default_factory=list)
    model_source: str = "simulator"

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "purchased_kwh": {k: round(v, 0) for k, v in self.purchased.items()},
            "gross_kwh": {k: round(v, 0) for k, v in self.gross.items()},
            "basis": self.basis,
            "gross_basis": self.gross_basis,
            "raw_metered_kwh": {k: (round(v, 0) if v is not None else None) for k, v in self.raw_metered.items()},
            "weather_factor": round(self.weather_factor, 3),
            "range_p10_p90": {k: [round(a, 0), round(b, 0)] for k, (a, b) in self.range.items()},
            "generation_kwh": round(self.generation_kwh, 0),
            "export_kwh": round(self.export_kwh, 0),
            "calibration": self.calibration_diag,
            "reconciliation": self.reconciliation,
            "model_source": self.model_source,
            "flags": self.flags,
        }


def annualise(monthly: list[float | None], shape: np.ndarray) -> tuple[float, int]:
    """Scale the months present by typical annual / typical present-month demand.

    Missing months are never treated as zero.
    """
    idx = [i for i, x in enumerate(monthly) if x is not None]
    present = sum(monthly[i] for i in idx)
    share = float(shape[idx].sum())
    return present / share, len(idx)


def usage_to_annual(usage: Usage | None, fuel: str, region: str) -> tuple[float | None, str, list | None]:
    """(annual kWh, basis, monthly kWh list or None)."""
    if usage is None or usage.is_empty():
        return None, "none", None
    shape = electricity_shape() if fuel == "electricity" else gas_shape(region)
    cp = prices()["customer"]
    if usage.monthly_kwh is not None and any(x is not None for x in usage.monthly_kwh):
        annual, n = annualise(usage.monthly_kwh, shape)
        return annual, "metered" if n == 12 else "annualised", usage.monthly_kwh
    if usage.monthly_bills_gbp is not None and any(x is not None for x in usage.monthly_bills_gbp):
        rate = cp["unit_rate_gbp_per_kwh"][fuel]
        standing = cp["standing_charge_gbp_per_year"].get(fuel, 0.0) / 12
        kwh = [None if b is None else max(b - standing, 0.0) / rate for b in usage.monthly_bills_gbp]
        annual, _ = annualise(kwh, shape)
        return annual, "bills", kwh
    return float(usage.annual_kwh), "annual_total", None


def _bisect(fn, target: float, lo: float, hi: float, iters: int = 40) -> tuple[float, bool]:
    """Solve fn(x) = target for increasing fn; returns (x, hit_bound)."""
    f_lo, f_hi = fn(lo), fn(hi)
    if target <= f_lo:
        return lo, True
    if target >= f_hi:
        return hi, True
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def calibrate_home(state: HomeState, targets: dict[str, float], monthly_elec: list | None = None) -> tuple[Calibration, dict]:
    """Calibrate kappa_home (heating) and alpha (appliances) to metered purchased energy."""
    hc = params()["home_calibration"]
    calib = Calibration()
    diag = {"kappa_home": 1.0, "alpha": 1.0, "method": "none", "at_bound": [], "residual_pct": {}}
    heat_fuel = state.heating_fuel

    def sim_with(k=None, a=None) -> SimResult:
        return simulate(state, Calibration(kappa_home=k if k is not None else calib.kappa_home,
                                           alpha=a if a is not None else calib.alpha))

    if heat_fuel in ("gas", "oil", "lpg", "solid"):
        if heat_fuel in targets:
            k, hit = _bisect(lambda k: sim_with(k=k).purchased[heat_fuel], targets[heat_fuel], hc["kappa_min"], hc["kappa_max"])
            calib.kappa_home = k
            if hit:
                diag["at_bound"].append("kappa_home")
            diag["method"] = "bisection"
        if "electricity" in targets:
            a, hit = _bisect(lambda a: sim_with(a=a).purchased["electricity"], targets["electricity"], hc["alpha_min"], hc["alpha_max"])
            calib.alpha = a
            if hit:
                diag["at_bound"].append("alpha")
            diag["method"] = "bisection"
    elif "electricity" in targets:
        months = [i for i, x in enumerate(monthly_elec or []) if x is not None]
        if len(months) >= 9 and state.solar_kwp <= 0:
            base = simulate(state, Calibration())
            N = base.monthly["appliances"] + base.monthly["elec_dhw"]
            H = base.monthly["elec_space"]
            X = np.column_stack([N[months], H[months]])
            y = np.asarray([monthly_elec[i] for i in months], dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a = float(np.clip(coef[0], hc["alpha_min"], hc["alpha_max"]))
            k = float(np.clip(coef[1], hc["kappa_min"], hc["kappa_max"]))
            if a != coef[0]:
                diag["at_bound"].append("alpha")
            if k != coef[1]:
                diag["at_bound"].append("kappa_home")
            calib.alpha, calib.kappa_home = a, k
            # final annual touch-up on alpha so the annual total matches
            a2, _ = _bisect(lambda a: sim_with(a=a).purchased["electricity"], targets["electricity"], hc["alpha_min"], hc["alpha_max"])
            calib.alpha = a2
            diag["method"] = "monthly_regression"
        else:
            k, hit = _bisect(lambda k: sim_with(k=k).purchased["electricity"], targets["electricity"], hc["kappa_min"], hc["kappa_max"])
            calib.kappa_home = k
            diag["method"] = "bisection_heating_first"
            if hit:
                a, hit_a = _bisect(lambda a: sim_with(a=a).purchased["electricity"], targets["electricity"], hc["alpha_min"], hc["alpha_max"])
                calib.alpha = a
                if hit_a:
                    diag["at_bound"].append("alpha")
                    diag["at_bound"].append("kappa_home")
        if "gas" in targets and heat_fuel == "electricity":
            pass  # cooking gas only: not calibrated

    final = simulate(state, calib)
    for fuel, target in targets.items():
        if target > 0:
            diag["residual_pct"][fuel] = round(100 * (final.purchased[fuel] - target) / target, 2)
    diag["kappa_home"] = round(calib.kappa_home, 3)
    diag["alpha"] = round(calib.alpha, 3)
    tol = 100 * hc["tolerance"]
    diag["within_tolerance"] = all(abs(r) <= tol for r in diag["residual_pct"].values()) and not diag["at_bound"]
    return calib, diag


def weather_factor(inp: CustomerInput, region: str) -> float:
    if inp.hdd_metered_period:
        return float(region_hdd(region).sum() / inp.hdd_metered_period)
    return 1.0


def build_baseline(inp: CustomerInput, state: HomeState, validation: Validation) -> Baseline:
    tier = validation.tier
    flags: list[str] = []
    basis: dict[str, str] = {f: "not_used" for f in FUELS}
    raw: dict[str, float | None] = {f: None for f in FUELS}

    elec_annual, elec_basis, elec_monthly = usage_to_annual(inp.electricity, "electricity", state.region)
    gas_annual, gas_basis, _ = usage_to_annual(inp.gas, "gas", state.region)

    wf = weather_factor(inp, state.region)
    if wf == 1.0:
        flags.append("No weather data for the metered period: weather normalisation factor is 1.00.")

    default_sim = simulate(state, Calibration())
    heat_share = {
        f: (default_sim.heating_part[f] / default_sim.purchased[f]) if default_sim.purchased[f] > 0 else 0.0
        for f in FUELS
    }

    targets: dict[str, float] = {}
    if elec_annual is not None:
        raw["electricity"] = elec_annual
        targets["electricity"] = elec_annual * (1 - heat_share["electricity"]) + elec_annual * heat_share["electricity"] * wf
        basis["electricity"] = elec_basis
    if gas_annual is not None:
        raw["gas"] = gas_annual
        targets["gas"] = gas_annual * (1 - heat_share["gas"]) + gas_annual * heat_share["gas"] * wf
        basis["gas"] = gas_basis

    model_source = "simulator"
    ranges: dict[str, list[float]] = {}
    estimated_fuels = [f for f in ("electricity", "gas") if f not in targets and default_sim.purchased[f] > 0]
    if estimated_fuels:
        ml_pred = None
        try:
            from ml.runtime import predict_baseline

            ml_pred = predict_baseline(inp)
        except Exception as exc:  # models missing or failed: fall back to the simulator
            flags.append(f"ML baseline unavailable ({type(exc).__name__}); simulator estimate used.")
        for f in estimated_fuels:
            if ml_pred and f in ml_pred:
                p50, p10, p90 = ml_pred[f]
                targets[f] = p50
                ranges[f] = [p10, p90]
                model_source = "lightgbm_surrogate+conformal"
            else:
                targets[f] = default_sim.purchased[f]
            basis[f] = "estimated"

    calib, diag = calibrate_home(state, targets, elec_monthly if basis["electricity"] in ("metered", "annualised") else None)
    if not diag["within_tolerance"]:
        flags.append(
            "Metered use differs substantially from the model for this home "
            f"(kappa_home {diag['kappa_home']}, alpha {diag['alpha']}); savings ranges are widened."
        )
    sim = simulate(state, calib)

    purchased = dict(sim.purchased)
    for f in targets:
        purchased[f] = targets[f]
    for f in FUELS:
        if f not in targets and sim.purchased[f] > 0:
            basis[f] = "estimated"
    for f in FUELS:
        if f in ranges:
            continue
        half = TIER_RANGE[tier] if basis[f] != "estimated" else TIER_RANGE["C"]
        if not diag["within_tolerance"]:
            half *= 1.5
        ranges[f] = [purchased[f] * (1 - half), purchased[f] * (1 + half)]

    # Gross consumption
    gross = dict(sim.gross)
    for f in FUELS[1:]:
        gross[f] = purchased[f]
    gross_basis = "metered" if tier in ("A", "B") else "estimated"
    if state.solar_kwp > 0:
        if inp.solar_generation_kwh is not None and inp.solar_export_kwh is not None and "electricity" in targets and basis["electricity"] != "estimated":
            gross["electricity"] = gross_from_meters(purchased["electricity"], inp.solar_generation_kwh, inp.solar_export_kwh, sim.annual("Z"))
        else:
            gross["electricity"] = sim.gross["electricity"]
            gross_basis = "estimated"
            flags.append("Home has solar but no generation/export readings: gross electricity is estimated, so the score is provisional.")
    else:
        gross["electricity"] = purchased["electricity"]

    reconciliation = []
    for f in targets:
        reconciliation.append({
            "fuel": f,
            "metered_kwh": round(raw[f], 0) if raw[f] is not None else None,
            "normalised_kwh": round(targets[f], 0),
            "model_kwh": round(sim.purchased[f], 0),
            "residual_kwh": round(sim.purchased[f] - targets[f], 0),
        })

    return Baseline(
        tier=tier,
        purchased=purchased,
        gross=gross,
        basis=basis,
        gross_basis=gross_basis,
        raw_metered=raw,
        weather_factor=wf,
        range=ranges,
        calibration=calib,
        calibration_diag=diag,
        sim=sim,
        generation_kwh=sim.annual("g") if inp.solar_generation_kwh is None else inp.solar_generation_kwh,
        export_kwh=sim.annual("X") if inp.solar_export_kwh is None else inp.solar_export_kwh,
        flags=flags,
        reconciliation=reconciliation,
        model_source=model_source,
    )
