"""End-to-end flow (spec 2). Produces the results JSON, the single source of truth that
the dashboard and the LLM explanation read from."""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from datetime import datetime, timezone

from engine import memory
from engine.baseline import build_baseline
from engine.carbon import EmissionSet, PriceSet, annual_bill, annual_carbon
from engine.feasibility import assess
from engine.inputs import AGE_BAND_LABELS, CustomerInput, CustomiseOptions
from engine.planner import PlanContext, SubsetEval, build_context, customise, feasible_list, improvement_potential
from engine.score import compute_score
from engine.state_builder import build_state
from engine.validation import validate
from engine.value import value_impact
from sim.home import FUELS
from sim.parameters import catalogue, emission_factors, params, prices, regions
from sim.rating import band_from_points, band_lower_edge, rating

ENGINE_VERSION = "0.1.0"
DISCLAIMER = (
    "Estimated by a simulation model, not measured. Savings, costs and values are estimates, "
    "not guarantees. A qualified assessor should confirm before any work."
)

_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_SIZE = 64

DEFAULT_TO_COMPONENT = {
    "wall_type": "insulation", "loft": "insulation", "floor_type": "insulation",
    "glazing": "glazing", "controls": "smart", "boiler_age": "heating",
}


def _key(inp: CustomerInput, apply_grants: bool) -> str:
    return hashlib.sha256((inp.property_key() + str(apply_grants)).encode()).hexdigest()


def _efficiency_anchor(inp: CustomerInput, sim_points: float) -> tuple[float | None, str | None, str]:
    """(SAP recorded or None, recorded band, basis). Band-only inputs anchor on the band's lower edge."""
    if inp.sap_points is not None:
        return float(inp.sap_points), band_from_points(inp.sap_points), "recorded"
    if inp.epc_band != "not_sure":
        return band_lower_edge(inp.epc_band), inp.epc_band, "recorded"
    return None, None, "estimated"


def _stage1(inp: CustomerInput, v, apply_grants: bool, mc_draws: int) -> dict:
    state, defaults = build_state(inp)
    baseline = build_baseline(inp, state, v)
    sim_rating = rating(state)
    sap_recorded, band_recorded, eff_basis = _efficiency_anchor(inp, sim_rating["sap_points"])

    estimated_components = {DEFAULT_TO_COMPONENT[d["field"]] for d in defaults if d["field"] in DEFAULT_TO_COMPONENT}
    intensity_basis_current = "metered" if (baseline.tier in ("A", "B") and baseline.gross_basis == "metered") else "estimated"
    area = state.floor_area

    def score_fn(ev: SubsetEval, ctx: PlanContext) -> dict:
        is_current = len(ev.ids) == 0
        base_ev = ctx.base if ctx is not None and ctx.base is not None and not is_current else ev
        after_gross = {f: max(baseline.gross[f] - (base_ev.gross[f] - ev.gross[f]), 0.0) for f in FUELS}
        intensity = sum(after_gross.values()) / area
        if is_current:
            sap = sap_recorded if sap_recorded is not None else sim_rating["sap_points"]
            band = band_recorded or sim_rating["band"]
        else:
            sap = ev.sap_anchored
            band = ev.band
        return compute_score(
            ev.state if ev.state is not None else state,
            sap_points=sap,
            band=band,
            efficiency_basis=(eff_basis if is_current else "modelled"),
            intensity_kwh_m2=intensity,
            intensity_basis=(intensity_basis_current if is_current else "modelled"),
            generation_kwh=ev.generation,
            gross_electricity_kwh=after_gross["electricity"],
            estimated_components=estimated_components,
        )

    feas = assess(state)
    ctx = build_context(state, baseline, feas, sap_recorded, score_fn=score_fn, apply_grants=apply_grants, mc_draws=mc_draws)
    current_score = ctx.base.score
    current_score["basis"] = {"A": "measured", "B": "measured", "C": "estimated"}[baseline.tier]
    if baseline.tier == "C":
        current_score["provisional"] = True

    return {
        "state": state, "defaults": defaults, "baseline": baseline, "sim_rating": sim_rating,
        "sap_recorded": sap_recorded, "band_recorded": band_recorded, "efficiency_basis": eff_basis,
        "feasibility": feas, "ctx": ctx, "current_score": current_score,
        "improvement_potential": improvement_potential(ctx), "stage1_lists": {},
    }


def get_stage1(inp: CustomerInput, v, apply_grants: bool = False, mc_draws: int = 60) -> dict:
    key = _key(inp, apply_grants)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    bundle = _stage1(inp, v, apply_grants, mc_draws)
    _CACHE[key] = bundle
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return bundle


def options_from_input(inp: CustomerInput, opts: CustomiseOptions | None) -> CustomiseOptions:
    if opts is not None:
        o = opts.model_copy(deep=True)
        o.declined = sorted(set(o.declined) | set(inp.declined))
        return o
    return CustomiseOptions(goal=inp.goal, budget=inp.budget, disruption_tolerance=inp.disruption_tolerance, declined=list(inp.declined))


def insufficient(v, inp: CustomerInput) -> dict:
    return {
        "status": "insufficient_information",
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": v.questions,
        "conflicts": v.conflicts,
        "warnings": v.warnings,
        "message": "We need a little more information before we can give a result.",
        "disclaimer": DISCLAIMER,
    }


def analyse(
    inp: CustomerInput,
    opts: CustomiseOptions | None = None,
    with_ranges: bool = True,
    use_memory: bool = False,
    mc_draws: int = 60,
) -> dict:
    t0 = time.perf_counter()
    if use_memory:
        inp = memory.apply_to_input(inp, memory.load(inp.customer_id))
    v = validate(inp)
    if v.status == "abstain":
        return insufficient(v, inp)
    opts = options_from_input(inp, opts)
    b = get_stage1(inp, v, opts.apply_grants, mc_draws)
    t_stage1 = time.perf_counter()

    ctx: PlanContext = b["ctx"]
    baseline = b["baseline"]
    state = b["state"]
    ps = PriceSet.from_options(opts)
    es = EmissionSet.from_options(opts)

    list_key = (repr(ps.to_dict()), repr(es.to_dict()), with_ranges)
    if list_key not in b["stage1_lists"]:
        b["stage1_lists"][list_key] = feasible_list(ctx, ps, es, with_ranges=with_ranges)
    stage1 = b["stage1_lists"][list_key]

    plan = customise(ctx, opts, with_ranges=with_ranges)
    t_stage2 = time.perf_counter()

    current_band = b["band_recorded"] or b["sim_rating"]["band"]
    best = plan.get("best")
    after_band = best["band_after"] if best else current_band
    value = value_impact(state, current_band, after_band, inp.current_value_estimate)

    cur = b["current_score"]
    poor = cur["class"] in ("Poor", "Very poor")
    region_label = regions()["regions"][state.region]["label"]
    bill_base = annual_bill(baseline.purchased, baseline.export_kwh, ps)
    carbon_base = annual_carbon(baseline.purchased, es)

    results = {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "property": {
            "postcode_district": inp.postcode.split()[0].upper() if inp.postcode else None,
            "region": state.region,
            "region_label": region_label,
            "property_type": state.property_type,
            "age_band": state.age_band,
            "age_band_label": AGE_BAND_LABELS[state.age_band],
            "floor_area_m2": state.floor_area,
            "bedrooms": state.bedrooms,
            "occupants": state.occupants,
            "heating": state.heating,
            "wall_type": state.wall_type,
            "loft": state.loft,
            "glazing": state.glazing,
            "solar_kwp": state.solar_kwp,
            "tenure": state.tenure,
            "hlc_w_per_k": round(ctx.base.hlc, 0),
            "hlp_w_per_m2k": round(ctx.base.hlp, 2),
        },
        "confidence": {
            "tier": baseline.tier,
            "label": {"A": "Full result from metered use", "B": "Partial usage data: wider ranges", "C": "No usage data: provisional estimate"}[baseline.tier],
            "warnings": v.warnings + baseline.flags,
            "defaults_used": b["defaults"],
        },
        "baseline": {
            **baseline.to_dict(),
            "bill_gbp": round(bill_base, 0),
            "carbon_kg": round(carbon_base, 0),
            "intensity_kwh_per_m2": round(sum(baseline.gross.values()) / state.floor_area, 0),
            "explanation": "Raw metered figures are annualised from the months given and weather-normalised (heating part only); the model is calibrated to this baseline before any retrofit is simulated.",
        },
        "rating": {
            "recorded_band": b["band_recorded"],
            "recorded_sap": b["sap_recorded"],
            "simulated": b["sim_rating"],
            "basis": b["efficiency_basis"],
            "mismatch": (b["band_recorded"] is not None and b["band_recorded"] != b["sim_rating"]["band"]),
        },
        "score": {
            "current": cur,
            "projected": best["score_after"] if best else None,
            "projected_basis": "modelled",
            "projected_components": (ctx.get(best["retrofits"]).score["components"] if best else None),
        },
        "efficiency_flag": {
            "is_poor": poor,
            "class": cur["class"],
            "key_issues": cur["key_issues"] if poor else cur["key_issues"][:2],
            "improvement_potential": b["improvement_potential"],
        },
        "retrofits": {
            "feasible_list": stage1["rows"],
            "package_row": stage1["package"],
            "note": stage1["note"],
            "not_available": b["feasibility"].to_dict()["not_available"],
            "hp_readiness": b["feasibility"].hp_readiness,
            "enumeration_truncated": ctx.truncated,
        },
        "plan": plan,
        "value": value,
        "assumptions": assumptions_summary(ps, es),
        "timing_ms": {
            "stage1": round(1000 * (t_stage1 - t0), 1),
            "stage2": round(1000 * (t_stage2 - t_stage1), 1),
            "total": round(1000 * (time.perf_counter() - t0), 1),
            "subsets_cached": len(ctx.cache),
        },
    }
    if results["rating"]["mismatch"]:
        results["confidence"]["warnings"].append(
            f"Your recorded EPC band ({b['band_recorded']}) differs from our EPC-style estimate ({b['sim_rating']['band']}); "
            "projected ratings are anchored on your recorded rating."
        )
    if state.tenure == "rent":
        results["confidence"]["warnings"].append("Rented home: fabric measures are shown but need landlord consent; major installations are the owner's decision.")
    return results


def assumptions_summary(ps: PriceSet | None = None, es: EmissionSet | None = None) -> dict:
    ps = ps or PriceSet.from_options()
    es = es or EmissionSet.from_options()
    p = params()
    return {
        "prices": ps.to_dict(),
        "prices_meta": prices()["_meta"],
        "emissions": es.to_dict(),
        "emissions_meta": emission_factors()["_meta"],
        "parameters_meta": p["_meta"],
        "calibration": p.get("_calibration"),
        "catalogue_meta": catalogue()["_meta"],
        "rating_note": "EPC-style estimate: SAP-like formula with fixed standard prices and standard occupancy, not an official EPC.",
        "heat_pump_readiness_hlp": p["heat_pump"]["readiness_hlp"],
        "standing_charges": "excluded from bill savings",
        "export": "export income included at the SEG rate shown; exported electricity is not credited in carbon",
        "not_a_guarantee": DISCLAIMER,
    }


def clear_cache() -> None:
    _CACHE.clear()
