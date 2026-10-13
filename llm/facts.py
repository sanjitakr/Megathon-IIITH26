"""Compact, pre-rounded facts from the results JSON. The LLM sees only this, and the
verifier checks the reply against exactly this object."""
from __future__ import annotations

from engine.currency import to_display
from sim.parameters import catalogue

FUEL_LABEL = {"electricity": "electricity", "gas": "gas", "oil": "oil", "lpg": "LPG", "solid": "solid fuel"}


def _r(x, nd=0):
    if x is None:
        return None
    return round(float(x), nd) if nd else int(round(float(x)))


def _m(gbp):
    """Money for the LLM: converted to display currency (rupees), whole units."""
    return None if gbp is None else int(round(to_display(float(gbp))))


def _plan_facts(best: dict) -> dict:
    d = best["delta_kwh"]
    rng = best.get("delta_kwh_range") or {}
    out = {
        "retrofits": best["names"],
        "electricity_saving_kwh": _r(d["electricity"]),
        "gas_saving_kwh": _r(d["gas"]),
        "cost_inr_mid": _m(best["cost_gbp"][1]),
        "cost_range_inr": [_m(best["cost_gbp"][0]), _m(best["cost_gbp"][2])],
        "bill_saving_inr_per_year": _m(best["bill_saving_gbp"]),
        "payback_years": _r(best["payback_years"], 1) if best["payback_years"] is not None else "no payback (bills do not fall)",
        "carbon_saving_kg_per_year": _r(best["carbon_saving_kg"]),
        "carbon_saving_tonnes_per_year": _r(best["carbon_saving_kg"] / 1000, 1),
        "epc_style_band_after": best["band_after"],
        "electricity_rises": d["electricity"] < 0,
    }
    for f in ("oil", "lpg", "solid"):
        if abs(d.get(f, 0)) > 0.5:
            out[f"{f}_saving_kwh"] = _r(d[f])
    if rng.get("electricity"):
        out["electricity_saving_range_kwh_p10_p90"] = [_r(rng["electricity"][0]), _r(rng["electricity"][2])]
    if rng.get("gas"):
        out["gas_saving_range_kwh_p10_p90"] = [_r(rng["gas"][0]), _r(rng["gas"][2])]
    if best.get("bill_saving_range_gbp"):
        out["bill_saving_range_inr_p10_p90"] = [_m(best["bill_saving_range_gbp"][0]), _m(best["bill_saving_range_gbp"][2])]
    if best.get("score_after"):
        out["green_home_score_after"] = best["score_after"]["score_rounded"]
        out["green_home_class_after"] = best["score_after"]["class"]
    if best.get("electricity_rises_note"):
        out["why_electricity_rises"] = best["electricity_rises_note"]
    if best.get("scop"):
        out["heat_pump_scop"] = _r(best["scop"], 1)
    return out


def build_facts(results: dict, question: str = "", requested: list[str] | None = None, diff: dict | None = None) -> dict:
    if results.get("status") != "ok":
        return {
            "status": "insufficient_information",
            "questions_to_ask": [q["question"] for q in results.get("questions", [])],
            "conflicts": results.get("conflicts", []),
        }
    p, b, s, plan = results["property"], results["baseline"], results["score"], results["plan"]
    cur = s["current"]
    purchased = b["purchased_kwh"]
    baseline = {"electricity_kwh_per_year": _r(purchased["electricity"]), "gas_kwh_per_year": _r(purchased["gas"]),
                "bill_inr_per_year": _m(b["bill_gbp"]), "carbon_kg_per_year": _r(b["carbon_kg"]),
                "carbon_tonnes_per_year": _r(b["carbon_kg"] / 1000, 1)}
    for f in ("oil", "lpg", "solid"):
        if purchased.get(f, 0) > 0:
            baseline[f"{f}_kwh_per_year"] = _r(purchased[f])
    facts = {
        "status": "ok",
        "home": {"type": p["property_type"], "built": p["age_band_label"], "floor_area_m2": _r(p["floor_area_m2"]),
                 "region": p["region_label"], "heating": p["heating"].replace("_", " "), "tenure": p["tenure"]},
        "confidence": {"tier": results["confidence"]["tier"], "label": results["confidence"]["label"]},
        "baseline": baseline,
        "green_home_score": {"current": cur["score_rounded"], "class": cur["class"], "provisional": cur["provisional"]},
        "key_issues": [i["issue"] for i in results["efficiency_flag"]["key_issues"]],
        "epc_style_band_now": results["rating"]["recorded_band"] or results["rating"]["simulated"]["band"],
        "improvement_potential_pct": results["efficiency_flag"]["improvement_potential"].get("pct"),
        "goal": plan["goal"].replace("_", " "),
        "budget_inr": _m(plan.get("budget")),
        "currency": "Indian rupees (INR, ₹)",
        "recommended_plan": _plan_facts(plan["best"]) if plan.get("best") else None,
        "plan_message": plan.get("message"),
        "optional_extras": [
            {"retrofit": o["name"], "extra_bill_saving_inr": _m(o["extra_bill_saving_gbp"]), "cost_inr_mid": _m(o["cost_gbp"][1])}
            for o in plan.get("optional", [])[:3] if "extra_bill_saving_gbp" in o
        ],
        "not_suitable": [{"retrofit": n["name"], "reason": n["reason"]} for n in results["retrofits"]["not_available"]],
        "warnings": plan.get("warnings", [])[:4],
        "value": ({"uplift_pct_range": [results["value"]["uplift_pct_range"][0], results["value"]["uplift_pct_range"][2]],
                   "label": results["value"]["label"],
                   **({"demo_warning": results["value"]["warning"]} if results["value"].get("warning") else {}),
                   **({"note": results["value"]["note"]} if results["value"].get("note") else {})}
                  if results["value"].get("status") == "ok" else {"status": "cannot be estimated", "reason": results["value"].get("reason")}),
        "disclaimer": results["disclaimer"],
    }
    if results["value"].get("uplift_gbp_range"):
        facts["value"]["uplift_inr_range"] = [_m(results["value"]["uplift_gbp_range"][0]), _m(results["value"]["uplift_gbp_range"][2])]
    if requested:
        cat = catalogue()["retrofits"]
        na = {n["retrofit"]: n["reason"] for n in results["retrofits"]["not_available"]}
        facts["requested_but_not_suitable"] = [{"retrofit": cat[r]["name"], "reason": na[r]} for r in requested if r in na]
    if diff:
        facts["change_vs_previous_plan"] = {
            (k.replace("_gbp", "_inr") if k.endswith("_gbp") else k): (_m(v) if k.endswith("_gbp") else v) for k, v in diff.items()
        }
    return facts
