"""What-if (spec 11): a validated scenario -> rerun planner, savings, carbon and value.

Goal, budget, ticks and prices re-rank cached results; a new solar size is a property fact
and triggers a stage-1 rerun for the property.
"""
from __future__ import annotations

from engine.currency import from_display
from engine.inputs import CustomerInput, CustomiseOptions
from engine.pipeline import analyse, options_from_input
from llm.schema import Scenario


def apply_scenario(inp: CustomerInput, opts: CustomiseOptions | None, sc: Scenario) -> tuple[CustomerInput, CustomiseOptions]:
    o = options_from_input(inp, opts)
    tick = [r for r in o.tick if r not in sc.retrofits_remove] + [r for r in sc.retrofits_add if r not in o.tick]
    untick = [r for r in o.untick if r not in sc.retrofits_add] + [r for r in sc.retrofits_remove if r not in o.untick]
    o.tick, o.untick = tick, untick
    inp2 = inp
    if sc.solar_kwp:
        inp2 = inp.model_copy(update={"solar_size_kwp": sc.solar_kwp})
        if "solar_pv" not in o.tick:
            o.tick.append("solar_pv")
    # scenario money is in display currency (rupees); the engine works in GBP
    if sc.budget is not None:
        o.budget = from_display(sc.budget)
    if sc.goal:
        o.goal = sc.goal
    if sc.electricity_price:
        o.electricity_price = from_display(sc.electricity_price)
    if sc.gas_price:
        o.gas_price = from_display(sc.gas_price)
    if sc.grid_carbon_year:
        o.grid_carbon_year = sc.grid_carbon_year
    return inp2, o


def _plan_numbers(res: dict) -> dict:
    best = (res.get("plan") or {}).get("best")
    if not best:
        return {"retrofits": [], "bill_saving_gbp": 0, "carbon_saving_kg": 0, "cost_gbp": 0, "band_after": None}
    return {"retrofits": best["names"], "bill_saving_gbp": best["bill_saving_gbp"], "carbon_saving_kg": best["carbon_saving_kg"],
            "cost_gbp": best["cost_gbp"][1], "band_after": best["band_after"]}


def run_whatif(inp: CustomerInput, sc: Scenario, opts: CustomiseOptions | None = None, with_ranges: bool = True) -> dict:
    before = analyse(inp, opts, with_ranges=False)
    inp2, o2 = apply_scenario(inp, opts, sc)
    after = analyse(inp2, o2, with_ranges=with_ranges)
    if after.get("status") != "ok":
        return {"scenario": sc.model_dump(), "results": after, "diff": None, "requested_infeasible": []}
    b, a = _plan_numbers(before), _plan_numbers(after)
    diff = {
        "plan_before": b["retrofits"], "plan_after": a["retrofits"],
        "bill_saving_before_gbp": b["bill_saving_gbp"], "bill_saving_after_gbp": a["bill_saving_gbp"],
        "carbon_saving_before_kg": b["carbon_saving_kg"], "carbon_saving_after_kg": a["carbon_saving_kg"],
        "cost_before_gbp": b["cost_gbp"], "cost_after_gbp": a["cost_gbp"],
        "band_after_before": b["band_after"], "band_after_now": a["band_after"],
    }
    na = {n["retrofit"] for n in after["retrofits"]["not_available"]}
    return {
        "scenario": sc.model_dump(),
        "options_used": o2.model_dump(),
        "results": after,
        "diff": diff,
        "requested_infeasible": [r for r in sc.retrofits_add if r in na],
    }
