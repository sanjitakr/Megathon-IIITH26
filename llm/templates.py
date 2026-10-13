"""Deterministic template explanations. Every number comes straight from the facts, so the
template always passes the verifier and keeps the demo reliable when no LLM is available."""
from __future__ import annotations

from engine.currency import fmt


def _gbp(x) -> str:
    """Amounts in facts are already in display currency (rupees)."""
    return fmt(x)


def _kwh(x) -> str:
    return f"{abs(x):,.0f} kWh"


def render(facts: dict, question_type: str = "general") -> str:
    if facts.get("status") != "ok":
        lines = ["We need a little more information before we can give a result."]
        lines += [f"- {q}" for q in facts.get("questions_to_ask", [])]
        lines += [f"- {c}" for c in facts.get("conflicts", [])]
        return "\n".join(lines)

    out = []
    for item in facts.get("requested_but_not_suitable", []):
        out.append(f"We can't recommend {item['retrofit'].lower()} for this home. {item['reason']} So we have not estimated savings for it.")

    gs = facts["green_home_score"]
    prov = " (provisional, because some inputs are estimated)" if gs["provisional"] else ""
    out.append(f"Your Green Home Score is {gs['current']} ({gs['class']}){prov}. Your EPC-style band is {facts['epc_style_band_now']}.")
    if facts.get("key_issues"):
        out.append("The main issues are: " + "; ".join(facts["key_issues"]) + ".")

    diff = facts.get("change_vs_previous_plan")
    if diff:
        out.append(
            f"With this change the plan's yearly bill saving goes from {_gbp(diff['bill_saving_before_inr'])} to {_gbp(diff['bill_saving_after_inr'])}, "
            f"and the carbon saving from {diff['carbon_saving_before_kg']:,.0f} kg to {diff['carbon_saving_after_kg']:,.0f} kg a year."
        )

    plan = facts.get("recommended_plan")
    if plan:
        out.append(f"For your goal ({facts['goal']}), the recommended package is: {', '.join(plan['retrofits'])}.")
        parts = []
        e, g = plan["electricity_saving_kwh"], plan["gas_saving_kwh"]
        if e < 0:
            parts.append(f"electricity use rises by {_kwh(e)} a year")
        elif e > 0:
            parts.append(f"electricity use falls by {_kwh(e)} a year")
        if g > 0:
            parts.append(f"gas use falls by {_kwh(g)} a year")
        for fuel, label in (("oil", "oil"), ("lpg", "LPG"), ("solid", "solid fuel")):
            if plan.get(f"{fuel}_saving_kwh"):
                parts.append(f"{label} use falls by {_kwh(plan[f'{fuel}_saving_kwh'])} a year")
        if parts:
            out.append("Estimated change: " + ", ".join(parts) + ".")
        if plan.get("electricity_rises"):
            out.append(plan.get("why_electricity_rises", "Electricity use goes up because heating moves to electricity."))
        cost = plan["cost_range_inr"]
        line = f"It costs about {_gbp(plan['cost_inr_mid'])} (range {_gbp(cost[0])} to {_gbp(cost[1])})"
        if plan["bill_saving_inr_per_year"] > 0:
            line += f" and saves about {_gbp(plan['bill_saving_inr_per_year'])} a year on bills"
            if isinstance(plan["payback_years"], (int, float)):
                line += f", paying back in about {plan['payback_years']} years"
        else:
            line += f"; at current prices it does not cut bills (change {_gbp(plan['bill_saving_inr_per_year'])} a year)"
        out.append(line + ".")
        out.append(f"It cuts carbon by about {plan['carbon_saving_tonnes_per_year']} tonnes a year and moves the EPC-style band to {plan['epc_style_band_after']}.")
        if "green_home_score_after" in plan:
            out.append(f"Your projected Green Home Score is {plan['green_home_score_after']} ({plan['green_home_class_after']}).")
    elif facts.get("plan_message"):
        out.append(facts["plan_message"])

    v = facts.get("value", {})
    if v.get("note"):
        out.append(v["note"])
    elif "uplift_pct_range" in v:
        lo, hi = v["uplift_pct_range"]
        out.append(f"Indicative value effect: {lo}% to {hi}% (an association, not a guarantee).")
        if v.get("demo_warning"):
            out.append(v["demo_warning"])
    else:
        out.append("We cannot estimate a change in property value for this home.")
    for w in facts.get("warnings", [])[:2]:
        out.append(w)
    out.append(facts["disclaimer"])
    return " ".join(out)
