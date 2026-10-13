"""Retrofit planner (spec 4).

Stage 1 (once per property): feasibility gate, then simulate every valid subset of the
feasible set F (<= 2^|F|), cache per-fuel kWh, cost, rating and score, and order the list
by dependency tier, then suitability and saving per GBP.

Stage 2 (customisation): goal, budget, disruption, ticks, presets and price/carbon
sliders re-rank the cached subset table only: a filter, an objective and a sort.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from engine.baseline import Baseline
from engine.carbon import EmissionSet, PriceSet, annual_bill, annual_carbon, payback_years, savings
from engine.costs import cost_range, disruption
from engine.currency import money
from engine.feasibility import Feasibility
from engine.inputs import CustomiseOptions
from engine.percentiles import percentile
from engine.uncertainty import mc_pair
from sim.home import FUELS, HomeState
from sim.parameters import catalogue, params, score_config
from sim.rating import band_from_points, band_lower_edge, rating
from sim.retrofits import ORDER, TIER_NAMES, apply_package, recommended_kwp, sort_in_tier_order, tier
from sim.simulator import Calibration, simulate

MAX_FEASIBLE_FOR_ENUMERATION = 11
TOLERANCE_LEVEL = {"low": 1, "medium": 2, "high": 3}


@dataclass
class SubsetEval:
    ids: tuple
    purchased: dict
    gross: dict
    generation: float
    export: float
    hlc: float
    hlp: float
    scop: float | None
    sap_sim: float
    sap_anchored: float
    band: str
    cost: list
    disruption: int
    readiness_unmet: bool
    score: dict | None = None
    state: HomeState | None = None

    def cost_mid(self) -> float:
        return self.cost[1]


@dataclass
class PlanContext:
    state: HomeState
    calib: Calibration
    baseline: Baseline
    feasibility: Feasibility
    feasible: list
    cache: dict
    base: SubsetEval
    sap_recorded: float | None
    feasible_ordered: list = field(default_factory=list)
    package_row: dict | None = None
    standalone: dict = field(default_factory=dict)
    apply_grants: bool = False
    truncated: list = field(default_factory=list)
    mc_draws: int = 60

    def get(self, ids) -> SubsetEval | None:
        return self.cache.get(frozenset(ids))


def _valid(ids: set, state: HomeState) -> bool:
    if "battery" in ids and not (state.solar_kwp > 0 or "solar_pv" in ids):
        return False
    return True


def _evaluate(state, calib, ids, baseline_sap_sim, sap_recorded, apply_grants, score_fn) -> SubsetEval:
    after = apply_package(state, ids) if ids else state
    sim = simulate(after, calib)
    sap_sim = rating(after)["sap_points"]
    sap_anchor = (sap_recorded + (sap_sim - baseline_sap_sim)) if sap_recorded is not None else sap_sim
    cost = [0.0, 0.0, 0.0]
    for rid in ids:
        c = cost_range(rid, state, apply_grants)
        cost = [cost[i] + c[i] for i in range(3)]
    thr = params()["heat_pump"]["readiness_hlp"]
    ev = SubsetEval(
        ids=tuple(sort_in_tier_order(ids)),
        purchased=sim.purchased,
        gross=sim.gross,
        generation=sim.annual("g"),
        export=sim.annual("X"),
        hlc=sim.hlc,
        hlp=sim.hlp,
        scop=sim.scop,
        sap_sim=sap_sim,
        sap_anchored=round(min(max(sap_anchor, 1.0), 120.0), 1),
        band=band_from_points(sap_anchor),
        cost=cost,
        disruption=max([disruption(r) for r in ids], default=0),
        readiness_unmet="heat_pump" in ids and sim.hlp > thr,
        state=after,
    )
    if score_fn is not None:
        ev.score = score_fn(ev)
    return ev


def build_context(
    state: HomeState,
    baseline: Baseline,
    feasibility: Feasibility,
    sap_recorded: float | None,
    score_fn: Callable[["SubsetEval", "PlanContext | None"], dict] | None = None,
    apply_grants: bool = False,
    mc_draws: int = 60,
) -> PlanContext:
    calib = baseline.calibration
    F = feasibility.feasible_ids()
    truncated = []
    if len(F) > MAX_FEASIBLE_FOR_ENUMERATION:
        truncated = F[MAX_FEASIBLE_FOR_ENUMERATION:]
        F = F[:MAX_FEASIBLE_FOR_ENUMERATION]
    base_sap = rating(state)["sap_points"]

    holder: dict = {}
    sf = (lambda ev: score_fn(ev, holder.get("ctx"))) if score_fn else None

    base_ev = _evaluate(state, calib, [], base_sap, sap_recorded, apply_grants, None)
    ctx = PlanContext(
        state=state, calib=calib, baseline=baseline, feasibility=feasibility, feasible=F,
        cache={frozenset(): base_ev}, base=base_ev, sap_recorded=sap_recorded,
        apply_grants=apply_grants, truncated=truncated, mc_draws=mc_draws,
    )
    holder["ctx"] = ctx
    if sf:
        base_ev.score = sf(base_ev)
    for k in range(1, len(F) + 1):
        for combo in itertools.combinations(F, k):
            ids = set(combo)
            if not _valid(ids, state):
                continue
            ctx.cache[frozenset(ids)] = _evaluate(state, calib, ids, base_sap, sap_recorded, apply_grants, sf)

    _order_rows(ctx)
    return ctx


# ---------------------------------------------------------------- stage 1 list

def _purchased_total(p: dict) -> float:
    return sum(p[f] for f in FUELS)


def delta_purchased(ctx: PlanContext, before: SubsetEval, after: SubsetEval) -> dict:
    return {f: before.purchased[f] - after.purchased[f] for f in FUELS}


def _prerequisites(ctx: PlanContext, rid: str) -> list[str]:
    return list(ctx.feasibility.feasible.get(rid, {}).get("prerequisites", []))


def suitability(ctx: PlanContext, rid: str, ps: PriceSet) -> dict:
    cfg = score_config()["suitability"]
    single = ctx.get([rid])
    if single is None:  # e.g. battery standalone needs solar
        prereq = _prerequisites(ctx, rid)
        single = ctx.get([rid] + prereq)
        ref = ctx.get(prereq) or ctx.base
    else:
        ref = ctx.base
    base_total = _purchased_total(ref.purchased)
    d_total = base_total - _purchased_total(single.purchased)
    pct = 100 * d_total / base_total if base_total > 0 else 0.0
    bill = annual_bill(ref.purchased, ref.export, ps) - annual_bill(single.purchased, single.export, ps)
    cost_mid = max(cost_range(rid, ctx.state, ctx.apply_grants)[1], 1.0)
    per_gbp = bill / cost_mid
    prereq = _prerequisites(ctx, rid)
    if not prereq:
        R = 1.0
    elif all(p in ctx.feasible for p in prereq):
        R = 0.5
    else:
        R = 0.0
    p_saving = percentile(rid, "pct_saving", pct)
    p_value = percentile(rid, "saving_per_gbp", per_gbp)
    si = cfg["w_saving"] * p_saving + cfg["w_readiness"] * R + cfg["w_value"] * p_value
    label = "High" if si >= cfg["high"] else ("Medium" if si >= cfg["medium"] else "Low")
    return {
        "si": round(si, 3), "label": label, "p_saving": round(p_saving, 3), "readiness": R,
        "p_value": round(p_value, 3), "standalone_pct_saving": round(pct, 2),
        "standalone_bill_saving_gbp": round(bill, 0), "saving_per_gbp": round(per_gbp, 4),
        "standalone_delta_kwh": {f: round(ref.purchased[f] - single.purchased[f], 0) for f in FUELS},
    }


def _order_rows(ctx: PlanContext) -> None:
    ps = PriceSet.from_options()
    for rid in ctx.feasible:
        ctx.standalone[rid] = suitability(ctx, rid, ps)
    ordered = sorted(
        ctx.feasible,
        key=lambda r: (tier(r), -ctx.standalone[r]["si"], -ctx.standalone[r]["saving_per_gbp"]),
    )
    # enforce prerequisites inside a tier (battery after solar)
    for rid in list(ordered):
        for pre in _prerequisites(ctx, rid):
            if pre in ordered and ordered.index(pre) > ordered.index(rid):
                ordered.remove(pre)
                ordered.insert(ordered.index(rid), pre)
    ctx.feasible_ordered = ordered


def _fmt_q(q: list, nd: int = 0) -> list:
    return [None if x is None else round(x, nd) for x in q]


def _row_dict(ctx, rank, rid, prev, cur, ps, es, mc, rows_above) -> dict:
    cat = catalogue()["retrofits"][rid]
    s = savings(prev.purchased, cur.purchased, prev.export, cur.export, ps, es)
    cost = cost_range(rid, ctx.state, ctx.apply_grants)
    st = ctx.standalone.get(rid, {})
    prereq = _prerequisites(ctx, rid)
    info = ctx.feasibility.feasible.get(rid, {})
    row = {
        "rank": rank,
        "retrofit": rid,
        "name": cat["name"],
        "tier": tier(rid),
        "tier_name": TIER_NAMES[tier(rid)],
        "suitability": st.get("label"),
        "suitability_index": st.get("si"),
        "saving_in_sequence_kwh": {f: round(s["delta_kwh"][f], 0) for f in FUELS},
        "saving_in_sequence_range_kwh": {f: _fmt_q(mc["delta_kwh"][f]) for f in FUELS} if mc else None,
        "standalone_saving_kwh": st.get("standalone_delta_kwh"),
        "cost_gbp": cost,
        "bill_saving_gbp": round(s["bill_saving_gbp"], 0),
        "bill_saving_range_gbp": _fmt_q(mc["bill_saving_gbp"]) if mc else None,
        "payback_years": None if payback_years(cost[1], s["bill_saving_gbp"]) is None else round(payback_years(cost[1], s["bill_saving_gbp"]), 1),
        "carbon_saving_kg": round(s["carbon_saving_kg"], 0),
        "carbon_saving_range_kg": _fmt_q(mc["carbon_saving_kg"]) if mc else None,
        "needs_first": [catalogue()["retrofits"][p]["name"] for p in prereq],
        "needs_first_ids": prereq,
        "prerequisite_in_list_above": all(p in rows_above for p in prereq),
        "reason": info.get("reason", ""),
        "flags": list(info.get("flags", [])),
        "needs_landlord_consent": info.get("needs_landlord_consent", False),
        "disruption": cat["disruption"],
    }
    if rid == "solar_pv":
        row["solar_kwp"] = recommended_kwp(ctx.state)
    if rid == "heat_pump" and cur.scop is not None:
        row["scop"] = round(cur.scop, 2)
    if s["delta_kwh"]["electricity"] < -0.5:
        row["flags"].append("Electricity use rises: the heat pump runs on electricity while gas or other fuel falls.")
    if s["bill_saving_gbp"] < 0:
        row["flags"].append("Bills rise at current prices; see the carbon saving.")
    return row


def feasible_list(ctx: PlanContext, ps: PriceSet, es: EmissionSet, with_ranges: bool = True) -> dict:
    """Stage-1 default view: rows in order, sequenced savings, and the package row."""
    rows = []
    prefix: list[str] = []
    prev = ctx.base
    for i, rid in enumerate(ctx.feasible_ordered):
        nxt = prefix + [rid]
        cur = ctx.get(nxt)
        if cur is None:
            continue
        mc = mc_pair(ctx.state, ctx.calib, prefix, nxt, ps, es, ctx.baseline.tier, n=ctx.mc_draws, seed=100 + i, apply_grants=ctx.apply_grants) if with_ranges else None
        rows.append(_row_dict(ctx, len(rows) + 1, rid, prev, cur, ps, es, mc, prefix))
        prefix, prev = nxt, cur
    package = package_summary(ctx, list(ctx.feasible_ordered), ps, es, with_ranges, seed=999)
    return {"rows": rows, "package": package, "note": "Savings in sequence assume the retrofits ranked above are done first; standalone savings are not additive. The package row is one simulator run, not a sum of rows."}


def package_summary(ctx: PlanContext, ids, ps: PriceSet, es: EmissionSet, with_ranges: bool = True, seed: int = 1) -> dict:
    ev = ctx.get(ids)
    if ev is None:
        return {"retrofits": sort_in_tier_order(ids), "valid": False}
    b = ctx.baseline
    s = savings(ctx.base.purchased, ev.purchased, ctx.base.export, ev.export, ps, es)
    after_purchased = {f: max(b.purchased[f] - s["delta_kwh"][f], 0.0) for f in FUELS}
    mc = mc_pair(ctx.state, ctx.calib, [], list(ids), ps, es, b.tier, n=ctx.mc_draws, seed=seed,
                 widen=1.0 if b.calibration_diag.get("within_tolerance", True) else 1.5,
                 apply_grants=ctx.apply_grants) if (with_ranges and ids) else None
    pb = payback_years(ev.cost[1], s["bill_saving_gbp"])
    out = {
        "retrofits": list(ev.ids),
        "names": [catalogue()["retrofits"][r]["name"] for r in ev.ids],
        "valid": True,
        "delta_kwh": {f: round(s["delta_kwh"][f], 0) for f in FUELS},
        "delta_kwh_range": {f: _fmt_q(mc["delta_kwh"][f]) for f in FUELS} if mc else None,
        "baseline_purchased_kwh": {f: round(b.purchased[f], 0) for f in FUELS},
        "after_purchased_kwh": {f: round(after_purchased[f], 0) for f in FUELS},
        "bill_baseline_gbp": round(annual_bill(b.purchased, b.export_kwh, ps), 0),
        "bill_after_gbp": round(annual_bill(b.purchased, b.export_kwh, ps) - s["bill_saving_gbp"], 0),
        "bill_saving_gbp": round(s["bill_saving_gbp"], 0),
        "bill_saving_range_gbp": _fmt_q(mc["bill_saving_gbp"]) if mc else None,
        "cost_gbp": ev.cost,
        "cost_range_gbp": _fmt_q(mc["cost_gbp"]) if mc else None,
        "payback_years": None if pb is None else round(pb, 1),
        "payback_range_years": _fmt_q(mc["payback_years"], 1) if mc else None,
        "carbon_baseline_kg": round(annual_carbon(b.purchased, es), 0),
        "carbon_saving_kg": round(s["carbon_saving_kg"], 0),
        "carbon_saving_range_kg": _fmt_q(mc["carbon_saving_kg"]) if mc else None,
        "sap_after": ev.sap_anchored,
        "band_after": ev.band,
        "hlc_after_w_per_k": round(ev.hlc, 0),
        "hlc_reduction_w_per_k": round(ctx.base.hlc - ev.hlc, 0),
        "hlp_after": round(ev.hlp, 2),
        "scop": None if ev.scop is None else round(ev.scop, 2),
        "readiness_unmet": ev.readiness_unmet,
        "disruption": {1: "low", 2: "medium", 3: "high"}.get(ev.disruption, "none"),
        "score_after": None if ev.score is None else {"score_rounded": ev.score["score_rounded"], "class": ev.score["class"], "score": ev.score["score"]},
        "no_bill_saving": s["bill_saving_gbp"] <= 0,
    }
    if s["delta_kwh"]["electricity"] < -0.5:
        out["electricity_rises_note"] = (
            "Electricity use goes up because heating moves from gas or another fuel to electricity; "
            "total purchased energy and carbon still fall."
        )
    return out


# ---------------------------------------------------------------- stage 2

PRESET_FIELDS = ("tick", "untick")


def apply_preset(ctx: PlanContext, opts: CustomiseOptions) -> CustomiseOptions:
    if not opts.preset:
        return opts
    preset = catalogue()["presets"][opts.preset]
    o = opts.model_copy(deep=True)
    tick = set(o.tick)
    untick = set(o.untick)
    if preset.get("tick_all_feasible"):
        tick |= set(ctx.feasible)
    tick |= {r for r in preset.get("tick", []) if r in ctx.feasible}
    if preset.get("tick_hp_prerequisites") and "heat_pump" in ctx.feasible:
        tick |= set(ctx.feasibility.feasible["heat_pump"]["prerequisites"])
    untick |= set(preset.get("untick", []))
    if preset.get("only_ticked"):
        untick |= set(ctx.feasible) - tick
    tick -= untick
    o.tick, o.untick = sorted(tick), sorted(untick)
    if "disruption" in preset:
        o.disruption_tolerance = preset["disruption"]
    if "budget" in preset and o.budget is None:
        o.budget = preset["budget"]
    return o


def objective(goal: str, ev: SubsetEval, ctx: PlanContext, ps: PriceSet, es: EmissionSet) -> tuple[float, bool]:
    """(value to maximise, satisfies hard constraint)."""
    base = ctx.base
    bill = annual_bill(base.purchased, base.export, ps) - annual_bill(ev.purchased, ev.export, ps)
    if goal == "cut_bills":
        return bill, True
    if goal == "cut_carbon":
        return annual_carbon(base.purchased, es) - annual_carbon(ev.purchased, es), True
    if goal == "reach_c":
        return -ev.cost_mid(), ev.sap_anchored >= band_lower_edge("C")
    if goal == "comfort":
        return base.hlc - ev.hlc, True
    # value for money
    if not ev.ids or bill <= 0:
        return 0.0, bool(ev.ids) is False
    return bill / max(ev.cost_mid(), 1.0), True


def customise(ctx: PlanContext, opts: CustomiseOptions, with_ranges: bool = True) -> dict:
    opts = apply_preset(ctx, opts)
    ps = PriceSet.from_options(opts)
    es = EmissionSet.from_options(opts)
    cat = catalogue()["retrofits"]
    warnings: list[str] = []
    excluded: dict[str, str] = {}

    F = set(ctx.feasible)
    tick = set(opts.tick)
    for rid in list(tick):
        if rid not in F:
            reason = ctx.feasibility.infeasible.get(rid, "not in the feasible list for this home")
            warnings.append(f"{cat.get(rid, {}).get('name', rid)} cannot be included: {reason}")
            tick.discard(rid)
    for rid in set(opts.untick):
        if rid in F:
            excluded[rid] = "unticked"
    for rid in set(opts.declined):
        if rid in F:
            excluded[rid] = "declined previously"
    limit = TOLERANCE_LEVEL[opts.disruption_tolerance]
    for rid in F:
        if disruption(rid) > limit and rid not in tick and rid not in excluded:
            excluded[rid] = f"disruption {cat[rid]['disruption']} is above your {opts.disruption_tolerance} limit"
    for rid in tick:
        if disruption(rid) > limit:
            warnings.append(f"{cat[rid]['name']} is above your disruption limit but was ticked, so it is kept.")
        if rid in excluded and excluded[rid] != "unticked":
            excluded.pop(rid)
    allowed = (F - set(excluded)) | tick

    for rid in tick:
        prereq = _prerequisites(ctx, rid)
        missing = [p for p in prereq if p not in tick]
        if missing:
            warnings.append(
                f"{cat[rid]['name']} was ticked without {', '.join(cat[p]['name'] for p in missing)}; "
                "the planner may add them, otherwise the heat pump runs with an efficiency penalty."
                if rid == "heat_pump" else
                f"{cat[rid]['name']} needs {', '.join(cat[p]['name'] for p in missing)} first."
            )

    candidates = [ev for key, ev in ctx.cache.items() if key <= allowed and tick <= key]
    in_budget = [ev for ev in candidates if opts.budget is None or ev.cost_mid() <= opts.budget]

    scored = []
    for ev in in_budget:
        val, ok = objective(opts.goal, ev, ctx, ps, es)
        scored.append((ev, val, ok))

    best = None
    message = None
    feasible_scored = [(ev, v) for ev, v, ok in scored if ok and (ev.ids or opts.goal == "reach_c")]
    if opts.goal == "reach_c":
        if feasible_scored:
            best = max(feasible_scored, key=lambda t: (t[1], -len(t[0].ids)))[0]
        else:
            closest = max(scored, key=lambda t: (t[0].sap_anchored, -t[0].cost_mid()), default=None)
            if closest:
                best = closest[0]
                message = f"No package within the constraints reaches band C; the closest reaches SAP {closest[0].sap_anchored:.0f} (band {closest[0].band})."
    else:
        positive = [(ev, v) for ev, v in feasible_scored if v > 1e-9]
        if positive:
            best = max(positive, key=lambda t: (round(t[1], 6), -t[0].cost_mid()))[0]

    if best is None and message is None:
        useful = []
        for ev in candidates:
            v, ok = objective(opts.goal, ev, ctx, ps, es)
            if ok and v > 1e-9 and ev.ids:
                useful.append(ev)
        if useful:
            cheapest = min(useful, key=lambda e: e.cost_mid())
            message = (
                f"No package fits the budget of {money(opts.budget)}. The cheapest useful option is "
                f"{', '.join(cat[r]['name'] for r in cheapest.ids)} at about {money(cheapest.cost_mid())}."
                if opts.budget is not None else "No package improves on the current home for this goal."
            )
            cheapest_summary = package_summary(ctx, cheapest.ids, ps, es, with_ranges, seed=5)
        else:
            message = "No feasible retrofit improves this home for the chosen goal."
            cheapest_summary = None
        return {
            "goal": opts.goal, "budget": opts.budget, "options": opts.model_dump(),
            "best": None, "message": message, "cheapest_useful": cheapest_summary,
            "optional": [], "frontier": _frontier(ctx, candidates, opts.goal, ps, es, None),
            "warnings": warnings, "excluded": excluded, "prices": ps.to_dict(), "emissions": es.to_dict(),
        }

    best_summary = package_summary(ctx, list(best.ids), ps, es, with_ranges, seed=11)
    # rows inside the best package, in tier order, savings in sequence
    rows, prefix, prev = [], [], ctx.base
    for rid in [r for r in ctx.feasible_ordered if r in best.ids]:
        nxt = prefix + [rid]
        cur = ctx.get(nxt)
        if cur is None:
            continue
        rows.append(_row_dict(ctx, len(rows) + 1, rid, prev, cur, ps, es, None, prefix))
        prefix, prev = nxt, cur
    best_summary["rows"] = rows
    if best.readiness_unmet:
        warnings.append(
            f"Heat pump readiness not met in this package (HLP {best.hlp:.2f} W/m2K above "
            f"{params()['heat_pump']['readiness_hlp']:.1f}); SCOP is reduced to {best.scop:.2f}."
        )
    if best_summary["no_bill_saving"]:
        warnings.append("This package does not cut bills at current prices; it is chosen for its carbon or comfort benefit.")

    optional = []
    for rid in [r for r in ctx.feasible_ordered if r not in best.ids and r in allowed]:
        ev = ctx.get(set(best.ids) | {rid})
        if ev is None:
            optional.append({"retrofit": rid, "name": cat[rid]["name"], "note": "needs " + ", ".join(cat[p]["name"] for p in _prerequisites(ctx, rid))})
            continue
        s = savings(best.purchased, ev.purchased, best.export, ev.export, ps, es)
        c = cost_range(rid, ctx.state, ctx.apply_grants)
        optional.append({
            "retrofit": rid, "name": cat[rid]["name"],
            "extra_saving_kwh": {f: round(s["delta_kwh"][f], 0) for f in FUELS},
            "extra_bill_saving_gbp": round(s["bill_saving_gbp"], 0),
            "extra_carbon_saving_kg": round(s["carbon_saving_kg"], 0),
            "cost_gbp": c,
            "within_budget": opts.budget is None or best.cost_mid() + c[1] <= opts.budget,
        })

    return {
        "goal": opts.goal,
        "budget": opts.budget,
        "options": opts.model_dump(),
        "best": best_summary,
        "message": message,
        "optional": optional,
        "frontier": _frontier(ctx, in_budget or candidates, opts.goal, ps, es, best),
        "warnings": warnings,
        "excluded": excluded,
        "prices": ps.to_dict(),
        "emissions": es.to_dict(),
    }


def _frontier(ctx, evs, goal, ps, es, best) -> list[dict]:
    pts = []
    for ev in evs:
        bill = annual_bill(ctx.base.purchased, ctx.base.export, ps) - annual_bill(ev.purchased, ev.export, ps)
        carbon = annual_carbon(ctx.base.purchased, es) - annual_carbon(ev.purchased, es)
        y = {"cut_carbon": carbon, "comfort": ctx.base.hlc - ev.hlc, "reach_c": ev.sap_anchored}.get(goal, bill)
        pts.append({"retrofits": list(ev.ids), "cost_gbp": ev.cost_mid(), "saving": round(y, 1),
                    "bill_saving_gbp": round(bill, 0), "carbon_saving_kg": round(carbon, 0),
                    "sap": ev.sap_anchored, "is_best": best is not None and ev.ids == best.ids})
    pts.sort(key=lambda p: (p["cost_gbp"], -p["saving"]))
    best_y = -np.inf
    for p in pts:
        p["on_frontier"] = p["saving"] > best_y
        if p["on_frontier"]:
            best_y = p["saving"]
    return pts


def improvement_potential(ctx: PlanContext) -> dict:
    """100 x (E_base - E_after)/E_base with E = grid electricity + gas, full feasible package."""
    b = ctx.baseline
    full = ctx.get(ctx.feasible)
    E_base = b.purchased["electricity"] + b.purchased["gas"]
    if full is None or E_base <= 0:
        return {"pct": None, "reason": "unavailable: baseline grid electricity plus gas is zero or missing"}
    d = (ctx.base.purchased["electricity"] - full.purchased["electricity"]) + (ctx.base.purchased["gas"] - full.purchased["gas"])
    out = {"pct": round(100 * d / E_base, 1), "package": list(full.ids), "definition": "grid electricity + gas, full feasible package"}
    if b.purchased["oil"] + b.purchased["lpg"] + b.purchased["solid"] > 0:
        out["note"] = "Oil, LPG and solid fuel are excluded from this metric (they are shown in the savings table)."
    return out
