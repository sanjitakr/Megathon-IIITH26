"""Feasibility rules (spec 6.8) and heat pump readiness (spec 4.4).

Retrofits that fail are hidden from the list; their reasons are kept to explain an
infeasible request.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.costs import cost_range
from sim.geometry import compute_geometry, usable_roof_area
from sim.home import HomeState
from sim.parameters import catalogue, params
from sim.retrofits import ORDER, apply_package, sort_in_tier_order
from sim.simulator import heat_loss

FABRIC = ["loft_insulation", "cavity_wall_insulation", "solid_wall_insulation", "floor_insulation", "double_glazing", "draught_proofing"]
LOFT_LABEL = {"none": "no insulation", "under_100": "under 100 mm", "100_250": "100 to 250 mm", "over_250": "over 250 mm"}


@dataclass
class Feasibility:
    feasible: dict[str, dict] = field(default_factory=dict)
    infeasible: dict[str, str] = field(default_factory=dict)
    hp_readiness: dict = field(default_factory=dict)

    def feasible_ids(self) -> list[str]:
        return sort_in_tier_order(self.feasible)

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "not_available": [{"retrofit": r, "name": catalogue()["retrofits"][r]["name"], "reason": why} for r, why in self.infeasible.items()],
            "hp_readiness": self.hp_readiness,
        }


def _hlp(state: HomeState) -> float:
    return heat_loss(state)[0] / state.floor_area


def _fabric_rules(state: HomeState, geom) -> tuple[dict, dict]:
    ok, no = {}, {}
    heritage = state.heritage
    # Loft
    if geom.roof_exposure == 0:
        no["loft_insulation"] = "No roof of its own (flat below the top floor)."
    elif state.roof_type == "flat" or state.loft == "flat_roof":
        no["loft_insulation"] = "Flat roof: a loft top-up does not apply."
    elif state.loft == "over_250":
        no["loft_insulation"] = "Loft already has over 250 mm of insulation."
    else:
        ok["loft_insulation"] = {"reason": f"Loft insulation is {LOFT_LABEL[state.loft]}, below the 270 mm target.", "flags": []}
    # Walls
    if state.wall_type == "cavity_unfilled":
        ok["cavity_wall_insulation"] = {"reason": "Cavity walls are unfilled.", "flags": []}
        no["solid_wall_insulation"] = "Walls are cavity walls: cavity fill is the right measure."
    elif state.wall_type in ("solid", "system_build"):
        no["cavity_wall_insulation"] = "Walls have no cavity to fill."
        if heritage == "listed":
            no["solid_wall_insulation"] = "Listed building: wall insulation is restricted without consent."
        else:
            flags = []
            if heritage == "conservation":
                flags.append("Conservation area: external insulation may be refused; internal insulation assumed.")
            if heritage == "unknown":
                flags.append("Check whether the home is listed or in a conservation area.")
            ok["solid_wall_insulation"] = {"reason": f"{'Solid' if state.wall_type == 'solid' else 'System-built'} walls are uninsulated.", "flags": flags}
    else:
        no["cavity_wall_insulation"] = "Walls are already insulated."
        no["solid_wall_insulation"] = "Walls are already insulated."
    # Floor
    if geom.floor_exposure == 0:
        no["floor_insulation"] = "No ground floor (upper-floor flat)."
    elif state.floor_type == "suspended":
        ok["floor_insulation"] = {"reason": "Suspended timber floor is uninsulated.", "flags": []}
    elif state.floor_type == "solid":
        no["floor_insulation"] = "Solid floor: insulating it needs the floor dug up, so it is not offered."
    else:
        no["floor_insulation"] = "Floor is already insulated."
    # Glazing
    if state.glazing in ("double_modern", "triple"):
        no["double_glazing"] = "Glazing is already modern double or triple."
    elif heritage == "listed":
        no["double_glazing"] = "Listed building: replacement windows are restricted (secondary glazing may be possible)."
    else:
        flags = ["Conservation area: window style may need approval."] if heritage == "conservation" else []
        ok["double_glazing"] = {"reason": f"Current glazing is {state.glazing.replace('_', ' ')}.", "flags": flags}
    # Draught-proofing
    if state.draught_proofed:
        no["draught_proofing"] = "Home is already draught-proofed."
    else:
        ok["draught_proofing"] = {"reason": "No draught-proofing recorded.", "flags": []}
    return ok, no


def assess(state: HomeState) -> Feasibility:
    p = params()
    cat = catalogue()["retrofits"]
    geom = compute_geometry(state)
    fz = Feasibility()
    ok, no = _fabric_rules(state, geom)
    fz.feasible.update(ok)
    fz.infeasible.update(no)
    renting = state.tenure == "rent"

    # Heat pump
    thr = p["heat_pump"]["readiness_hlp"]
    hlp_now = _hlp(state)
    fabric_ok = [r for r in FABRIC if r in fz.feasible]
    hlp_all_fabric = _hlp(apply_package(state, fabric_ok)) if fabric_ok else hlp_now
    fz.hp_readiness = {"hlp_now": round(hlp_now, 2), "hlp_with_all_fabric": round(hlp_all_fabric, 2), "threshold": thr, "ready_now": hlp_now <= thr}
    if state.heating == "heat_pump":
        fz.infeasible["heat_pump"] = "Home already has a heat pump."
    elif renting:
        fz.infeasible["heat_pump"] = "Rented home: a heat pump is the property owner's decision (landlord consent needed)."
    elif state.outdoor_space is False:
        fz.infeasible["heat_pump"] = "No outdoor space for the heat pump unit."
    elif hlp_now > thr and hlp_all_fabric > thr:
        fz.infeasible["heat_pump"] = (
            f"Heat loss is too high for a heat pump to run efficiently (HLP {hlp_now:.2f} W/m2K, and "
            f"{hlp_all_fabric:.2f} even with every feasible fabric measure; readiness threshold {thr:.1f})."
        )
    else:
        prereq = [] if hlp_now <= thr else minimal_prerequisites(state, fabric_ok, thr)
        flags = []
        if state.outdoor_space is None:
            flags.append("Confirm there is outdoor space for the unit.")
        if state.heritage in ("listed", "conservation"):
            flags.append("Heritage status: planning permission may be needed for the outdoor unit.")
        if state.heating_fuel == "electricity":
            reason = "Replaces direct electric heating with a far more efficient system."
        else:
            reason = f"Replaces the {state.heating.replace('_', ' ')} with low-carbon heating."
        if prereq:
            reason += f" Needs fabric first to bring HLP from {hlp_now:.2f} to at or below {thr:.1f} W/m2K."
        fz.feasible["heat_pump"] = {"reason": reason, "flags": flags, "prerequisites": prereq}
    fz.hp_readiness["prerequisites"] = fz.feasible.get("heat_pump", {}).get("prerequisites", [])

    # Solar
    sp = p["solar"]
    roof_free = usable_roof_area(state, geom)
    if state.solar_kwp > 0:
        fz.infeasible["solar_pv"] = "Home already has solar panels."
    elif renting:
        fz.infeasible["solar_pv"] = "Rented home: solar panels are the property owner's decision (landlord consent needed)."
    elif state.heritage == "listed":
        fz.infeasible["solar_pv"] = "Listed building: roof panels are restricted."
    elif geom.roof_exposure == 0 or roof_free < sp["min_free_roof_m2"]:
        fz.infeasible["solar_pv"] = f"Not enough free roof area ({roof_free:.0f} m2; at least {sp['min_free_roof_m2']:.0f} m2 needed)."
    elif state.orientation == "N":
        fz.infeasible["solar_pv"] = "Roof faces north."
    elif state.shading == "heavy":
        fz.infeasible["solar_pv"] = "Roof is heavily shaded."
    else:
        flags = []
        if state.orientation == "unknown":
            flags.append("Roof orientation not given: an average orientation is assumed.")
        if state.heritage == "conservation":
            flags.append("Conservation area: panels on a front roof may need approval.")
        fz.feasible["solar_pv"] = {"reason": f"About {roof_free:.0f} m2 of usable roof.", "flags": flags}

    # Battery
    has_solar = state.solar_kwp > 0
    if state.battery_kwh > 0:
        fz.infeasible["battery"] = "Home already has a battery."
    elif renting:
        fz.infeasible["battery"] = "Rented home: a battery is the property owner's decision."
    elif not has_solar and "solar_pv" not in fz.feasible:
        fz.infeasible["battery"] = "A battery here stores solar power, and solar is not available for this home."
    else:
        fz.feasible["battery"] = {
            "reason": "Stores daytime solar for evening use.",
            "flags": [],
            "prerequisites": [] if has_solar else ["solar_pv"],
        }

    # Smart controls
    if state.controls == "smart_trvs":
        fz.infeasible["smart_controls"] = "Smart zoning or TRVs are already fitted."
    elif state.heating == "electric":
        fz.infeasible["smart_controls"] = "Direct electric or storage heaters need their own controls (not a wet-system thermostat)."
    else:
        fz.feasible["smart_controls"] = {"reason": f"Current controls: {state.controls.replace('_', ' ')}.", "flags": []}

    for rid, info in fz.feasible.items():
        info.setdefault("flags", [])
        info.setdefault("prerequisites", [])
        info["needs_landlord_consent"] = renting and cat[rid].get("needs_landlord_consent", False)
        if info["needs_landlord_consent"]:
            info["flags"].append("Rented home: needs landlord consent.")
    # keep catalogue order
    fz.feasible = {r: fz.feasible[r] for r in ORDER if r in fz.feasible}
    return fz


def minimal_prerequisites(state: HomeState, fabric_options: list[str], threshold: float) -> list[str]:
    """Greedy: add the fabric measure with the best HLC reduction per GBP until HLP <= threshold."""
    chosen: list[str] = []
    current = state
    while _hlp(current) > threshold:
        best, best_val = None, -1.0
        hlc_now = heat_loss(current)[0]
        for rid in fabric_options:
            if rid in chosen:
                continue
            trial = apply_package(current, [rid])
            gain = hlc_now - heat_loss(trial)[0]
            value = gain / max(cost_range(rid, state)[1], 1.0)
            if value > best_val:
                best, best_val = rid, value
        if best is None:
            break
        chosen.append(best)
        current = apply_package(state, chosen)
    return sort_in_tier_order(chosen)
