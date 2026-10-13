"""Retrofits change simulator inputs, never outputs (spec 6.4).

A package is applied in dependency order and then simulated once, so interactions
(e.g. fabric lowering HLP and so raising heat pump SCOP) emerge from the equations.
"""
from __future__ import annotations

from sim.geometry import compute_geometry, usable_roof_area
from sim.home import HomeState
from sim.parameters import catalogue, params

ORDER = [
    "loft_insulation",
    "cavity_wall_insulation",
    "solid_wall_insulation",
    "floor_insulation",
    "double_glazing",
    "draught_proofing",
    "heat_pump",
    "solar_pv",
    "battery",
    "smart_controls",
]

TIER_NAMES = {1: "Fabric", 2: "Heating", 3: "Generation", 4: "Controls"}


def tier(retrofit_id: str) -> int:
    return catalogue()["retrofits"][retrofit_id]["tier"]


def sort_in_tier_order(ids) -> list[str]:
    return sorted(ids, key=ORDER.index)


def effect_value(retrofit_id: str, key: str, draw: dict | None = None) -> float:
    """Mid value of an effect parameter unless a Monte Carlo draw overrides it."""
    if draw and key in draw:
        return draw[key]
    return catalogue()["retrofits"][retrofit_id]["effect"][key][1]


def recommended_kwp(state: HomeState) -> float:
    geom = compute_geometry(state)
    sp = params()["solar"]
    area = usable_roof_area(state, geom)
    roof_limit = area / sp["m2_per_kwp"]
    if state.preferred_solar_kwp:
        return round(max(0.0, min(state.preferred_solar_kwp, roof_limit)), 1)
    return round(max(0.0, min(sp["default_max_kwp"], roof_limit)), 1)


def apply_retrofit(state: HomeState, retrofit_id: str, draw: dict | None = None, solar_kwp: float | None = None) -> HomeState:
    s = state.copy()
    ov = s.overrides
    if retrofit_id == "loft_insulation":
        s.loft = "over_250"
        ov["u_roof"] = effect_value(retrofit_id, "u_roof", draw)
    elif retrofit_id == "cavity_wall_insulation":
        s.wall_type = "cavity_filled"
        ov["u_wall"] = effect_value(retrofit_id, "u_wall", draw)
    elif retrofit_id == "solid_wall_insulation":
        s.wall_type = "system_insulated" if s.wall_type == "system_build" else "solid_insulated"
        ov["u_wall"] = effect_value(retrofit_id, "u_wall", draw)
    elif retrofit_id == "floor_insulation":
        s.floor_type = "insulated"
        ov["u_floor"] = effect_value(retrofit_id, "u_floor", draw)
    elif retrofit_id == "double_glazing":
        s.glazing = "double_modern"
        ov["u_window"] = effect_value(retrofit_id, "u_window", draw)
    elif retrofit_id == "draught_proofing":
        s.draught_proofed = True
        ov["air_change_factor"] = effect_value(retrofit_id, "air_change_factor", draw)
    elif retrofit_id == "heat_pump":
        if s.heating != "heat_pump":
            # Cooking on gas (gas_cooking) is kept if it existed; hot water moves to the heat pump.
            s.backup_heating = None
            s.heating = "heat_pump"
            s.hp_fraction = 1.0
            s.hot_water = "heat_pump"
            ov.pop("boiler_efficiency", None)
        ov["scop"] = effect_value(retrofit_id, "scop", draw)
    elif retrofit_id == "solar_pv":
        s.solar_kwp = solar_kwp if solar_kwp is not None else recommended_kwp(state)
        ov["yield_factor"] = effect_value(retrofit_id, "yield_factor", draw)
        ov["f_sc_factor"] = effect_value(retrofit_id, "f_sc_factor", draw)
    elif retrofit_id == "battery":
        s.battery_kwh = effect_value(retrofit_id, "battery_kwh", draw)
        ov["eta_rt"] = effect_value(retrofit_id, "eta_rt", draw)
    elif retrofit_id == "smart_controls":
        s.controls = "smart_trvs"
        ov["controls_factor"] = effect_value(retrofit_id, "controls_factor", draw)
    else:
        raise ValueError(f"unknown retrofit {retrofit_id}")
    s.retrofits_applied = list(s.retrofits_applied) + [retrofit_id]
    return s


def apply_package(state: HomeState, retrofit_ids, draws: dict | None = None, solar_kwp: float | None = None) -> HomeState:
    """Apply retrofits in dependency order. draws maps retrofit_id -> {effect_key: value}."""
    s = state
    for rid in sort_in_tier_order(retrofit_ids):
        s = apply_retrofit(s, rid, (draws or {}).get(rid), solar_kwp=solar_kwp)
    return s
