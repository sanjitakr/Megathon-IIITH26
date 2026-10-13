"""Customer answers -> simulator HomeState, recording every default that was filled in."""
from __future__ import annotations

from engine.inputs import CustomerInput
from sim.home import HomeState
from sim.parameters import params, postcode_to_region, sampling_priors
from sim.retrofits import ORDER, apply_retrofit
from sim.simulator import standard_occupancy

AGE_GROUP_DEFAULTS = {
    "floor_type": {"old": "suspended", "mid": "solid", "new": "insulated"},
    "glazing": {"old": "double_old", "mid": "double_old", "new": "double_modern"},
    "draught_proofed": {"old": False, "mid": False, "new": True},
}


def _age_group(age_band: str) -> str:
    return sampling_priors()["age_group"][age_band]


def default_wall(age_band: str) -> str:
    dist = sampling_priors()["wall_given_age"][age_band]
    return max(dist, key=dist.get)


def build_state(inp: CustomerInput) -> tuple[HomeState, list[dict]]:
    """Assumes validation passed (type, age, area, heating and region are known)."""
    p = params()
    defaults: list[dict] = []

    def assume(field: str, value, reason: str):
        defaults.append({"field": field, "assumed": value, "reason": reason})
        return value

    group = _age_group(inp.age_band)
    region = postcode_to_region(inp.postcode)

    storeys = inp.storeys or assume("storeys", p["default_storeys"][inp.property_type], "typical for the property type")
    bedrooms = inp.bedrooms or assume("bedrooms", int(min(6, max(1, round(inp.floor_area / 30)))), "estimated from floor area")
    if inp.occupants:
        occupants = inp.occupants
    elif inp.bedrooms:
        occupants = assume("occupants", round(min(6.0, 0.85 + 0.55 * inp.bedrooms), 1), "estimated from bedrooms")
    else:
        occupants = assume("occupants", round(standard_occupancy(inp.floor_area), 1), "standard occupancy for the floor area")

    wall = inp.wall_type if inp.wall_type != "not_sure" else assume(
        "wall_type", default_wall(inp.age_band), "most common wall type for the age band"
    )

    if inp.roof_type != "not_sure":
        roof_type = inp.roof_type
    else:
        roof_type = assume(
            "roof_type",
            "flat" if (inp.loft == "none_or_flat" and inp.property_type == "flat") else "pitched",
            "typical for the property type",
        )

    if inp.loft == "not_sure":
        loft = "flat_roof" if roof_type == "flat" else assume("loft", "100_250", "most common loft depth")
    elif inp.loft == "none_or_flat":
        loft = "flat_roof" if roof_type == "flat" else "none"
    else:
        loft = inp.loft
        if roof_type == "flat":
            loft = "flat_roof"

    floor = inp.floor_type if inp.floor_type != "not_sure" else assume(
        "floor_type", AGE_GROUP_DEFAULTS["floor_type"][group], "typical for the age band"
    )
    glazing = inp.glazing if inp.glazing != "not_sure" else assume(
        "glazing", AGE_GROUP_DEFAULTS["glazing"][group], "typical for the age band"
    )
    if inp.draught_proofed == "not_sure":
        draught = assume("draught_proofed", AGE_GROUP_DEFAULTS["draught_proofed"][group], "typical for the age band")
    else:
        draught = inp.draught_proofed == "yes"

    heating = inp.main_heating
    boiler_age = "5_10"
    if heating in ("gas_boiler", "oil_boiler", "lpg_boiler"):
        boiler_age = inp.boiler_age if inp.boiler_age != "not_sure" else assume("boiler_age", "5_10", "most common boiler age")

    controls = inp.controls if inp.controls != "not_sure" else assume("controls", "programmer", "most common controls")

    if inp.hot_water != "not_sure":
        hot_water = inp.hot_water
    else:
        hw_default = {"electric": "immersion", "heat_pump": "heat_pump"}.get(heating, "boiler")
        hot_water = assume("hot_water", hw_default, "follows the main heating system")
    if heating == "electric" and hot_water == "boiler":
        hot_water = "immersion"

    gas_usage_given = inp.gas is not None and not inp.gas.is_empty()
    if inp.gas_cooking == "not_sure":
        gas_cooking = assume(
            "gas_cooking",
            heating == "gas_boiler" or gas_usage_given,
            "assumed gas hob where the home has a gas supply",
        )
    else:
        gas_cooking = inp.gas_cooking == "yes"

    outdoor = {"yes": True, "no": False, "not_sure": None}[inp.outdoor_space]
    if outdoor is None:
        defaults.append({"field": "outdoor_space", "assumed": "unknown", "reason": "heat pump siting needs to be confirmed"})
    heritage = {"not_sure": "unknown"}.get(inp.heritage, inp.heritage)
    orientation = "unknown" if inp.orientation == "not_sure" else inp.orientation
    shading = "unknown" if inp.shading == "not_sure" else inp.shading

    state = HomeState(
        property_type=inp.property_type,
        age_band=inp.age_band,
        floor_area=float(inp.floor_area),
        region=region,
        storeys=int(storeys),
        occupants=float(occupants),
        bedrooms=int(bedrooms),
        flat_position=inp.flat_position if inp.flat_position != "not_sure" else "unknown",
        wall_type=wall,
        loft=loft,
        roof_type=roof_type,
        floor_type=floor,
        glazing=glazing,
        draught_proofed=bool(draught),
        heating=heating,
        boiler_age=boiler_age,
        controls=controls,
        tariff_aware=inp.tariff_aware,
        hot_water=hot_water,
        gas_cooking=bool(gas_cooking),
        solar_kwp=float(inp.solar_kwp or 0.0),
        orientation=orientation,
        shading=shading,
        roof_free_area_m2=inp.roof_free_area_m2,
        preferred_solar_kwp=inp.solar_size_kwp,
        battery_kwh=5.0 if inp.battery else 0.0,
        outdoor_space=outdoor,
        heritage=heritage,
        tenure=inp.tenure,
    )

    # Completed retrofits update the baseline (customer memory)
    for rid in [r for r in ORDER if r in inp.past_retrofits]:
        if rid == "solar_pv" and state.solar_kwp > 0:
            continue
        state = apply_retrofit(state, rid)
    return state, defaults
