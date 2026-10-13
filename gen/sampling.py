"""Property sampling (spec 6.1): a conditional chain
region -> type -> age -> area | type -> fabric | age -> heating | age, region.

Distributions come from data/reference/sampling_priors.json. fit_from_epc() can re-estimate
the categorical conditionals from an EPC open-data extract (no real rows are copied).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from sim.home import HomeState
from sim.parameters import REFERENCE_DIR, params, regions, sampling_priors
from sim.rating import rating


def _choice(rng: np.random.Generator, dist: dict) -> str:
    keys = list(dist)
    p = np.asarray([dist[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _cond(table: dict, key: str) -> dict:
    return table.get(key, table["default"])


def synthetic_postcode(rng: np.random.Generator, region: str) -> str:
    areas = regions()["postcode_area_to_region"][region]
    area = areas[rng.integers(len(areas))]
    return f"{area}{rng.integers(1, 30)} {rng.integers(1, 10)}{'ABDEFGHJLNPQRSTUWXYZ'[rng.integers(20)]}{'ABDEFGHJLNPQRSTUWXYZ'[rng.integers(20)]}"


def sample_home(rng: np.random.Generator, pri: dict | None = None) -> tuple[HomeState, dict]:
    pri = pri or sampling_priors()
    p = params()
    region = _choice(rng, pri["region"])
    ptype = _choice(rng, _cond(pri["type_given_region"], region))
    age = _choice(rng, _cond(pri["age_given_type"], ptype))
    group = pri["age_group"][age]
    a = pri["area_given_type"][ptype]
    area = float(np.clip(a["median"] * math.exp(rng.normal(0, a["sigma"])), 25, 600))
    storeys = p["default_storeys"][ptype]
    if ptype == "detached" and area > 190 and rng.random() < 0.3:
        storeys = 3
    if ptype == "terraced" and age in ("A", "B") and rng.random() < 0.15:
        storeys = 3
    bedrooms = int(np.clip(round(area / 28 + rng.normal(0, 0.6)), 1, 7))
    occupants = float(np.clip(round(0.85 + 0.55 * bedrooms + rng.normal(0, 0.8)), 1, 7))

    wall = _choice(rng, pri["wall_given_age"][age])
    roof_type = "flat" if rng.random() < pri["flat_roof_share"][ptype] else "pitched"
    loft = "flat_roof" if roof_type == "flat" else _choice(rng, pri["loft_given_age"][group])
    floor = _choice(rng, pri["floor_given_age"][group])
    glazing = _choice(rng, pri["glazing_given_age"][group])
    draught = rng.random() < pri["draught_proofed"][group]

    heat_dist = dict(_cond(pri["heating_given_region"], region))
    if ptype == "flat":
        heat_dist["electric"] += pri["flat_electric_heating_boost"]
    if age in ("J", "K"):
        heat_dist["heat_pump"] += pri["new_build_heat_pump_boost"]
    heating = _choice(rng, heat_dist)
    boiler_age = _choice(rng, pri["boiler_age"])
    controls = _choice(rng, pri["controls"])
    if heating == "electric":
        hot_water = "immersion"
    elif heating == "heat_pump":
        hot_water = "heat_pump"
    else:
        r = rng.random()
        hot_water = "immersion" if r < 0.05 else ("solar_thermal" if r < 0.07 else "boiler")
    gas_cooking = (heating == "gas_boiler" and rng.random() < 0.7) or (heating != "gas_boiler" and region != "northern_ireland" and rng.random() < 0.04)

    flat_position = "unknown"
    if ptype == "flat":
        flat_position = ["ground", "middle", "top"][rng.integers(3)]
    has_solar = rng.random() < pri["solar_probability"] and not (ptype == "flat" and flat_position != "top")
    solar_kwp = 0.0
    battery = 0.0
    orientation = _choice(rng, pri["orientation"])
    shading = _choice(rng, pri["shading"])
    if has_solar:
        s = pri["solar_kwp"]
        solar_kwp = round(float(np.clip(s["median"] * math.exp(rng.normal(0, s["sigma"])), 1.0, 8.0)), 1)
        battery = 5.0 if rng.random() < pri["battery_given_solar"] else 0.0
        if orientation == "N":
            orientation = "E_W"
    outdoor = bool(rng.random() < pri["outdoor_space_by_type"][ptype])
    heritage = _choice(rng, pri["heritage"])
    if heritage == "listed" and age not in ("A", "B", "C"):
        heritage = "no"
    tenure = _choice(rng, pri["tenure"])

    hidden = pri["hidden"]
    hidden_state = {
        "behaviour": float(math.exp(rng.normal(0, hidden["behaviour_sigma"]))),
        "build_quality": float(math.exp(rng.normal(0, hidden["build_quality_sigma"]))),
        "appliance_hidden": float(math.exp(rng.normal(0, hidden["appliance_sigma"]))),
        "dhw_hidden": float(math.exp(rng.normal(0, 0.2))),
        "setpoint_c": float(np.clip(19.5 + rng.normal(0, 1.2), 16, 23)),
    }
    state = HomeState(
        property_type=ptype, age_band=age, floor_area=round(area, 1), region=region, storeys=storeys,
        occupants=occupants, bedrooms=bedrooms, flat_position=flat_position, wall_type=wall, loft=loft,
        roof_type=roof_type, floor_type=floor, glazing=glazing, draught_proofed=bool(draught), heating=heating,
        boiler_age=boiler_age, controls=controls, hot_water=hot_water, gas_cooking=bool(gas_cooking),
        solar_kwp=solar_kwp, orientation=orientation, shading=shading, battery_kwh=battery,
        outdoor_space=outdoor, heritage=heritage, tenure=tenure,
    )
    return state, hidden_state


def sample_population(n: int, seed: int = 42) -> pd.DataFrame:
    """n properties with customer ids (some customers own 2-3 properties, for group splits)."""
    rng = np.random.default_rng(seed)
    pri = sampling_priors()
    rows = []
    customer = 0
    i = 0
    while i < n:
        k = int(rng.choice([1, 2, 3], p=[0.88, 0.09, 0.03]))
        for _ in range(min(k, n - i)):
            state, hidden = sample_home(rng, pri)
            rows.append({
                "property_id": f"P{i:06d}",
                "customer_id": f"C{customer:06d}",
                "postcode": synthetic_postcode(rng, state.region),
                "state_json": json.dumps(state.to_dict()),
                **{f"hidden_{k2}": v for k2, v in hidden.items()},
            })
            i += 1
        customer += 1
    return pd.DataFrame(rows)


# ------------------------------------------------------------- customer view

FORM_WALL = {"system_insulated": "solid_insulated"}


def form_view(state: HomeState, postcode: str, rng: np.random.Generator) -> dict:
    """What the customer would type: true facts with 'not sure' injected, EPC sometimes known."""
    pri = sampling_priors()["observation"]
    ns = pri["not_sure_rate"]

    def maybe(value):
        return "not_sure" if rng.random() < ns else value

    loft = "none_or_flat" if state.loft in ("none", "flat_roof") else state.loft
    epc = "not_sure"
    if rng.random() < pri["epc_known_rate"]:
        band = rating(state)["band"]
        bands = "ABCDEFG"
        j = bands.index(band)
        if rng.random() < 0.2:  # older methodology / assessor variation
            j = int(np.clip(j + rng.choice([-1, 1]), 0, 6))
        epc = bands[j]
    return {
        "postcode": postcode,
        "property_type": state.property_type,
        "age_band": state.age_band,
        "floor_area": state.floor_area,
        "bedrooms": state.bedrooms,
        "occupants": state.occupants if rng.random() > 0.3 else None,
        "storeys": state.storeys if rng.random() > 0.5 else None,
        "flat_position": state.flat_position if state.flat_position != "unknown" else "not_sure",
        "wall_type": maybe(FORM_WALL.get(state.wall_type, state.wall_type)),
        "loft": maybe(loft),
        "floor_type": maybe(state.floor_type),
        "glazing": maybe(state.glazing),
        "draught_proofed": maybe("yes" if state.draught_proofed else "no"),
        "main_heating": state.heating,
        "boiler_age": maybe(state.boiler_age),
        "controls": maybe(state.controls),
        "hot_water": maybe(state.hot_water),
        "gas_cooking": maybe("yes" if state.gas_cooking else "no"),
        "solar_kwp": state.solar_kwp,
        "battery": state.battery_kwh > 0,
        "roof_type": maybe(state.roof_type),
        "orientation": maybe(state.orientation if state.orientation != "unknown" else "not_sure"),
        "shading": maybe(state.shading if state.shading != "unknown" else "not_sure"),
        "outdoor_space": maybe("yes" if state.outdoor_space else "no"),
        "heritage": maybe(state.heritage),
        "tenure": state.tenure,
        "epc_band": epc,
    }


def fit_from_epc(csv_path: str, out_path: str | None = None) -> dict:
    """Re-estimate conditional category frequencies from an EPC extract.

    Expects columns already mapped to our vocabulary: region, property_type, age_band,
    wall_type, glazing, main_heating (see docs/data_sources.md). Writes a new priors file.
    """
    df = pd.read_csv(csv_path)
    pri = json.loads(json.dumps(sampling_priors()))
    pri["region"] = df["region"].value_counts(normalize=True).to_dict()
    pri["type_given_region"] = {"default": df["property_type"].value_counts(normalize=True).to_dict()}
    for r, g in df.groupby("region"):
        pri["type_given_region"][r] = g["property_type"].value_counts(normalize=True).to_dict()
    pri["age_given_type"] = {"default": df["age_band"].value_counts(normalize=True).to_dict()}
    for t, g in df.groupby("property_type"):
        pri["age_given_type"][t] = g["age_band"].value_counts(normalize=True).to_dict()
    for a, g in df.groupby("age_band"):
        pri["wall_given_age"][a] = g["wall_type"].value_counts(normalize=True).to_dict()
    pri["heating_given_region"] = {"default": df["main_heating"].value_counts(normalize=True).to_dict()}
    for r, g in df.groupby("region"):
        pri["heating_given_region"][r] = g["main_heating"].value_counts(normalize=True).to_dict()
    pri["_meta"]["status"] = f"Fitted from EPC extract {csv_path}"
    out = out_path or str(REFERENCE_DIR / "sampling_priors.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(pri, fh, indent=2)
    return pri
