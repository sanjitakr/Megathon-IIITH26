"""Features (spec 7): raw customer inputs (with 'not_sure' kept as its own category),
plus physics features computed from the defaults-filled state: HLC, HLP, envelope to
floor-area ratio, regional HDD, age-band default U-values, solar yield estimate and the
simulator's own estimate (used for residual learning).

The same function serves training and prediction, so there is no train/serve skew.
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

from engine.inputs import CustomerInput
from engine.state_builder import build_state
from sim.parameters import region_hdd, regions
from sim.simulator import Calibration, simulate

CAT_COLS = [
    "property_type", "age_band", "region", "wall_type", "loft", "floor_type", "glazing", "draught_proofed",
    "main_heating", "boiler_age", "controls", "hot_water", "gas_cooking", "epc_band", "roof_type", "orientation",
    "flat_position", "tenure", "heritage", "outdoor_space", "shading",
]
NUM_COLS = ["floor_area", "bedrooms", "occupants", "storeys", "solar_kwp", "battery"]
PHYS_COLS = [
    "phys_hlc", "phys_hlp", "phys_envelope_ratio", "phys_hdd", "phys_u_wall", "phys_u_roof", "phys_u_floor",
    "phys_u_window", "phys_solar_yield", "phys_elec", "phys_gas", "phys_other", "phys_space_heat",
]
FEATURES = CAT_COLS + NUM_COLS + PHYS_COLS
FEATURES_NO_PHYS = CAT_COLS + NUM_COLS

CATEGORIES = {
    "property_type": ["detached", "semi", "terraced", "end_terrace", "flat", "bungalow"],
    "age_band": list("ABCDEFGHIJK"),
    "region": None,
    "wall_type": ["solid", "cavity_unfilled", "cavity_filled", "system_build", "solid_insulated", "not_sure"],
    "loft": ["none_or_flat", "under_100", "100_250", "over_250", "not_sure"],
    "floor_type": ["solid", "suspended", "insulated", "not_sure"],
    "glazing": ["single", "double_old", "double_modern", "triple", "mixed", "secondary", "not_sure"],
    "draught_proofed": ["yes", "no", "not_sure"],
    "main_heating": ["gas_boiler", "oil_boiler", "lpg_boiler", "electric", "heat_pump", "other"],
    "boiler_age": ["under_5", "5_10", "10_15", "over_15", "not_sure"],
    "controls": ["none", "programmer", "smart_thermostat", "smart_trvs", "not_sure"],
    "hot_water": ["boiler", "immersion", "heat_pump", "solar_thermal", "not_sure"],
    "gas_cooking": ["yes", "no", "not_sure"],
    "epc_band": list("ABCDEFG") + ["not_sure"],
    "roof_type": ["pitched", "flat", "not_sure"],
    "orientation": ["S", "SE_SW", "E_W", "N", "not_sure"],
    "flat_position": ["ground", "middle", "top", "not_sure"],
    "tenure": ["own", "rent"],
    "heritage": ["no", "conservation", "listed", "not_sure"],
    "outdoor_space": ["yes", "no", "not_sure"],
    "shading": ["none", "some", "heavy", "not_sure"],
}


def features_from_input(inp: CustomerInput) -> dict:
    state, _ = build_state(inp)
    sim = simulate(state, Calibration())
    geom = sim.geometry
    feats = {c: getattr(inp, c) for c in CAT_COLS if c != "region"}
    feats["region"] = state.region
    feats.update({
        "floor_area": inp.floor_area,
        "bedrooms": inp.bedrooms if inp.bedrooms is not None else np.nan,
        "occupants": inp.occupants if inp.occupants is not None else np.nan,
        "storeys": inp.storeys if inp.storeys is not None else np.nan,
        "solar_kwp": inp.solar_kwp,
        "battery": int(inp.battery),
        "phys_hlc": sim.hlc,
        "phys_hlp": sim.hlp,
        "phys_envelope_ratio": geom.envelope_area / state.floor_area,
        "phys_hdd": float(region_hdd(state.region).sum()),
        "phys_u_wall": sim.u_values["wall"],
        "phys_u_roof": sim.u_values["roof"],
        "phys_u_floor": sim.u_values["floor"],
        "phys_u_window": sim.u_values["window"],
        "phys_solar_yield": state.solar_kwp * regions()["regions"][state.region]["solar_kwh_per_kwp"],
        "phys_elec": sim.purchased["electricity"],
        "phys_gas": sim.purchased["gas"],
        "phys_other": sim.purchased["oil"] + sim.purchased["lpg"] + sim.purchased["solid"],
        "phys_space_heat": sim.annual("Q"),
    })
    return feats


def _from_form_json(form_json: str) -> dict:
    return features_from_input(CustomerInput(**json.loads(form_json)))


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in CAT_COLS:
        cats = CATEGORIES[c] if CATEGORIES[c] is not None else list(regions()["regions"].keys())
        df[c] = pd.Categorical(df[c].astype("object"), categories=cats)
    for c in NUM_COLS + PHYS_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df[FEATURES]


def build_feature_frame(form_jsons: list[str], workers: int | None = None) -> pd.DataFrame:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(form_jsons) < 500:
        rows = [_from_form_json(f) for f in form_jsons]
    else:
        with get_context("fork").Pool(workers) as pool:
            rows = pool.map(_from_form_json, form_jsons, chunksize=200)
    return to_frame(rows)
