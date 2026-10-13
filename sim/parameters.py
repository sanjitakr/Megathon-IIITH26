"""Loads every reference table (parameters, regions, prices, emission factors, catalogue).

All numbers live in data/reference/*.json so they can be replaced with sourced values
without touching code. Global calibration (gen/calibration.py) is written to
data/processed/calibration.json and overlaid on the parameter table here.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "reference"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "data" / "models"

CALIBRATED_KEYS = (
    "kappa_global",
    "appliance_scale_global",
    "air_change_scale_global",
    "boiler_efficiency_offset_global",
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _load(name: str, directory: Path = REFERENCE_DIR) -> dict:
    with open(directory / name, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def params() -> dict:
    p = _load("parameters.json")
    calib_path = PROCESSED_DIR / "calibration.json"
    if calib_path.exists():
        calib = _load("calibration.json", PROCESSED_DIR)
        for key in CALIBRATED_KEYS:
            if key in calib.get("theta", {}):
                p[key] = calib["theta"][key]
        p["_calibration"] = {k: calib.get(k) for k in ("version", "created", "objective_before", "objective_after")}
    return p


@lru_cache(maxsize=None)
def regions() -> dict:
    return _load("regions.json")


@lru_cache(maxsize=None)
def catalogue() -> dict:
    return _load("retrofit_catalogue.json")


@lru_cache(maxsize=None)
def emission_factors() -> dict:
    return _load("emission_factors.json")


@lru_cache(maxsize=None)
def prices() -> dict:
    return _load("prices.json")


@lru_cache(maxsize=None)
def need_targets() -> dict:
    return _load("need_targets.json")


@lru_cache(maxsize=None)
def sampling_priors() -> dict:
    return _load("sampling_priors.json")


@lru_cache(maxsize=None)
def score_config() -> dict:
    return _load("score_config.json")


def reload() -> None:
    """Clear caches, e.g. after calibration has been rewritten."""
    for fn in (params, regions, catalogue, emission_factors, prices, need_targets, sampling_priors, score_config):
        fn.cache_clear()
    _postcode_lookup.cache_clear()


@lru_cache(maxsize=None)
def _postcode_lookup() -> dict[str, str]:
    lookup = {}
    for region, areas in regions()["postcode_area_to_region"].items():
        for area in areas:
            lookup[area] = region
    return lookup


def postcode_area(postcode: str) -> str | None:
    match = re.match(r"\s*([A-Za-z]{1,2})", postcode or "")
    return match.group(1).upper() if match else None


def postcode_district(postcode: str) -> str | None:
    match = re.match(r"\s*([A-Za-z]{1,2}\d[A-Za-z\d]?)", postcode or "")
    return match.group(1).upper() if match else None


def postcode_to_region(postcode: str) -> str | None:
    area = postcode_area(postcode)
    if area is None:
        return None
    lookup = _postcode_lookup()
    # Two-letter areas take precedence (e.g. "SE" over "S").
    if area in lookup:
        return lookup[area]
    if len(area) == 2 and area[0] in lookup:
        return None  # e.g. "SX": unknown two-letter area, do not guess
    return None


def region_hdd(region: str) -> np.ndarray:
    r = regions()
    return np.asarray(r["hdd_national_monthly"], dtype=float) * r["regions"][region]["hdd_factor"]


def region_solar_monthly(region: str) -> np.ndarray:
    """kWh per kWp per month for a south-facing, unshaded array."""
    r = regions()
    frac = np.asarray(r["solar_monthly_fraction"], dtype=float)
    frac = frac / frac.sum()
    return frac * r["regions"][region]["solar_kwh_per_kwp"]


def days_in_month() -> np.ndarray:
    return np.asarray(params()["days_in_month"], dtype=float)


def electricity_shape() -> np.ndarray:
    """Monthly share of annual appliance electricity (sums to 1)."""
    shape = np.asarray(params()["electricity_monthly_shape"], dtype=float) * days_in_month()
    return shape / shape.sum()


def gas_shape(region: str) -> np.ndarray:
    """Typical monthly share of annual gas: heating follows HDD, a base share is flat."""
    hdd = region_hdd(region)
    shape = 0.8 * hdd / hdd.sum() + 0.2 * days_in_month() / days_in_month().sum()
    return shape / shape.sum()


def all_regions() -> list[str]:
    return list(regions()["regions"].keys())


def retrofit_ids() -> list[str]:
    return list(catalogue()["retrofits"].keys())
