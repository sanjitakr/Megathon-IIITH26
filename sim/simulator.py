"""Monthly physics simulator for a home (spec 6.3).

simulate() turns a HomeState into monthly kWh per fuel, the solar/battery balance and
the heat-loss figures. Calibration (global kappa and per-home kappa_home/alpha) only
scales demand; retrofits only change the HomeState.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim.energy_balance import solar_battery_flows
from sim.geometry import Geometry, compute_geometry
from sim.home import FUELS, HomeState
from sim.parameters import days_in_month, electricity_shape, params, region_hdd, region_solar_monthly

DHW_SEASONAL = np.array([1.10, 1.06, 1.02, 0.98, 0.94, 0.90, 0.88, 0.90, 0.94, 0.98, 1.04, 1.10])


@dataclass
class Calibration:
    """Per-home scales. kappa_home scales space heating, alpha scales appliance load.

    behaviour/appliance_hidden/dhw_hidden are hidden synthetic factors (never model inputs).
    """

    kappa_home: float = 1.0
    alpha: float = 1.0
    behaviour: float = 1.0
    appliance_hidden: float = 1.0
    dhw_hidden: float = 1.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SimResult:
    monthly: dict[str, np.ndarray]
    hlc: float
    hlp: float
    u_values: dict[str, float]
    hlc_parts: dict[str, float]
    scop: float | None
    geometry: Geometry
    notes: list[str] = field(default_factory=list)

    def annual(self, key: str) -> float:
        return float(self.monthly[key].sum())

    @property
    def purchased(self) -> dict[str, float]:
        """Grid import plus fuels bought, kWh/yr."""
        out = {"electricity": self.annual("I")}
        for fuel in FUELS[1:]:
            out[fuel] = self.annual(fuel)
        return out

    @property
    def gross(self) -> dict[str, float]:
        """What the home uses: gross electricity L plus fuels, kWh/yr."""
        out = {"electricity": self.annual("L")}
        for fuel in FUELS[1:]:
            out[fuel] = self.annual(fuel)
        return out

    @property
    def heating_part(self) -> dict[str, float]:
        return {f: self.annual(f"heat_{f}") for f in FUELS}

    def summary(self) -> dict:
        return {
            "purchased_kwh": self.purchased,
            "gross_kwh": self.gross,
            "generation_kwh": self.annual("g"),
            "export_kwh": self.annual("X"),
            "battery_losses_kwh": self.annual("Z"),
            "space_heat_demand_kwh": self.annual("Q"),
            "hot_water_demand_kwh": self.annual("dhw"),
            "hlc_w_per_k": self.hlc,
            "hlp_w_per_m2k": self.hlp,
            "scop": self.scop,
        }


def standard_occupancy(floor_area: float) -> float:
    """SAP-style assumed occupancy from floor area (used for the rating and as a default)."""
    tfa = max(floor_area, 13.9)
    return 1 + 1.76 * (1 - math.exp(-0.000349 * (tfa - 13.9) ** 2)) + 0.0013 * (tfa - 13.9)


def u_values(state: HomeState) -> dict[str, float]:
    p = params()
    ov = state.overrides
    age = state.age_band

    if "u_wall" in ov:
        u_wall = ov["u_wall"]
    else:
        u_wall = p["u_wall"][state.wall_type]
        cap = p["u_wall_age_cap"].get(age)
        if cap is not None:
            u_wall = min(u_wall, cap)

    if "u_roof" in ov:
        u_roof = ov["u_roof"]
    elif state.loft == "flat_roof" or state.roof_type == "flat":
        u_roof = p["u_roof_flat_by_age"][age]
    else:
        u_roof = p["u_roof"][state.loft]

    if "u_floor" in ov:
        u_floor = ov["u_floor"]
    else:
        u_floor = p["u_floor"][state.floor_type]
        cap = p["u_floor_age_cap"].get(age)
        if cap is not None:
            u_floor = min(u_floor, cap)

    u_window = ov.get("u_window", p["u_window"][state.glazing])

    quality = ov.get("build_quality", 1.0)  # hidden synthetic factor
    return {
        "wall": u_wall * quality,
        "roof": u_roof * quality,
        "floor": u_floor * quality,
        "window": u_window,
    }


def air_change_rate(state: HomeState) -> float:
    p = params()
    n = p["air_change_by_age"][state.age_band] * p["air_change_scale_global"]
    if state.draught_proofed:
        n *= state.overrides.get("air_change_factor", p["draught_proofing_factor"])
    return n


def heat_loss(state: HomeState, geom: Geometry | None = None) -> tuple[float, dict[str, float], dict[str, float]]:
    """HLC (W/K) = sum(U x A) + 0.33 n V."""
    geom = geom or compute_geometry(state)
    u = u_values(state)
    parts = {
        "wall": u["wall"] * geom.wall_area_net,
        "roof": u["roof"] * geom.roof_area * geom.roof_exposure,
        "floor": u["floor"] * geom.floor_ground_area * geom.floor_exposure,
        "window": u["window"] * geom.window_area,
        "ventilation": 0.33 * air_change_rate(state) * geom.volume,
    }
    return float(sum(parts.values())), parts, u


def boiler_efficiency(system: str, boiler_age: str, state: HomeState) -> float:
    p = params()
    if "boiler_efficiency" in state.overrides:
        return state.overrides["boiler_efficiency"]
    if system in p["boiler_efficiency"]:
        eff = p["boiler_efficiency"][system][boiler_age] + p["boiler_efficiency_offset_global"]
    else:
        eff = p["other_heating_efficiency"][system]
    return float(np.clip(eff, 0.4, 1.0))


def heat_pump_scop(state: HomeState, hlp: float) -> float:
    hp = params()["heat_pump"]
    scop = state.overrides.get("scop", hp["scop_base"])
    scop -= hp["scop_penalty_per_hlp"] * max(0.0, hlp - hp["readiness_hlp"])
    return float(np.clip(scop, hp["scop_min"], hp["scop_max"]))


def simulate(state: HomeState, calib: Calibration | None = None, rating_mode: bool = False) -> SimResult:
    """Run the monthly model.

    rating_mode uses standard occupancy, no per-home calibration, no cooking and lighting
    instead of appliances, mirroring how a SAP-style rating is computed.
    """
    p = params()
    calib = calib or Calibration()
    geom = compute_geometry(state)
    hlc, hlc_parts, u = heat_loss(state, geom)
    hlp = hlc / state.floor_area
    days = days_in_month()
    hdd = region_hdd(state.region)
    zeros = np.zeros(12)
    notes: list[str] = []

    occupants = standard_occupancy(state.floor_area) if rating_mode else state.occupants
    controls_factor = state.overrides.get("controls_factor", p["controls_factor"][state.controls])
    if rating_mode:
        kappa = 1.0
    else:
        kappa = p["kappa_global"] * calib.kappa_home * calib.behaviour
    Q = kappa * controls_factor * hlc * 24.0 * hdd / 1000.0  # useful space heat, kWh

    dhw_annual = p["hot_water"]["a0"] + p["hot_water"]["a1"] * occupants
    if not rating_mode:
        dhw_annual *= calib.dhw_hidden
    dhw_shape = DHW_SEASONAL * days
    dhw = dhw_annual * dhw_shape / dhw_shape.sum()

    fuel = {f: zeros.copy() for f in ("gas", "oil", "lpg", "solid")}
    heat = {f: zeros.copy() for f in ("electricity", "gas", "oil", "lpg", "solid")}
    elec_space = zeros.copy()
    elec_dhw = zeros.copy()
    scop = None

    system = state.heating
    boiler_fuel = {"gas_boiler": "gas", "oil_boiler": "oil", "lpg_boiler": "lpg", "other": "solid"}

    # Space heating
    if system == "heat_pump":
        scop = heat_pump_scop(state, hlp)
        phi = float(np.clip(state.hp_fraction, 0.0, 1.0))
        Q_hp = phi * Q
        elec_space += Q_hp / scop
        heat["electricity"] += Q_hp / scop
        if phi < 1.0 and state.backup_heating in boiler_fuel:
            eff = boiler_efficiency(state.backup_heating, state.boiler_age, state)
            f = boiler_fuel[state.backup_heating]
            fuel[f] += (1 - phi) * Q / eff
            heat[f] += (1 - phi) * Q / eff
        elif phi < 1.0:
            elec_space += (1 - phi) * Q
            heat["electricity"] += (1 - phi) * Q
    elif system == "electric":
        elec_space += Q
        heat["electricity"] += Q
    else:
        eff = boiler_efficiency(system, state.boiler_age, state)
        f = boiler_fuel[system]
        fuel[f] += Q / eff
        heat[f] += Q / eff

    # Hot water
    dhw_load = dhw.copy()
    if state.hot_water == "solar_thermal":
        dhw_load = dhw * (1 - p["solar_thermal_fraction"])
    if state.hot_water == "heat_pump":
        elec_dhw += dhw_load / p["heat_pump"]["scop_dhw"]
    elif state.hot_water == "immersion":
        elec_dhw += dhw_load
    else:  # boiler or solar thermal topped up by the main system
        dhw_system = system
        if system == "heat_pump":
            dhw_system = state.backup_heating or "heat_pump"
        if dhw_system == "heat_pump":
            elec_dhw += dhw_load / p["heat_pump"]["scop_dhw"]
        elif dhw_system == "electric":
            elec_dhw += dhw_load
        else:
            eff = boiler_efficiency(dhw_system, state.boiler_age, state) * p["dhw_from_boiler_efficiency_factor"]
            fuel[boiler_fuel[dhw_system]] += dhw_load / eff

    # Cooking gas (excluded from the rating, as in SAP)
    if state.gas_cooking and not rating_mode:
        fuel["gas"] += p["cooking_gas_kwh"] * days / days.sum()

    # Appliances and lighting
    if rating_mode:
        appliances = p["lighting_kwh_per_m2_rating"] * state.floor_area * days / days.sum()
    else:
        a = p["appliances"]
        annual = (a["a0"] + a["a1"] * occupants + a["a2"] * state.floor_area) * p["appliance_scale_global"]
        annual *= calib.alpha * calib.appliance_hidden
        appliances = annual * electricity_shape()

    L = appliances + elec_space + elec_dhw

    # Solar and battery (heating change first, then solar, then battery)
    g = zeros.copy()
    if state.solar_kwp > 0:
        sp = p["solar"]
        g = (
            state.solar_kwp
            * region_solar_monthly(state.region)
            * sp["orientation_factor"].get(state.orientation, sp["orientation_factor"]["unknown"])
            * sp["shading_factor"].get(state.shading, sp["shading_factor"]["unknown"])
            * state.overrides.get("yield_factor", 1.0)
        )
    bp = p["battery"]
    eta_rt = state.overrides.get("eta_rt", bp["eta_charge"] * bp["eta_discharge"])
    flows = solar_battery_flows(
        g,
        L,
        days,
        p["self_consumption_curve"],
        battery_kwh=state.battery_kwh if state.solar_kwp > 0 else 0.0,
        eta_rt=eta_rt,
        cycles_per_day=bp["cycles_per_day"],
        f_sc_factor=state.overrides.get("f_sc_factor", 1.0),
    )
    if state.battery_kwh > 0 and state.solar_kwp <= 0:
        notes.append("battery without solar: grid charging is off by default, so it has no effect")

    monthly = {
        "hdd": hdd,
        "Q": Q,
        "dhw": dhw,
        "appliances": appliances,
        "elec_space": elec_space,
        "elec_dhw": elec_dhw,
        **{k: flows[k] for k in ("g", "L", "D", "C_in", "B_out", "Z", "X", "I")},
        **fuel,
        **{f"heat_{k}": v for k, v in heat.items()},
    }
    return SimResult(
        monthly=monthly,
        hlc=hlc,
        hlp=hlp,
        u_values=u,
        hlc_parts=hlc_parts,
        scop=scop,
        geometry=geom,
        notes=notes,
    )
