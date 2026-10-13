"""Descriptive state of a home as the simulator sees it.

Retrofits change these descriptors (and numeric overrides), never simulator outputs.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

PROPERTY_TYPES = ("detached", "semi", "end_terrace", "terraced", "flat", "bungalow")
AGE_BANDS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K")
WALL_TYPES = ("solid", "cavity_unfilled", "cavity_filled", "system_build", "solid_insulated", "system_insulated")
LOFT_STATES = ("none", "under_100", "100_250", "over_250", "flat_roof")
FLOOR_TYPES = ("solid", "suspended", "insulated")
GLAZING_TYPES = ("single", "double_old", "double_modern", "triple", "mixed", "secondary")
HEATING_SYSTEMS = ("gas_boiler", "oil_boiler", "lpg_boiler", "electric", "heat_pump", "other")
BOILER_AGES = ("under_5", "5_10", "10_15", "over_15")
CONTROLS = ("none", "programmer", "smart_thermostat", "smart_trvs")
HOT_WATER = ("boiler", "immersion", "heat_pump", "solar_thermal")
ORIENTATIONS = ("S", "SE_SW", "E_W", "N", "unknown")
SHADING = ("none", "some", "heavy", "unknown")

HEATING_FUEL = {
    "gas_boiler": "gas",
    "oil_boiler": "oil",
    "lpg_boiler": "lpg",
    "electric": "electricity",
    "heat_pump": "electricity",
    "other": "solid",
}

FUELS = ("electricity", "gas", "oil", "lpg", "solid")


@dataclass
class HomeState:
    property_type: str
    age_band: str
    floor_area: float
    region: str
    storeys: int
    occupants: float
    bedrooms: int = 3
    flat_position: str = "unknown"  # ground, middle, top, unknown
    wall_type: str = "cavity_filled"
    loft: str = "100_250"
    roof_type: str = "pitched"
    floor_type: str = "solid"
    glazing: str = "double_old"
    draught_proofed: bool = False
    heating: str = "gas_boiler"
    boiler_age: str = "5_10"
    hp_fraction: float = 1.0  # phi; < 1 is a hybrid heat pump
    backup_heating: str | None = None  # boiler kept alongside a hybrid heat pump
    controls: str = "programmer"
    tariff_aware: bool = False
    hot_water: str = "boiler"
    gas_cooking: bool = False
    solar_kwp: float = 0.0
    orientation: str = "unknown"
    shading: str = "unknown"
    roof_free_area_m2: float | None = None
    preferred_solar_kwp: float | None = None  # what-if / customer choice for a new array
    battery_kwh: float = 0.0
    outdoor_space: bool | None = True
    heritage: str = "no"  # no, conservation, listed, unknown
    tenure: str = "own"
    # Numeric overrides set by retrofits or Monte Carlo draws, e.g. u_roof, scop, eta_rt.
    overrides: dict = field(default_factory=dict)
    retrofits_applied: list = field(default_factory=list)

    def copy(self) -> "HomeState":
        return copy.deepcopy(self)

    @property
    def heating_fuel(self) -> str:
        return HEATING_FUEL[self.heating]

    @property
    def has_gas_supply(self) -> bool:
        return self.heating == "gas_boiler" or self.gas_cooking or self.backup_heating == "gas_boiler"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HomeState":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
