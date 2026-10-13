"""Customer form schema (spec 3.2). Every optional question accepts 'not_sure'."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

PropertyType = Literal["detached", "semi", "terraced", "end_terrace", "flat", "bungalow"]
AgeBand = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
Goal = Literal["cut_bills", "cut_carbon", "reach_c", "comfort", "value_for_money"]
Disruption = Literal["low", "medium", "high"]

AGE_BAND_LABELS = {
    "A": "pre-1900", "B": "1900-1929", "C": "1930-1949", "D": "1950-1966", "E": "1967-1975",
    "F": "1976-1982", "G": "1983-1990", "H": "1991-1995", "I": "1996-2002", "J": "2003-2006", "K": "2007+",
}


class Usage(BaseModel):
    """Last 12 months of one fuel. Give any one of the three forms.

    monthly lists have 12 entries, index 0 = January; use null for a missing month.
    Missing months are never treated as zero.
    """

    monthly_kwh: Optional[list[Optional[float]]] = None
    annual_kwh: Optional[float] = None
    monthly_bills_gbp: Optional[list[Optional[float]]] = None

    @field_validator("monthly_kwh", "monthly_bills_gbp")
    @classmethod
    def twelve_months(cls, v):
        if v is not None and len(v) != 12:
            raise ValueError("monthly values need 12 entries (use null for missing months)")
        if v is not None and any(x is not None and x < 0 for x in v):
            raise ValueError("monthly values cannot be negative")
        return v

    def is_empty(self) -> bool:
        has_monthly = self.monthly_kwh is not None and any(x is not None for x in self.monthly_kwh)
        has_bills = self.monthly_bills_gbp is not None and any(x is not None for x in self.monthly_bills_gbp)
        return not (has_monthly or has_bills or self.annual_kwh is not None)


class CustomerInput(BaseModel):
    customer_id: str = "anonymous"
    property_id: Optional[str] = None

    # Step 1: location
    postcode: str
    address: Optional[str] = None

    # Step 2: basics
    property_type: Optional[PropertyType] = None
    age_band: Optional[AgeBand] = None
    floor_area: Optional[float] = Field(default=None, description="m2")
    bedrooms: Optional[int] = None
    occupants: Optional[float] = None
    storeys: Optional[int] = None
    flat_position: Literal["ground", "middle", "top", "not_sure"] = "not_sure"

    # Step 3: fabric
    wall_type: Literal["solid", "cavity_unfilled", "cavity_filled", "system_build", "solid_insulated", "not_sure"] = "not_sure"
    loft: Literal["none_or_flat", "under_100", "100_250", "over_250", "not_sure"] = "not_sure"
    floor_type: Literal["solid", "suspended", "insulated", "not_sure"] = "not_sure"
    glazing: Literal["single", "double_old", "double_modern", "triple", "mixed", "secondary", "not_sure"] = "not_sure"
    draught_proofed: Literal["yes", "no", "not_sure"] = "not_sure"

    # Step 4: heating
    main_heating: Optional[Literal["gas_boiler", "oil_boiler", "lpg_boiler", "electric", "heat_pump", "other"]] = None
    boiler_age: Literal["under_5", "5_10", "10_15", "over_15", "not_sure"] = "not_sure"
    controls: Literal["none", "programmer", "smart_thermostat", "smart_trvs", "not_sure"] = "not_sure"
    tariff_aware: bool = False
    hot_water: Literal["boiler", "immersion", "heat_pump", "solar_thermal", "not_sure"] = "not_sure"
    gas_cooking: Literal["yes", "no", "not_sure"] = "not_sure"

    # Step 5: usage
    electricity: Optional[Usage] = None
    gas: Optional[Usage] = None
    hdd_metered_period: Optional[float] = Field(default=None, description="Heating degree-days in the metered period, if known")

    # Step 6: generation
    solar_kwp: float = 0.0
    battery: bool = False
    solar_generation_kwh: Optional[float] = None
    solar_export_kwh: Optional[float] = None
    roof_type: Literal["pitched", "flat", "not_sure"] = "not_sure"
    orientation: Literal["S", "SE_SW", "E_W", "N", "not_sure"] = "not_sure"
    roof_free_area_m2: Optional[float] = None
    solar_size_kwp: Optional[float] = Field(default=None, description="Preferred size for new solar panels (what-if)")
    shading: Literal["none", "some", "heavy", "not_sure"] = "not_sure"
    outdoor_space: Literal["yes", "no", "not_sure"] = "not_sure"

    # Step 7: context
    heritage: Literal["no", "conservation", "listed", "not_sure"] = "not_sure"
    tenure: Literal["own", "rent"] = "own"
    epc_band: Literal["A", "B", "C", "D", "E", "F", "G", "not_sure"] = "not_sure"
    sap_points: Optional[float] = None
    current_value_estimate: Optional[float] = None

    # Step 8: goals and memory
    budget: Optional[float] = None
    goal: Goal = "value_for_money"
    disruption_tolerance: Disruption = "high"
    past_retrofits: list[str] = Field(default_factory=list)
    declined: list[str] = Field(default_factory=list)

    def property_key(self) -> str:
        """Stable key of the property facts (excludes goals), used to cache stage 1."""
        facts = self.model_dump(exclude={"budget", "goal", "disruption_tolerance", "declined", "customer_id", "address"})
        return str(sorted(facts.items(), key=lambda kv: kv[0]))


class CustomiseOptions(BaseModel):
    """Stage-2 customisation (spec 4.2). Re-ranks cached results only."""

    goal: Goal = "value_for_money"
    budget: Optional[float] = None
    disruption_tolerance: Disruption = "high"
    tick: list[str] = Field(default_factory=list)
    untick: list[str] = Field(default_factory=list)
    declined: list[str] = Field(default_factory=list)
    preset: Optional[Literal["quick_wins", "fabric_first", "heat_pump_ready", "solar_and_smart", "full_green"]] = None
    electricity_price: Optional[float] = None
    gas_price: Optional[float] = None
    oil_price: Optional[float] = None
    lpg_price: Optional[float] = None
    export_price: Optional[float] = None
    grid_carbon_year: Optional[int] = None
    grid_carbon_factor: Optional[float] = None
    apply_grants: bool = False
