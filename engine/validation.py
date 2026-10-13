"""Validation and missing-data tiering (spec 3.3).

Tier A: property facts complete and >= 9 metered months for each metered fuel.
Tier B: facts complete, usage partial, annual-only or bills.
Tier C: no usage data; baseline is a model estimate and the score is provisional.
Abstain: missing area/type/age/heating, unknown postcode, or conflicting entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.inputs import CustomerInput, Usage
from sim.parameters import postcode_to_region

QUESTIONS = {
    "postcode": "What is the property's postcode? We use it to look up your region's weather and solar yield.",
    "property_type": "What type of property is it (detached, semi, terraced, end-terrace, flat or bungalow)?",
    "age_band": "Roughly when was the property built? A range such as 1930-1949 is fine.",
    "floor_area": "What is the total floor area in m2? It is on your EPC, or estimate it from the number of rooms.",
    "main_heating": "What is the main heating system (gas boiler, oil, LPG, electric, heat pump or other)?",
}

MONTHS_FOR_TIER_A = 9


@dataclass
class UsageSummary:
    kind: str  # monthly, bills, annual, none
    months_present: int
    rough_annual_kwh: float | None


@dataclass
class Validation:
    status: str  # ok | abstain
    tier: str | None
    questions: list[dict] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, UsageSummary] = field(default_factory=dict)
    region: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "tier": self.tier,
            "questions": self.questions,
            "conflicts": self.conflicts,
            "warnings": self.warnings,
            "usage": {k: v.__dict__ for k, v in self.usage.items()},
            "region": self.region,
        }


def summarise_usage(usage: Usage | None, unit_rate: float, standing_per_year: float) -> UsageSummary:
    if usage is None or usage.is_empty():
        return UsageSummary("none", 0, None)
    if usage.monthly_kwh is not None and any(x is not None for x in usage.monthly_kwh):
        present = [x for x in usage.monthly_kwh if x is not None]
        return UsageSummary("monthly", len(present), sum(present) * 12 / len(present))
    if usage.monthly_bills_gbp is not None and any(x is not None for x in usage.monthly_bills_gbp):
        present = [x for x in usage.monthly_bills_gbp if x is not None]
        kwh = [max(b - standing_per_year / 12, 0) / unit_rate for b in present]
        return UsageSummary("bills", len(present), sum(kwh) * 12 / len(kwh))
    return UsageSummary("annual", 12, float(usage.annual_kwh))


def validate(inp: CustomerInput) -> Validation:
    from sim.parameters import prices

    cp = prices()["customer"]
    v = Validation(status="ok", tier=None)

    region = postcode_to_region(inp.postcode)
    v.region = region
    if region is None:
        v.questions.append({"field": "postcode", "question": QUESTIONS["postcode"] + " We could not recognise the one given."})
    for fld in ("property_type", "age_band", "floor_area", "main_heating"):
        if getattr(inp, fld) is None:
            v.questions.append({"field": fld, "question": QUESTIONS[fld]})
    if inp.floor_area is not None and not (15 <= inp.floor_area <= 1500):
        v.questions.append({"field": "floor_area", "question": f"A floor area of {inp.floor_area:g} m2 looks unlikely. Please check it (total of all floors, in m2)."})

    elec = summarise_usage(inp.electricity, cp["unit_rate_gbp_per_kwh"]["electricity"], cp["standing_charge_gbp_per_year"]["electricity"])
    gas = summarise_usage(inp.gas, cp["unit_rate_gbp_per_kwh"]["gas"], cp["standing_charge_gbp_per_year"]["gas"])
    v.usage = {"electricity": elec, "gas": gas}

    # Conflicting entries -> abstain and ask
    gas_annual = gas.rough_annual_kwh or 0.0
    heating = inp.main_heating
    if heating in ("heat_pump", "electric", "oil_boiler", "lpg_boiler") and gas_annual > 4000:
        v.conflicts.append(
            f"Main heating is given as {heating.replace('_', ' ')}, but gas use is about {gas_annual:,.0f} kWh a year, "
            "which is more than cooking alone would use. Is gas still used for heating or hot water?"
        )
    if heating == "gas_boiler" and gas.kind != "none" and inp.floor_area and gas_annual < 1000:
        v.conflicts.append(
            f"Main heating is a gas boiler but gas use is only about {gas_annual:,.0f} kWh a year. "
            "Please check the gas readings or the heating type."
        )
    if inp.bedrooms is not None and inp.floor_area is not None and inp.bedrooms > max(1, inp.floor_area / 10):
        v.conflicts.append(f"{inp.bedrooms} bedrooms in {inp.floor_area:g} m2 looks inconsistent. Please check both.")
    if inp.solar_kwp <= 0 and (inp.solar_generation_kwh or 0) > 0:
        v.conflicts.append("Solar generation readings were given but no solar panels. How big is the system (kWp)?")
    if inp.solar_generation_kwh is not None and inp.solar_export_kwh is not None and inp.solar_export_kwh > inp.solar_generation_kwh:
        v.conflicts.append("Solar export is higher than solar generation. Please check the two readings.")
    if inp.battery and inp.solar_kwp <= 0:
        v.warnings.append("A battery without solar is modelled with grid charging off, so it has no effect on the estimate.")
    if elec.rough_annual_kwh is not None and elec.rough_annual_kwh < 400:
        v.warnings.append("Electricity use below 400 kWh a year is unusual; check the readings (or an empty home).")

    if v.questions or v.conflicts:
        v.status = "abstain"
        return v

    # Tiering
    metered_fuels = {"electricity": elec}
    on_gas = heating == "gas_boiler" or gas.kind != "none"
    if on_gas:
        metered_fuels["gas"] = gas
    kinds = [u.kind for u in metered_fuels.values()]
    if all(k == "none" for k in kinds):
        v.tier = "C"
    elif all(u.kind == "monthly" and u.months_present >= MONTHS_FOR_TIER_A for u in metered_fuels.values()):
        v.tier = "A"
    else:
        v.tier = "B"
        missing = [f for f, u in metered_fuels.items() if u.kind == "none"]
        if missing:
            v.warnings.append(f"No usage given for {', '.join(missing)}; that fuel is estimated by the model.")
    if heating in ("oil_boiler", "lpg_boiler", "other"):
        v.warnings.append("Heating fuel other than gas or electricity is not metered here; it is estimated by the model.")
    return v
