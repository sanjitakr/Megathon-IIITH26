"""Bills and carbon (spec 9). Savings are on purchased energy, per fuel; negative = rise."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.home import FUELS
from sim.parameters import emission_factors, prices


@dataclass
class PriceSet:
    unit: dict[str, float]
    export: float
    include_export: bool
    standing: dict[str, float] = field(default_factory=dict)
    label: str = "default"

    @classmethod
    def from_options(cls, opts=None) -> "PriceSet":
        cp = prices()["customer"]
        unit = dict(cp["unit_rate_gbp_per_kwh"])
        export = cp["export_gbp_per_kwh"]
        label = cp["period"]
        if opts is not None:
            for fuel in ("electricity", "gas", "oil", "lpg"):
                val = getattr(opts, f"{fuel}_price", None)
                if val is not None:
                    unit[fuel] = float(val)
                    label = "customer-adjusted"
            if getattr(opts, "export_price", None) is not None:
                export = float(opts.export_price)
                label = "customer-adjusted"
        return cls(unit=unit, export=export, include_export=cp["include_export_income"], standing=dict(cp["standing_charge_gbp_per_year"]), label=label)

    def to_dict(self) -> dict:
        return {"unit_rate_gbp_per_kwh": self.unit, "export_gbp_per_kwh": self.export, "include_export_income": self.include_export, "label": self.label}


@dataclass
class EmissionSet:
    factors: dict[str, float]
    label: str

    @classmethod
    def from_options(cls, opts=None) -> "EmissionSet":
        ef = emission_factors()
        factors = dict(ef["factors"])
        label = f"DESNZ/DEFRA {ef['default_year']} (verify)"
        if opts is not None:
            if getattr(opts, "grid_carbon_factor", None) is not None:
                factors["electricity"] = float(opts.grid_carbon_factor)
                label = "customer-adjusted grid factor"
            elif getattr(opts, "grid_carbon_year", None) is not None:
                traj = {int(k): v for k, v in ef["grid_trajectory"].items() if not k.startswith("_")}
                years = sorted(traj)
                factors["electricity"] = float(np.interp(opts.grid_carbon_year, years, [traj[y] for y in years]))
                label = f"grid trajectory {opts.grid_carbon_year} (illustrative sensitivity)"
        return cls(factors=factors, label=label)

    def to_dict(self) -> dict:
        return {"kg_co2e_per_kwh": self.factors, "label": self.label, "export_credited": False}


def annual_bill(purchased: dict[str, float], export_kwh: float, ps: PriceSet) -> float:
    """Unit-rate cost minus export income; standing charges excluded (they rarely change)."""
    cost = sum(purchased.get(f, 0.0) * ps.unit[f] for f in FUELS)
    if ps.include_export:
        cost -= export_kwh * ps.export
    return cost


def annual_carbon(purchased: dict[str, float], es: EmissionSet) -> float:
    """kg CO2e from purchased energy. Export is not credited."""
    return sum(purchased.get(f, 0.0) * es.factors[f] for f in FUELS)


def savings(base: dict, after: dict, base_export: float, after_export: float, ps: PriceSet, es: EmissionSet) -> dict:
    delta = {f: base.get(f, 0.0) - after.get(f, 0.0) for f in FUELS}
    bill = annual_bill(base, base_export, ps) - annual_bill(after, after_export, ps)
    carbon = annual_carbon(base, es) - annual_carbon(after, es)
    return {"delta_kwh": delta, "bill_saving_gbp": bill, "carbon_saving_kg": carbon, "export_change_kwh": after_export - base_export}


def payback_years(cost: float, bill_saving: float) -> float | None:
    if bill_saving <= 0:
        return None
    return cost / bill_saving
