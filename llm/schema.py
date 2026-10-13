"""Scenario schema the LLM must return (spec 11), validated with Pydantic."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

RetrofitId = Literal[
    "loft_insulation", "cavity_wall_insulation", "solid_wall_insulation", "floor_insulation", "double_glazing",
    "draught_proofing", "heat_pump", "solar_pv", "battery", "smart_controls",
]
QuestionType = Literal["what_if", "explain_result", "compare", "infeasible_request", "missing_info", "general"]


class Scenario(BaseModel):
    question_type: QuestionType = "general"
    retrofits_add: list[RetrofitId] = Field(default_factory=list)
    retrofits_remove: list[RetrofitId] = Field(default_factory=list)
    solar_kwp: Optional[float] = Field(default=None, ge=0.5, le=15)
    budget: Optional[float] = Field(default=None, ge=0, description="Indian rupees")
    goal: Optional[Literal["cut_bills", "cut_carbon", "reach_c", "comfort", "value_for_money"]] = None
    electricity_price: Optional[float] = Field(default=None, description="rupees per kWh")
    gas_price: Optional[float] = Field(default=None, description="rupees per kWh")
    grid_carbon_year: Optional[int] = Field(default=None, ge=2020, le=2050)

    @field_validator("electricity_price", "gas_price")
    @classmethod
    def price_range(cls, v):
        if v is not None and not (0.1 <= v <= 500):
            raise ValueError("price out of range (rupees per kWh)")
        return v

    def is_empty(self) -> bool:
        return not any([self.retrofits_add, self.retrofits_remove, self.solar_kwp, self.budget is not None, self.goal,
                        self.electricity_price, self.gas_price, self.grid_carbon_year])


SCENARIO_JSON_SPEC = """Return ONLY a JSON object with these optional fields:
{
  "question_type": one of "what_if", "explain_result", "compare", "infeasible_request", "missing_info", "general",
  "retrofits_add": list of retrofit ids to add,
  "retrofits_remove": list of retrofit ids to remove,
  "solar_kwp": number (kWp of new solar panels),
  "budget": number (upfront ceiling in Indian rupees, full digits: 3 lakh = 300000),
  "goal": one of "cut_bills", "cut_carbon", "reach_c", "comfort", "value_for_money",
  "electricity_price": number (rupees per kWh),
  "gas_price": number (rupees per kWh),
  "grid_carbon_year": integer year for the grid carbon factor
}
Retrofit ids: loft_insulation, cavity_wall_insulation, solid_wall_insulation, floor_insulation, double_glazing,
draught_proofing, heat_pump, solar_pv, battery, smart_controls.
Leave out fields the user did not mention. Do not invent values."""
