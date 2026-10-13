"""User message -> validated Scenario. LLM first (JSON mode), rule-based parser as fallback."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from llm.providers import LLMChain, ProviderError
from llm.schema import SCENARIO_JSON_SPEC, Scenario

KEYWORDS = [
    ("heat_pump", r"heat ?pump|ashp|air source"),
    ("solar_pv", r"solar|\bpv\b|panels"),
    ("battery", r"batter(y|ies)"),
    ("loft_insulation", r"loft|attic|roof insulation"),
    ("cavity_wall_insulation", r"cavity"),
    ("solid_wall_insulation", r"solid wall|external wall insulation|internal wall insulation|\bewi\b|\biwi\b"),
    ("floor_insulation", r"floor"),
    ("double_glazing", r"glazing|windows?"),
    ("draught_proofing", r"draught|draft"),
    ("smart_controls", r"smart|thermostat|trv|zoning|controls"),
]
REMOVE_WORDS = r"without|remove|drop|skip|no |not |don'?t|exclude|instead of"
GOALS = [
    ("reach_c", r"band c|epc c|rating c|\bc rating"),
    ("cut_carbon", r"carbon|co2|emission|greener|net zero"),
    ("cut_bills", r"bill|cheaper|save money|cost of living"),
    ("comfort", r"comfort|warmer|cold|draughty"),
    ("value_for_money", r"value for money|best value|payback"),
]

SYSTEM = "You convert a homeowner's question about home energy retrofits into a JSON scenario. " + SCENARIO_JSON_SPEC


CUR = r"(?:₹|rs\.?|inr|rupees?|£)"
MULT = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7}


def _amount(num: str, mult: str | None, cur: str | None) -> float:
    val = float(num.replace(",", "")) * MULT.get((mult or "").strip().lower(), 1.0)
    if cur and "£" in cur:  # pounds given: convert to rupees
        from engine.currency import rate

        val *= rate()
    return val


def _budget(text: str) -> float | None:
    """Budget in rupees: '₹3,00,000', 'rs 2 lakh', '£3k' (converted), 'budget of 50000'."""
    num = r"(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand|lakhs?|lac|crores?|cr\b)?"
    m = re.search(r"(?:budget|spend|afford|for|under|up to|with|within|only have)[^0-9₹£]{0,12}(" + CUR + r")\s*" + num, text)
    if m:
        return _amount(m.group(2), m.group(3), m.group(1))
    m = re.search(num + r"\s*(rupees?|rs\.?|inr)?\s*budget", text) or re.search(r"budget[^0-9₹£]{0,15}(" + CUR + r")?\s*" + num, text)
    if m:
        g = m.groups()
        if len(g) == 3 and g[0] and not g[0][0].isdigit():  # budget ... (cur) num mult
            return _amount(g[1], g[2], g[0])
        if len(g) == 3 and g[0] is None:
            return _amount(g[1], g[2], None)
        return _amount(g[0], g[1], g[2])
    return None


def _price(text: str, fuel: str) -> float | None:
    """Unit price in rupees per kWh: 'gas to ₹9', 'electricity 30 rupees', '9p' (pence, converted)."""
    m = re.search(fuel + r"[^0-9₹£]{0,25}(?:₹|rs\.?\s?|inr\s?)\s*(\d+(?:\.\d+)?)", text) or \
        re.search(fuel + r"[^0-9₹£]{0,25}(\d+(?:\.\d+)?)\s*(?:rupees?|rs\b|₹)", text)
    if m:
        return float(m.group(1))
    m = re.search(fuel + r"[^0-9]{0,20}(\d+(?:\.\d+)?)\s*(?:p\b|pence)", text)
    if m:
        from engine.currency import rate

        return round(float(m.group(1)) / 100 * rate(), 2)
    return None


def rule_parse(message: str) -> Scenario:
    text = message.lower()
    add, remove = [], []
    for rid, pat in KEYWORDS:
        for m in re.finditer(pat, text):
            window = text[max(0, m.start() - 25): m.start()]
            if re.search(REMOVE_WORDS, window):
                remove.append(rid)
            else:
                add.append(rid)
    if "wall" in text and not any(r in add + remove for r in ("cavity_wall_insulation", "solid_wall_insulation")):
        add.append("cavity_wall_insulation" if "cavity" in text else "solid_wall_insulation")
    add = [r for r in dict.fromkeys(add) if r not in remove]
    remove = list(dict.fromkeys(remove))

    data: dict = {"retrofits_add": add, "retrofits_remove": remove}
    m = re.search(r"(\d+(?:\.\d+)?)\s*kwp", text)
    if m:
        data["solar_kwp"] = float(m.group(1))
        if "solar_pv" not in add:
            data["retrofits_add"].append("solar_pv")
    budget = _budget(text)
    if budget is not None:
        data["budget"] = budget
    for fuel, pat in (("gas_price", r"gas"), ("electricity_price", r"electric(?:ity)?")):
        price = _price(text, pat)
        if price is not None:
            data[fuel] = price
    m = re.search(r"\b(20[2-4]\d|2050)\b", text)
    if m and ("grid" in text or "carbon" in text):
        data["grid_carbon_year"] = int(m.group(1))
    for goal, pat in GOALS:
        if re.search(pat, text) and ("goal" in text or "want" in text or "priorit" in text or "focus" in text or "reach" in text):
            data["goal"] = goal
            break
    if add or remove or "what if" in text or "what about" in text or data.keys() - {"retrofits_add", "retrofits_remove"}:
        data["question_type"] = "what_if"
    elif re.search(r"why|explain|how come|what does", text):
        data["question_type"] = "explain_result"
    else:
        data["question_type"] = "general"
    try:
        return Scenario(**data)
    except ValidationError:
        return Scenario(question_type=data["question_type"], retrofits_add=add, retrofits_remove=remove)


def parse(message: str, chain: LLMChain | None = None) -> tuple[Scenario, str]:
    chain = chain or LLMChain()
    try:
        raw, provider = chain.generate(SYSTEM, f"User message: {message}", json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return Scenario.model_validate(json.loads(raw)), provider
    except (ProviderError, ValueError, ValidationError, json.JSONDecodeError):
        return rule_parse(message), "rules"
