"""Scenario parser, number verifier and template fallback (no network)."""
from engine.inputs import CustomerInput, Usage
from engine.pipeline import analyse
from engine.whatif import run_whatif
from llm.explainer import explain
from llm.parser import rule_parse
from llm.providers import LLMChain, Provider
from llm.schema import Scenario
from llm.verifier import verify


def test_verifier_accepts_rounding_and_formatting():
    facts = {"bill": 1234.4, "kwh": -1942.0, "pct": 0.25, "years": 7.3}
    assert verify("Saves about £1,234 or roughly £1,200 a year.", facts)["ok"]
    assert verify("Electricity rises by 1,942 kWh.", facts)["ok"]
    assert verify("A 25% cut, payback 7.3 years.", facts)["ok"]
    assert verify("Cuts CO2 and P90 range, per m2.", facts)["ok"]


def test_verifier_rejects_invented_numbers():
    r = verify("You will save £5,000 in 3 years.", {"bill": 1234})
    assert not r["ok"] and "£5,000" in r["unsupported"]


def test_rule_parser():
    sc = rule_parse("What if I add a heat pump and 4 kWp solar with a budget of ₹10 lakh?")
    assert set(sc.retrofits_add) >= {"heat_pump", "solar_pv"}
    assert sc.solar_kwp == 4.0 and sc.budget == 1_000_000
    sc = rule_parse("I want to reach band C without double glazing")
    assert sc.goal == "reach_c" and sc.retrofits_remove == ["double_glazing"]
    assert rule_parse("Can I reach band C for ₹9,00,000?").budget == 900_000
    assert rule_parse("What can I do with a budget of ₹3,50,000?").budget == 350_000
    assert rule_parse("I have a rs 2 lakh budget").budget == 200_000
    assert rule_parse("what if gas goes to ₹11 per kWh").gas_price == 11.0
    assert rule_parse("Is a heat pump worth it if electricity costs 34 rupees?").electricity_price == 34.0


def test_scenario_money_in_rupees_converted_for_engine():
    from engine.currency import rate
    from engine.whatif import apply_scenario

    _, o = apply_scenario(_inp(), None, Scenario(budget=rate() * 3000, gas_price=rate() * 0.10))
    assert abs(o.budget - 3000) < 1e-6 and abs(o.gas_price - 0.10) < 1e-9


def test_verifier_handles_rupees():
    facts = {"cost": 134400, "bill": 18256}
    assert verify("Costs about ₹1,34,400 and saves Rs. 18,256 a year.", facts)["ok"]
    assert not verify("Saves ₹99,999.", facts)["ok"]


class BadProvider(Provider):
    def __init__(self):
        super().__init__("fake")

    def generate(self, system, prompt, json_mode=False, temperature=0.2):
        return "You will save £99,999 every year."


def _inp():
    return CustomerInput(postcode="LS6 2AB", property_type="semi", age_band="D", floor_area=90, bedrooms=3,
                         wall_type="cavity_unfilled", loft="under_100", main_heating="gas_boiler", tenure="own",
                         electricity=Usage(annual_kwh=3000), gas=Usage(annual_kwh=12000))


def test_unverified_llm_falls_back_to_template():
    res = analyse(_inp(), with_ranges=False)
    chain = LLMChain([])
    chain.providers = [BadProvider()]
    out = explain(res, "Explain", chain=chain)
    assert out["source"] == "template" and out["verified"]
    assert len(out["attempts"]) == 2 and not any(a["ok"] for a in out["attempts"])


def test_template_passes_verifier_including_infeasible_and_abstain():
    sc = Scenario(question_type="what_if", retrofits_add=["solid_wall_insulation", "heat_pump"])
    w = run_whatif(_inp(), sc, with_ranges=False)
    out = explain(w["results"], "what about solid wall insulation and a heat pump?", requested=w["requested_infeasible"], diff=w["diff"], chain=LLMChain(["none"]))
    assert out["verified"]
    assert "can't recommend solid wall insulation" in out["text"]
    bad = analyse(CustomerInput(postcode="LS6 2AB", property_type="semi", main_heating="gas_boiler"))
    out = explain(bad, chain=LLMChain(["none"]))
    assert "more information" in out["text"]
