"""Validation tiers, baseline reconciliation, planner rules and organiser scenarios."""
import pytest

from engine.baseline import annualise
from engine.inputs import CustomerInput, CustomiseOptions, Usage
from engine.pipeline import analyse, get_stage1
from engine.validation import validate
from sim.parameters import electricity_shape

ELEC = [320, 290, 280, 250, 230, 210, 200, 205, 230, 260, 300, 330]
GAS = [2100, 1800, 1600, 1000, 600, 250, 180, 190, 350, 900, 1500, 2000]


def semi(**kw):
    base = dict(postcode="LS6 2AB", property_type="semi", age_band="D", floor_area=90, bedrooms=3, wall_type="cavity_unfilled",
                loft="under_100", glazing="double_old", main_heating="gas_boiler", boiler_age="10_15", controls="programmer",
                tenure="own", epc_band="D", outdoor_space="yes", electricity=Usage(monthly_kwh=ELEC), gas=Usage(monthly_kwh=GAS))
    base.update(kw)
    return CustomerInput(**base)


def test_tiers():
    assert validate(semi()).tier == "A"
    partial = ELEC[:6] + [None] * 6
    assert validate(semi(electricity=Usage(monthly_kwh=partial))).tier == "B"
    assert validate(semi(gas=Usage(annual_kwh=11000))).tier == "B"
    assert validate(semi(electricity=None, gas=None)).tier == "C"


def test_abstain_on_missing_and_conflict():
    v = validate(CustomerInput(postcode="LS6 2AB", property_type="semi", main_heating="gas_boiler"))
    assert v.status == "abstain" and {q["field"] for q in v.questions} == {"age_band", "floor_area"}
    v = validate(semi(main_heating="heat_pump", gas=Usage(annual_kwh=12000)))
    assert v.status == "abstain" and v.conflicts
    res = analyse(semi(main_heating="heat_pump", gas=Usage(annual_kwh=12000)))
    assert res["status"] == "insufficient_information"


def test_missing_months_not_zero():
    full, _ = annualise(ELEC, electricity_shape())
    holed, n = annualise(ELEC[:5] + [None, None] + ELEC[7:], electricity_shape())
    assert n == 10
    assert holed == pytest.approx(full, rel=0.08)
    assert holed > sum(x for x in ELEC[:5] + ELEC[7:])


def test_baseline_reconciles():
    res = analyse(semi(), with_ranges=False)
    for line in res["baseline"]["reconciliation"]:
        assert abs(line["residual_kwh"]) <= 0.03 * line["normalised_kwh"] + 1
    best = res["plan"]["best"]
    for f in ("electricity", "gas"):
        assert best["after_purchased_kwh"][f] == pytest.approx(best["baseline_purchased_kwh"][f] - best["delta_kwh"][f], abs=1)


def test_feasible_list_order_and_package():
    res = analyse(semi(), with_ranges=False)
    rows = res["retrofits"]["feasible_list"]
    tiers = [r["tier"] for r in rows]
    assert tiers == sorted(tiers)
    names = [r["retrofit"] for r in rows]
    assert "solid_wall_insulation" not in names
    assert any(n["retrofit"] == "solid_wall_insulation" for n in res["retrofits"]["not_available"])
    if "battery" in names and "solar_pv" in names:
        assert names.index("solar_pv") < names.index("battery")
    # package row is one simulator run, equal to the sum of sequenced rows by construction of the chain
    pk = res["retrofits"]["package_row"]
    seq_sum = sum(r["saving_in_sequence_kwh"]["gas"] for r in rows)
    assert pk["delta_kwh"]["gas"] == pytest.approx(seq_sum, abs=len(rows))


def test_heat_pump_negative_electricity_flagged():
    res = analyse(semi(), CustomiseOptions(tick=["heat_pump"], goal="cut_carbon"), with_ranges=False)
    best = res["plan"]["best"]
    assert "heat_pump" in best["retrofits"]
    hp_row = [r for r in best["rows"] if r["retrofit"] == "heat_pump"][0]
    assert hp_row["saving_in_sequence_kwh"]["electricity"] < 0
    assert any("Electricity use rises" in f for f in hp_row["flags"])


def test_budget_respected_and_no_fit_message():
    res = analyse(semi(), CustomiseOptions(goal="cut_bills", budget=3000), with_ranges=False)
    assert res["plan"]["best"]["cost_gbp"][1] <= 3000
    res = analyse(semi(), CustomiseOptions(goal="cut_bills", budget=50), with_ranges=False)
    assert res["plan"]["best"] is None
    assert "cheapest useful option" in res["plan"]["message"]


def test_infeasible_tick_explained():
    res = analyse(semi(), CustomiseOptions(tick=["solid_wall_insulation"]), with_ranges=False)
    assert any("cannot be included" in w for w in res["plan"]["warnings"])


def test_declined_and_disruption_excluded():
    res = analyse(semi(), CustomiseOptions(declined=["cavity_wall_insulation"], disruption_tolerance="low"), with_ranges=False)
    best = res["plan"]["best"]["retrofits"]
    assert "cavity_wall_insulation" not in best
    assert "heat_pump" not in best and "double_glazing" not in best


def test_reach_c_goal():
    res = analyse(semi(), CustomiseOptions(goal="reach_c"), with_ranges=False)
    assert res["plan"]["best"]["sap_after"] >= 69


def test_stage2_uses_cache():
    inp = semi()
    analyse(inp, with_ranges=False)
    b1 = get_stage1(inp, validate(inp))
    analyse(inp, CustomiseOptions(goal="cut_carbon", electricity_price=0.15), with_ranges=False)
    assert get_stage1(inp, validate(inp)) is b1


def test_tier_c_is_provisional():
    res = analyse(semi(electricity=None, gas=None), with_ranges=False)
    assert res["confidence"]["tier"] == "C"
    assert res["score"]["current"]["provisional"]


def test_tenant_flags():
    res = analyse(semi(tenure="rent"), with_ranges=False)
    names = [r["retrofit"] for r in res["retrofits"]["feasible_list"]]
    assert "heat_pump" not in names and "solar_pv" not in names
    cav = [r for r in res["retrofits"]["feasible_list"] if r["retrofit"] == "cavity_wall_insulation"][0]
    assert cav["needs_landlord_consent"]


def test_off_gas_improvement_potential_note():
    res = analyse(semi(main_heating="oil_boiler", gas=None, postcode="LL55 4AA"), with_ranges=False)
    assert res["status"] == "ok"
    assert res["baseline"]["purchased_kwh"]["oil"] > 0
    ip = res["efficiency_flag"]["improvement_potential"]
    assert "note" in ip
