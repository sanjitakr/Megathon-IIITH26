"""Energy balance identities and physics tests (spec 6.3.1)."""
import numpy as np
import pytest

from sim.energy_balance import solar_battery_flows
from sim.home import HomeState
from sim.parameters import days_in_month, params
from sim.retrofits import ORDER, apply_package
from sim.simulator import Calibration, simulate

FABRIC = ["loft_insulation", "cavity_wall_insulation", "floor_insulation", "double_glazing", "draught_proofing"]


def home(**kw):
    base = dict(property_type="semi", age_band="D", floor_area=90, region="london", storeys=2, occupants=2.5,
                wall_type="cavity_unfilled", loft="under_100", floor_type="suspended", glazing="double_old",
                heating="gas_boiler", boiler_age="10_15", gas_cooking=True, orientation="S", shading="none")
    base.update(kw)
    return HomeState(**base)


def check_balance(m, tol=1e-6):
    g, L, D, C, B, Z, X, I = (m[k] for k in ("g", "L", "D", "C_in", "B_out", "Z", "X", "I"))
    for arr in (g, L, D, C, B, Z, X, I):
        assert (arr >= -tol).all(), "every term is non-negative"
    np.testing.assert_allclose(g, D + C + X, atol=tol)
    np.testing.assert_allclose(L, D + B + I, atol=tol)
    np.testing.assert_allclose(C, B + Z, atol=tol)
    np.testing.assert_allclose(g + I, L + X + Z, atol=tol)


@pytest.mark.parametrize("battery", [0.0, 5.0, 13.5])
@pytest.mark.parametrize("kwp", [0.0, 2.0, 4.0, 10.0])
def test_balance_identities(kwp, battery):
    for heating in ("gas_boiler", "heat_pump", "electric"):
        r = simulate(home(solar_kwp=kwp, battery_kwh=battery, heating=heating, hot_water="heat_pump" if heating == "heat_pump" else "boiler"))
        check_balance(r.monthly)


def test_balance_random_flows():
    rng = np.random.default_rng(0)
    curve = params()["self_consumption_curve"]
    for _ in range(200):
        g, L = rng.uniform(0, 800, 12), rng.uniform(50, 1500, 12)
        m = solar_battery_flows(g, L, days_in_month(), curve, battery_kwh=rng.uniform(0, 15), eta_rt=rng.uniform(0.7, 0.95))
        check_balance(m)


def test_solar_and_battery_leave_gross_unchanged():
    base = simulate(home())
    pv = simulate(apply_package(home(), ["solar_pv"]))
    pvb = simulate(apply_package(home(), ["solar_pv", "battery"]))
    assert pv.gross["electricity"] == pytest.approx(base.gross["electricity"])
    assert pvb.gross["electricity"] == pytest.approx(base.gross["electricity"])
    assert pv.purchased["electricity"] < base.purchased["electricity"]
    assert pvb.purchased["electricity"] < pv.purchased["electricity"]


@pytest.mark.parametrize("rid", FABRIC)
def test_fabric_monotonic(rid):
    s = home()
    before = simulate(s).annual("Q")
    after = simulate(apply_package(s, [rid])).annual("Q")
    assert after <= before + 1e-9
    # and still true on top of any other fabric measure
    for other in FABRIC:
        if other != rid:
            b = simulate(apply_package(s, [other])).annual("Q")
            a = simulate(apply_package(s, [other, rid])).annual("Q")
            assert a <= b + 1e-9


def test_package_equals_sequenced_run():
    s = home()
    ids = ["smart_controls", "heat_pump", "loft_insulation", "solar_pv"]
    seq = s
    for rid in ORDER:
        if rid in ids:
            seq = apply_package(seq, [rid])
    a = simulate(apply_package(s, ids)).purchased
    b = simulate(seq).purchased
    for f in a:
        assert a[f] == pytest.approx(b[f])


def test_heat_pump_raises_electricity_and_keeps_gas_non_negative():
    s = home()
    base = simulate(s)
    hp = simulate(apply_package(s, ["heat_pump"]))
    assert hp.purchased["electricity"] > base.purchased["electricity"]
    assert 0 <= hp.purchased["gas"] <= base.purchased["gas"]
    assert hp.purchased["gas"] == pytest.approx(params()["cooking_gas_kwh"])  # only cooking gas remains


def test_fabric_first_raises_scop():
    s = home()
    hp_alone = simulate(apply_package(s, ["heat_pump"]))
    hp_fabric = simulate(apply_package(s, FABRIC + ["heat_pump"]))
    assert hp_fabric.scop > hp_alone.scop
    hp = params()["heat_pump"]
    assert hp["scop_min"] <= hp_alone.scop <= hp["scop_max"]


def test_off_gas_home_has_no_gas():
    r = simulate(home(heating="oil_boiler", gas_cooking=False))
    assert r.purchased["gas"] == 0
    assert r.purchased["oil"] > 0


def test_calibration_scales_heating_only():
    s = home()
    a = simulate(s, Calibration(kappa_home=1.0))
    b = simulate(s, Calibration(kappa_home=1.2))
    assert b.annual("heat_gas") == pytest.approx(1.2 * a.annual("heat_gas"))
    assert b.purchased["electricity"] == pytest.approx(a.purchased["electricity"])
