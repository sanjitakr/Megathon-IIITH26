"""Green Home Score (spec 8)."""
import pytest

from engine.score import classify, compute_score, piecewise
from sim.home import HomeState
from sim.parameters import score_config


def brief_example_score():
    w = score_config()["weights"]
    subs = {"efficiency": 40, "intensity": 50, "insulation": 40, "heating": 55, "glazing": 20, "renewable": 0, "smart": 30}
    return sum(w[k] * subs[k] for k in w)


def test_brief_example_is_38():
    assert brief_example_score() == pytest.approx(37.75)
    assert round(brief_example_score()) == 38
    assert classify(38) == "Poor"


def test_class_edges():
    assert classify(80) == "Highly efficient"
    assert classify(79) == "Good"
    assert classify(65) == "Good"
    assert classify(64) == "Moderate"
    assert classify(50) == "Moderate"
    assert classify(49) == "Poor"
    assert classify(29) == "Very poor"


def test_piecewise_interpolates():
    a = score_config()["intensity_anchors"]
    assert piecewise(125, a["x"], a["y"]) == pytest.approx(70)
    assert piecewise(10, a["x"], a["y"]) == 100
    assert piecewise(1000, a["x"], a["y"]) == 0


def _state(**kw):
    base = dict(property_type="semi", age_band="D", floor_area=90, region="london", storeys=2, occupants=2.5,
                loft="under_100", wall_type="solid", floor_type="solid", glazing="single", heating="gas_boiler",
                boiler_age="10_15", controls="programmer")
    base.update(kw)
    return HomeState(**base)


KW = dict(sap_points=45, band="E", efficiency_basis="recorded", intensity_kwh_m2=300, intensity_basis="metered",
          generation_kwh=0, gross_electricity_kwh=3000)


def test_contributions_sum_to_score():
    r = compute_score(_state(), **KW)
    assert sum(c["contribution"] for c in r["components"]) == pytest.approx(r["score"], abs=0.05)
    assert r["class"] in ("Poor", "Very poor")


@pytest.mark.parametrize("field,seq", [
    ("loft", ["none", "under_100", "100_250", "over_250"]),
    ("wall_type", ["solid", "cavity_unfilled", "cavity_filled", "solid_insulated"]),
    ("glazing", ["single", "double_old", "double_modern", "triple"]),
    ("controls", ["none", "programmer", "smart_thermostat", "smart_trvs"]),
    ("boiler_age", ["over_15", "10_15", "5_10", "under_5"]),
])
def test_component_monotonic(field, seq):
    scores = [compute_score(_state(**{field: v}), **KW)["score"] for v in seq]
    assert scores == sorted(scores)


def test_intensity_and_efficiency_monotonic():
    s = _state()
    assert compute_score(s, **{**KW, "intensity_kwh_m2": 100})["score"] > compute_score(s, **{**KW, "intensity_kwh_m2": 300})["score"]
    assert compute_score(s, **{**KW, "sap_points": 80})["score"] > compute_score(s, **KW)["score"]
    assert compute_score(s, **{**KW, "generation_kwh": 1500})["score"] > compute_score(s, **KW)["score"]


def test_band_lower_edge_matches_band_table():
    cfg = score_config()
    from sim.rating import band_lower_edge
    from engine.score import efficiency_score
    for band, s in cfg["efficiency_band_score"].items():
        assert efficiency_score(band_lower_edge(band), band) == pytest.approx(s)


def test_middle_flat_reweights_insulation():
    flat = _state(property_type="flat", storeys=1, flat_position="middle")
    r = compute_score(flat, **KW)
    parts = r["insulation_parts"]
    assert parts["roof"]["weight"] == 0 and parts["floor"]["weight"] == 0 and parts["wall"]["weight"] == 1
