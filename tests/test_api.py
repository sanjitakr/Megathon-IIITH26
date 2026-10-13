from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
INPUT = {"postcode": "LS6 2AB", "property_type": "semi", "age_band": "D", "floor_area": 90, "bedrooms": 3,
         "main_heating": "gas_boiler", "tenure": "own", "electricity": {"annual_kwh": 3000}, "gas": {"annual_kwh": 12000}}


def test_score_and_plan():
    r = client.post("/score", json=INPUT).json()
    assert r["status"] == "ok" and 0 <= r["score"]["score_rounded"] <= 100
    r = client.post("/plan", json={"input": INPUT, "options": {"goal": "cut_carbon"}, "with_ranges": False}).json()
    assert r["plan"]["goal"] == "cut_carbon"


def test_whatif_message_and_assumptions():
    r = client.post("/whatif", json={"input": INPUT, "message": "What if I add a heat pump?"}).json()
    assert "heat_pump" in r["scenario"]["retrofits_add"]
    assert r["explanation"]["verified"]
    assert "prices" in client.get("/assumptions").json()


def test_insufficient():
    r = client.post("/plan", json={"input": {"postcode": "LS6 2AB", "tenure": "own"}}).json()
    assert r["status"] == "insufficient_information"
