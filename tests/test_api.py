"""API tests: valid case, hard case, invalid inputs and the full flow."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app  # noqa: E402

client = TestClient(app)
DATA = pd.read_csv(ROOT / "data" / "daily_consumption.csv")


def history(item, until, days=56):
    d = DATA[(DATA["item"] == item) & (DATA["date"] <= until)].tail(days)
    return [{"fecha": r.date, "consumo": float(r.consumption)} for r in d.itertuples()], \
        float(d["stock_close"].iloc[-1])


def test_health_and_items():
    assert client.get("/api/salud").json()["estado"] == "ok"
    names = [i["nombre"] for i in client.get("/api/insumos").json()]
    assert "Leche entera" in names and len(names) == 14


def test_valid_case_milk():
    hist, stock = history("Leche entera", "2026-09-26")
    r = client.post("/api/predecir", json={"insumo": "Leche entera", "historial": hist,
                                           "stock_actual": stock, "horizonte": 7, "dias_entrega": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["pronostico"]) == 7
    assert all(f["consumo_predicho"] >= 0 for f in body["pronostico"])
    # prediction is in a realistic range for daily milk use (ml)
    mean_pred = sum(f["consumo_predicho"] for f in body["pronostico"]) / 7
    assert 1000 < mean_pred < 12000


def test_prediction_comes_from_model_not_constant():
    """Different histories must give different forecasts."""
    hist, stock = history("Leche entera", "2026-09-26")
    low = [{"fecha": h["fecha"], "consumo": h["consumo"] * 0.3} for h in hist]
    a = client.post("/api/predecir", json={"insumo": "Leche entera", "historial": hist, "stock_actual": stock}).json()
    b = client.post("/api/predecir", json={"insumo": "Leche entera", "historial": low, "stock_actual": stock}).json()
    assert a["pronostico"][0]["consumo_predicho"] != b["pronostico"][0]["consumo_predicho"]


def test_hard_case_intermittent_flour():
    """Flour is used in production batches: many zero days. The API must still answer
    with non-negative values and an order suggestion when stock is zero."""
    hist, _ = history("Harina", "2026-09-26")
    r = client.post("/api/predecir", json={"insumo": "Harina", "historial": hist,
                                           "stock_actual": 0, "horizonte": 14, "dias_entrega": 3})
    assert r.status_code == 200
    plan = r.json()["reposicion"]
    assert plan["dias_cobertura"] == 0 and plan["pedir_hoy"] is True
    assert plan["cantidad_sugerida"] > 0


@pytest.mark.parametrize("payload, fragment", [
    ({"insumo": "Gasolina", "historial": [{"fecha": "2026-01-01", "consumo": 1}], "stock_actual": 1}, "Insumo desconocido"),
    ({"insumo": "Leche entera", "historial": [{"fecha": "2026-01-01", "consumo": 1}], "stock_actual": 1}, "al menos 28"),
    ({"insumo": "Leche entera", "historial": [{"fecha": "2026-01-01", "consumo": -5}], "stock_actual": 1}, "consumo"),
    ({"insumo": "Leche entera", "historial": [{"fecha": "2026-01-01", "consumo": 5}], "stock_actual": 1, "horizonte": 60}, "horizonte"),
])
def test_invalid_inputs(payload, fragment):
    r = client.post("/api/predecir", json=payload)
    assert r.status_code == 422
    assert fragment in str(r.json()["detail"])


def test_gap_in_history_is_rejected():
    hist, stock = history("Queso", "2026-09-26")
    del hist[10]
    r = client.post("/api/predecir", json={"insumo": "Queso", "historial": hist, "stock_actual": stock})
    assert r.status_code == 422
    assert "consecutivos" in r.json()["detail"]


def test_full_flow_frontend_to_model():
    """Same path the UI follows: page, item list, history, prediction."""
    assert "Reposición de insumos" in client.get("/").text
    item = client.get("/api/insumos").json()[0]["nombre"]
    h = client.get("/api/historial", params={"insumo": item, "dias": 42, "hasta": "2026-09-20"}).json()
    r = client.post("/api/predecir", json={"insumo": item, "historial": h["historial"],
                                           "stock_actual": h["stock_actual"], "horizonte": 5})
    assert r.status_code == 200 and len(r.json()["pronostico"]) == 5
