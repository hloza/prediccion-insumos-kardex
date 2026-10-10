"""FastAPI backend: validates input, builds features and runs the real model.

Run:  uvicorn backend.app:app --port 8000
"""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.predictor import Predictor, check_history

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "daily_consumption.csv"

app = FastAPI(title="Predicción de consumo de insumos")
predictor = Predictor()
dataset = pd.read_csv(DATASET) if DATASET.exists() else None


class Day(BaseModel):
    fecha: str
    consumo: float = Field(ge=0, description="Cantidad consumida en la unidad del insumo")


class PredictRequest(BaseModel):
    insumo: str
    historial: list[Day] = Field(min_length=1, max_length=400)
    stock_actual: float = Field(ge=0)
    horizonte: int = Field(default=7, ge=1, le=14)
    dias_entrega: int = Field(default=2, ge=0, le=14)

    @field_validator("insumo")
    @classmethod
    def known_item(cls, v):
        if v not in predictor.items:
            raise ValueError(f"Insumo desconocido: '{v}'.")
        return v


@app.get("/api/salud")
def health():
    return {"estado": "ok", "modelo": predictor.meta["objective"],
            "iteraciones": predictor.meta["best_iteration"]}


@app.get("/api/insumos")
def items():
    m = predictor.meta
    return [{"nombre": i, "unidad": m["units"][i],
             "mase_modelo": m["test_mase_by_item"][i],
             "mase_linea_base": m["baseline_mase_by_item"][i]} for i in predictor.items]


@app.get("/api/historial")
def history(insumo: str, dias: int = Query(56, ge=28, le=365), hasta: str | None = None):
    """Sample history from the dataset so the demo can be filled quickly."""
    if dataset is None:
        raise HTTPException(404, "El dataset no está disponible en el servidor.")
    if insumo not in predictor.items:
        raise HTTPException(404, f"Insumo desconocido: '{insumo}'.")
    d = dataset[dataset["item"] == insumo]
    if hasta:
        d = d[d["date"] <= hasta]
    d = d.tail(dias)
    if len(d) < 28:
        raise HTTPException(400, "No hay suficiente historial antes de esa fecha.")
    return {"insumo": insumo, "unidad": predictor.meta["units"][insumo],
            "stock_actual": float(d["stock_close"].iloc[-1]),
            "historial": [{"fecha": r.date, "consumo": float(r.consumption)} for r in d.itertuples()]}


@app.post("/api/predecir")
def predict(req: PredictRequest):
    hist = pd.DataFrame([{"date": h.fecha, "consumption": h.consumo} for h in req.historial])
    error = check_history(hist)
    if error:
        raise HTTPException(422, error)
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)
    forecast = predictor.forecast(req.insumo, hist, req.stock_actual, req.horizonte)
    plan = predictor.replenishment(req.insumo, forecast, req.stock_actual, req.dias_entrega)
    m = predictor.meta
    return {"insumo": req.insumo, "unidad": m["units"][req.insumo],
            "pronostico": forecast, "reposicion": plan,
            "confiabilidad": {"mase_modelo": m["test_mase_by_item"][req.insumo],
                              "mase_linea_base": m["baseline_mase_by_item"][req.insumo]}}


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")
