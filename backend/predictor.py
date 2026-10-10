"""Load the trained LightGBM model and produce multi-day forecasts.

Forecasts are recursive: the prediction for day t+1 is appended to the
history and used as a lag for day t+2, and the stock is reduced by the
predicted consumption. Only the model artifact is used; nothing is precomputed.
"""
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from features import FEATURES, MIN_HISTORY_DAYS, add_features, scale_features  # noqa: E402

ART = ROOT / "model" / "artifacts"
SERVICE_Z = 1.65  # ~95 % service level for the safety stock


class Predictor:
    def __init__(self, art_dir=ART):
        self.meta = json.loads((art_dir / "metadata.json").read_text(encoding="utf-8"))
        self.booster = lgb.Booster(model_file=str(art_dir / "model.txt"))
        self.items = sorted(self.meta["item_codes"])

    def forecast(self, item, history, stock_now, horizon):
        """history: DataFrame with columns date, consumption (consecutive days)."""
        scales = self.meta["scales"]
        hist = history[["date", "consumption"]].copy()
        hist["date"] = pd.to_datetime(hist["date"])
        hist["item"] = item
        # stock is only known today; earlier closing stock is not needed for t+1
        hist["stock_close"] = np.nan
        hist.loc[hist.index[-1], "stock_close"] = stock_now

        stock = float(stock_now)
        out = []
        for _ in range(horizon):
            next_day = hist["date"].iloc[-1] + pd.Timedelta(days=1)
            row = pd.DataFrame({"date": [next_day], "item": [item],
                                "consumption": [np.nan], "stock_close": [np.nan]})
            frame = add_features(pd.concat([hist, row], ignore_index=True),
                                 self.meta["item_codes"])
            x = scale_features(frame.tail(1), scales)[FEATURES]
            y = float(max(self.booster.predict(x)[0], 0.0)) * scales[item]
            out.append({"fecha": next_day.strftime("%Y-%m-%d"), "consumo_predicho": round(y, 2),
                        "linea_base": round(float(frame["lag_7"].iloc[-1]), 2),
                        "stock_estimado": round(max(stock - y, 0.0), 2)})
            stock = max(stock - y, 0.0)
            row["consumption"] = y
            row["stock_close"] = stock
            hist = pd.concat([hist, row], ignore_index=True)
        return out

    def replenishment(self, item, forecast, stock_now, lead_time):
        """Days of coverage and suggested order for the forecast window."""
        cum, cover = 0.0, None
        for i, f in enumerate(forecast):
            cum += f["consumo_predicho"]
            if cover is None and cum > stock_now:
                cover = i  # stock runs out during day i+1
        total = sum(f["consumo_predicho"] for f in forecast)
        sigma = self.meta["val_resid_std"][item]
        safety = SERVICE_Z * sigma * math.sqrt(max(lead_time, 1))
        order = max(total + safety - stock_now, 0.0)
        return {
            "consumo_total_predicho": round(total, 2),
            "dias_cobertura": cover if cover is not None else len(forecast),
            "cubre_todo_el_horizonte": cover is None,
            "stock_seguridad": round(safety, 2),
            "cantidad_sugerida": round(order, 2),
            "pedir_hoy": cover is not None and cover <= lead_time,
        }


def check_history(df):
    """Return an error message in Spanish, or None if the history is usable."""
    if len(df) < MIN_HISTORY_DAYS:
        return f"Se necesitan al menos {MIN_HISTORY_DAYS} días de historial; llegaron {len(df)}."
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        return "Hay fechas con formato inválido (use AAAA-MM-DD)."
    if dates.duplicated().any():
        return "Hay fechas repetidas en el historial."
    gaps = dates.sort_values().diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        return "El historial debe tener días consecutivos, sin saltos (use 0 si no hubo consumo)."
    return None
