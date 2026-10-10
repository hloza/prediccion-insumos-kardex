"""Build the daily dataset: one row per supply (insumo) and calendar date.

Input:  data/catalog.csv, data/raw_movements.csv (from extract_from_dump.py)
Output: data/daily_consumption.csv

Columns:
    date, item, unit, consumption, opened, waste, stock_close
consumption = (production use + sales use - sales returns) of the day.
stock_close = balance of open containers in the kardex at the end of the day.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

OUT_TYPES = {"Consumo produccion", "Consumo venta"}
RETURN_TYPES = {"Devolucion venta"}
WASTE_TYPES = {"Merma", "Vencimiento"}
IN_TYPES = {"Apertura", "Ajuste positivo", "Devolucion venta"}
NEG_TYPES = {"Consumo produccion", "Consumo venta", "Merma", "Vencimiento", "Ajuste negativo"}


def main():
    cat = pd.read_csv(DATA_DIR / "catalog.csv")
    mov = pd.read_csv(DATA_DIR / "raw_movements.csv", parse_dates=["timestamp"])

    # base supplies only (packages such as "Harina bolsa 1 kg" roll up to their parent)
    items = cat[(cat["type"] == "Insumo") & cat["parent_id"].isna()]
    names = items.set_index("product_id")["name"]
    units = items.set_index("name")["unit"]

    mov = mov[mov["base_product_id"].isin(names.index) & (mov["deleted"] == "f")].copy()
    mov["item"] = mov["base_product_id"].map(names)
    mov["date"] = mov["timestamp"].dt.normalize()
    t = mov["movement_type"]
    q = mov["quantity"]
    mov["consumption"] = q.where(t.isin(OUT_TYPES), 0) - q.where(t.isin(RETURN_TYPES), 0)
    mov["opened"] = q.where(t == "Apertura", 0)
    mov["waste"] = q.where(t.isin(WASTE_TYPES), 0)
    mov["delta"] = q.where(t.isin(IN_TYPES), 0) - q.where(t.isin(NEG_TYPES), 0)

    daily = mov.groupby(["item", "date"])[["consumption", "opened", "waste", "delta"]].sum()

    # full calendar for every item: days without movements are zero consumption
    dates = pd.date_range(mov["date"].min(), mov["date"].max(), freq="D")
    full_idx = pd.MultiIndex.from_product([names.sort_values().values, dates], names=["item", "date"])
    daily = daily.reindex(full_idx, fill_value=0).reset_index()
    daily["stock_close"] = daily.groupby("item")["delta"].cumsum().clip(lower=0)
    daily["unit"] = daily["item"].map(units)
    daily = daily[["date", "item", "unit", "consumption", "opened", "waste", "stock_close"]]
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily.to_csv(DATA_DIR / "daily_consumption.csv", index=False)
    print(daily.shape, daily["date"].min(), daily["date"].max())


if __name__ == "__main__":
    main()
