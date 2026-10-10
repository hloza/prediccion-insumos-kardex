"""Feature engineering shared by training and the API.

All features for date t use only information known at the end of day t-1
(past consumption and previous closing stock) plus the calendar of day t.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 7, 14, 21, 28]
MIN_HISTORY_DAYS = 28

FEATURES = (
    ["item_code"]
    + [f"lag_{k}" for k in LAGS]
    + ["roll_mean_7", "roll_mean_28", "roll_std_7", "zero_share_28",
       "stock_prev", "stock_cover_days",
       "dow", "month", "day", "is_holiday", "is_weekend"]
)


def easter(year):
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def holidays(year):
    """National holidays of Bolivia plus the Tarija departmental day (15 April)."""
    e = easter(year)
    fixed = [(1, 1), (1, 22), (4, 15), (5, 1), (6, 21), (8, 6), (11, 2), (12, 25)]
    moving = [e - timedelta(days=48), e - timedelta(days=47),  # carnival
              e - timedelta(days=2),                           # Good Friday
              e + timedelta(days=60)]                          # Corpus Christi
    return {date(year, mo, d) for mo, d in fixed} | set(moving)


def is_holiday(dates):
    years = {d.year for d in dates}
    hol = set().union(*(holidays(y) for y in years))
    return np.array([d.date() in hol for d in dates], dtype=int)


def add_features(df, item_codes):
    """df columns: date, item, consumption, stock_close. Sorted output with features.

    The target of each row is `consumption`; every feature is shifted so the
    row's own consumption and closing stock are never used.
    """
    df = df.sort_values(["item", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby("item")["consumption"]
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)
    past = g.shift(1)
    grp = past.groupby(df["item"])
    df["roll_mean_7"] = grp.transform(lambda s: s.rolling(7).mean())
    df["roll_mean_28"] = grp.transform(lambda s: s.rolling(28).mean())
    df["roll_std_7"] = grp.transform(lambda s: s.rolling(7).std())
    df["zero_share_28"] = grp.transform(lambda s: (s == 0).rolling(28).mean())
    df["stock_prev"] = df.groupby("item")["stock_close"].shift(1)
    df["stock_cover_days"] = df["stock_prev"] / (df["roll_mean_7"] + 1e-6)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = is_holiday(df["date"])
    df["item_code"] = df["item"].map(item_codes).astype(int)
    return df


def seasonal_naive(df):
    """Baseline: same weekday of the previous week (lag 7)."""
    return df["lag_7"]


SCALED = [f"lag_{k}" for k in LAGS] + ["roll_mean_7", "roll_mean_28", "roll_std_7", "stock_prev"]


def scale_features(df, scales):
    """Divide quantity features by the item's training mean so that items
    measured in ml, g or units share one model."""
    out = df.copy()
    s = out["item"].map(scales)
    for col in SCALED:
        out[col] = out[col] / s
    return out
