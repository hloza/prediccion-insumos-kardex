"""Train LightGBM and compare it with the seasonal naive baseline (lag 7).

Temporal split (no shuffling):
    train       2021-10-29 .. 2025-06-30
    validation  2025-07-01 .. 2025-12-31  (objective choice + early stopping)
    test        2026-01-01 .. 2026-10-03  (used once, at the end)

Outputs:
    model/artifacts/model.txt, model/artifacts/metadata.json
    results/metrics.json, results/metrics_by_item.csv, results/test_predictions.csv
    results/*.png
"""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import FEATURES, MIN_HISTORY_DAYS, add_features, scale_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "model" / "artifacts"
RES = ROOT / "results"
SEED = 42
VAL_START, TEST_START = "2025-07-01", "2026-01-01"
OBJECTIVES = ["regression", "regression_l1", "tweedie"]


def mase_table(df, pred_col, naive_scale):
    """Per-item MAE, MASE, WAPE and bias for one prediction column."""
    rows = []
    for item, g in df.groupby("item"):
        err = g[pred_col] - g["consumption"]
        mae = err.abs().mean()
        rows.append({
            "item": item,
            "mae": mae,
            "mase": mae / naive_scale[item],
            "wape": err.abs().sum() / g["consumption"].sum(),
            "bias": err.sum() / g["consumption"].sum(),
        })
    return pd.DataFrame(rows).set_index("item")


def fit(train, valid, objective, scales):
    params = {
        "objective": objective, "learning_rate": 0.03, "num_leaves": 31,
        "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "seed": SEED, "verbose": -1,
        "metric": "l1",
    }
    dtr = lgb.Dataset(scale_features(train, scales)[FEATURES], train["y"],
                      categorical_feature=["item_code"])
    dva = lgb.Dataset(scale_features(valid, scales)[FEATURES], valid["y"],
                      categorical_feature=["item_code"], reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster, params


def predict(booster, df, scales):
    raw = booster.predict(scale_features(df, scales)[FEATURES],
                          num_iteration=booster.best_iteration)
    return np.clip(raw, 0, None) * df["item"].map(scales).values


def main():
    ART.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(ROOT / "data" / "daily_consumption.csv")
    items = sorted(daily["item"].unique())
    item_codes = {name: i for i, name in enumerate(items)}
    units = daily.groupby("item")["unit"].first().to_dict()

    df = add_features(daily, item_codes)
    first_day = df["date"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    df = df[df["date"] >= first_day].dropna(subset=FEATURES).copy()

    train = df[df["date"] < VAL_START].copy()
    valid = df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)].copy()
    test = df[df["date"] >= TEST_START].copy()

    # scales come from training data only
    scales = train.groupby("item")["consumption"].mean().to_dict()
    naive_scale = train.assign(e=(train["consumption"] - train["lag_7"]).abs()) \
        .groupby("item")["e"].mean().to_dict()
    for part in (train, valid, test):
        part["y"] = part["consumption"] / part["item"].map(scales)

    # model selection on validation only
    selection = {}
    best = None
    for obj in OBJECTIVES:
        booster, params = fit(train, valid, obj, scales)
        valid["pred"] = predict(booster, valid, scales)
        score = mase_table(valid, "pred", naive_scale)["mase"].mean()
        selection[obj] = {"val_mase": round(float(score), 4), "iterations": booster.best_iteration}
        print(f"{obj:15s} val MASE={score:.4f} iters={booster.best_iteration}")
        if best is None or score < best[0]:
            best = (score, obj, booster, params)
    _, best_obj, booster, params = best
    # residual spread on validation, used by the API for the safety stock
    valid["pred"] = predict(booster, valid, scales)
    val_resid_std = (valid["pred"] - valid["consumption"]).groupby(valid["item"]).std().to_dict()

    # final evaluation on the untouched test set
    test["model"] = predict(booster, test, scales)
    test["baseline"] = test["lag_7"]
    valid["baseline"] = valid["lag_7"]
    m_model = mase_table(test, "model", naive_scale)
    m_base = mase_table(test, "baseline", naive_scale)
    by_item = m_model.add_prefix("model_").join(m_base.add_prefix("baseline_"))
    by_item["unit"] = [units[i] for i in by_item.index]
    by_item["model_better"] = by_item["model_mase"] < by_item["baseline_mase"]
    by_item.round(4).to_csv(RES / "metrics_by_item.csv")

    def summary(col):
        err = test[col] - test["consumption"]
        t = mase_table(test, col, naive_scale)
        weekly = test.assign(e=test[col], w=test["date"].dt.to_period("W")) \
            .groupby(["item", "w"])[["e", "consumption"]].sum()
        return {
            "mase_mean": round(float(t["mase"].mean()), 4),
            "mase_median": round(float(t["mase"].median()), 4),
            "wape_mean": round(float(t["wape"].mean()), 4),
            "bias_mean": round(float(t["bias"].mean()), 4),
            "weekly_wape_mean": round(float(((weekly["e"] - weekly["consumption"]).abs()
                                             .groupby("item").sum()
                                             / weekly["consumption"].groupby("item").sum()).mean()), 4),
            "zero_day_mae_scaled": round(float((err.abs() / test["item"].map(scales))
                                               [test["consumption"] == 0].mean()), 4),
        }

    metrics = {
        "split": {"train": [str(train["date"].min().date()), str(train["date"].max().date())],
                  "validation": [str(valid["date"].min().date()), str(valid["date"].max().date())],
                  "test": [str(test["date"].min().date()), str(test["date"].max().date())],
                  "rows": {"train": len(train), "validation": len(valid), "test": len(test)}},
        "selection_on_validation": selection,
        "chosen_objective": best_obj,
        "test_model": summary("model"),
        "test_baseline": summary("baseline"),
        "items_where_model_wins": int(by_item["model_better"].sum()),
        "items_total": len(by_item),
    }
    # error analysis by weekday and holidays (test)
    test["abs_err_model"] = (test["model"] - test["consumption"]).abs() / test["item"].map(scales)
    test["abs_err_base"] = (test["baseline"] - test["consumption"]).abs() / test["item"].map(scales)
    metrics["error_by_dow_scaled"] = test.groupby("dow")[["abs_err_model", "abs_err_base"]] \
        .mean().round(3).to_dict()
    metrics["error_holiday_scaled"] = test.groupby("is_holiday")[["abs_err_model", "abs_err_base"]] \
        .mean().round(3).to_dict()
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    test[["date", "item", "consumption", "model", "baseline"]].round({"model": 2, "baseline": 2}) \
        .to_csv(RES / "test_predictions.csv", index=False)

    booster.save_model(str(ART / "model.txt"), num_iteration=booster.best_iteration)
    meta = {"item_codes": item_codes, "units": units, "scales": scales,
            "naive_scale": naive_scale, "val_resid_std": val_resid_std, "features": FEATURES, "params": params,
            "best_iteration": booster.best_iteration, "objective": best_obj,
            "lightgbm_version": lgb.__version__, "seed": SEED,
            "test_mase_by_item": by_item["model_mase"].round(3).to_dict(),
            "baseline_mase_by_item": by_item["baseline_mase"].round(3).to_dict()}
    (ART / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    plot(test, by_item, booster)
    print(json.dumps({k: metrics[k] for k in ("chosen_objective", "test_model", "test_baseline",
                                                "items_where_model_wins")}, indent=2))
    print(by_item[["model_mase", "baseline_mase", "model_wape", "baseline_wape", "model_bias"]]
          .round(3).to_string())


def plot(test, by_item, booster):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = by_item.sort_values("model_mase").index
    y = np.arange(len(order))
    ax.barh(y - 0.2, by_item.loc[order, "baseline_mase"], 0.4, label="Línea base (lag 7)", color="#9aa5b1")
    ax.barh(y + 0.2, by_item.loc[order, "model_mase"], 0.4, label="LightGBM", color="#1f6f8b")
    ax.axvline(1, color="#c0392b", lw=1, ls="--")
    ax.set_yticks(y, order)
    ax.set_xlabel("MASE en prueba (menor es mejor)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "mase_por_insumo.png", dpi=150)
    plt.close(fig)

    for item in ["Leche entera", "Harina"]:
        g = test[test["item"] == item].set_index("date").loc["2026-08-01":]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(g.index, g["consumption"], color="#222", lw=1.6, label="Real")
        ax.plot(g.index, g["baseline"], color="#9aa5b1", lw=1.2, ls="--", label="Línea base")
        ax.plot(g.index, g["model"], color="#1f6f8b", lw=1.6, label="LightGBM")
        ax.set_title(f"{item}: consumo diario (ago-oct 2026, prueba)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RES / f"serie_{item.split()[0].lower()}.png", dpi=150)
        plt.close(fig)

    imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values / imp.sum(), color="#1f6f8b")
    ax.set_xlabel("Importancia relativa (gain)")
    fig.tight_layout()
    fig.savefig(RES / "importancia_variables.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
