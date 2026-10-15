"""Demand forecasting with scikit-learn (GradientBoostingRegressor, one model per product)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

FEATURES = ["dow", "seasonal_factor", "price_ratio", "lag_1", "lag_7", "roll_7"]
HOLDOUT_DAYS = 14
MIN_MODEL_ROWS = 28  # usable feature rows needed to train the ML model


@dataclass
class ForecastResult:
    product_id: str
    history: pd.DataFrame   # columns: date, actual, predicted (predicted only on the hold-out period)
    future: pd.DataFrame    # columns: date, predicted
    mae: float
    wape: float             # weighted absolute % error on hold-out (lower is better)
    method: str


def _prepare(g: pd.DataFrame) -> pd.DataFrame:
    """Make the series daily (fill gaps) and add model features."""
    g = g.sort_values("date").set_index("date")
    g = g[["current_price", "units_sold", "competitor_price", "seasonal_factor"]].astype(float)
    g = g.asfreq("D")
    g = g.interpolate().bfill().ffill()
    g["dow"] = g.index.dayofweek
    g["price_ratio"] = g["current_price"] / g["competitor_price"]
    g["lag_1"] = g["units_sold"].shift(1)
    g["lag_7"] = g["units_sold"].shift(7)
    g["roll_7"] = g["units_sold"].shift(1).rolling(7).mean()
    return g


def _new_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )


def forecast_product(product_id: str, g: pd.DataFrame, horizon: int = 14) -> ForecastResult:
    feat = _prepare(g)
    data = feat.dropna(subset=FEATURES)
    units = feat["units_sold"]
    last_date = feat.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Fallback for short history: 7-day moving average (clearly labelled)
    if len(data) < MIN_MODEL_ROWS + HOLDOUT_DAYS:
        avg = float(units.tail(7).mean())
        history = pd.DataFrame({"date": feat.index, "actual": units.values, "predicted": np.nan})
        future = pd.DataFrame({"date": future_dates, "predicted": avg})
        return ForecastResult(product_id, history, future, float("nan"), float("nan"),
                              "7-day moving average (not enough history for ML)")

    train, test = data.iloc[:-HOLDOUT_DAYS], data.iloc[-HOLDOUT_DAYS:]
    model = _new_model().fit(train[FEATURES], train["units_sold"])
    pred_test = np.clip(model.predict(test[FEATURES]), 0, None)
    mae = float(np.mean(np.abs(test["units_sold"].values - pred_test)))
    denom = float(test["units_sold"].sum())
    wape = float(np.abs(test["units_sold"].values - pred_test).sum() / denom) if denom > 0 else float("nan")

    # Refit on all usable data, then forecast forward one day at a time
    final = clone(model).fit(data[FEATURES], data["units_sold"])
    hist_units = list(units.values)
    price = float(feat["current_price"].iloc[-1])
    comp = float(feat["competitor_price"].iloc[-1])
    season = float(feat["seasonal_factor"].tail(7).mean())  # assumption: recent seasonality continues
    preds = []
    for d in future_dates:
        row = pd.DataFrame([{
            "dow": d.dayofweek, "seasonal_factor": season, "price_ratio": price / comp,
            "lag_1": hist_units[-1], "lag_7": hist_units[-7], "roll_7": float(np.mean(hist_units[-7:])),
        }])[FEATURES]
        p = float(max(final.predict(row)[0], 0.0))
        preds.append(p)
        hist_units.append(p)

    predicted = pd.Series(np.nan, index=feat.index)
    predicted.loc[test.index] = pred_test
    history = pd.DataFrame({"date": feat.index, "actual": units.values, "predicted": predicted.values})
    future = pd.DataFrame({"date": future_dates, "predicted": preds})
    return ForecastResult(product_id, history, future, mae, wape, "Gradient Boosting (scikit-learn)")


def forecast_all(df: pd.DataFrame, horizon: int = 14) -> dict:
    return {pid: forecast_product(pid, g, horizon) for pid, g in df.groupby("product_id")}


def accuracy_table(forecasts: dict, names: dict) -> pd.DataFrame:
    rows = [{
        "product_id": pid, "product_name": names.get(pid, pid), "method": r.method,
        "holdout_MAE_units": round(r.mae, 2) if pd.notna(r.mae) else None,
        "holdout_WAPE_%": round(r.wape * 100, 1) if pd.notna(r.wape) else None,
        "avg_forecast_units_per_day": round(float(r.future["predicted"].mean()), 1),
    } for pid, r in forecasts.items()]
    return pd.DataFrame(rows)
