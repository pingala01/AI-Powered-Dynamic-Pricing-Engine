"""Flash-sale (demand surge) detection."""
import numpy as np
import pandas as pd


def detect_flash_sales(df: pd.DataFrame, recent_days: int = 3, baseline_days: int = 28,
                       z_flash: float = 2.5, ratio_flash: float = 1.5,
                       z_watch: float = 1.8, ratio_watch: float = 1.3,
                       low_cover_days: float = 7, medium_cover_days: float = 14) -> pd.DataFrame:
    """Compares the last `recent_days` of sales with the preceding `baseline_days`."""
    rows = []
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("date")
        name = g["product_name"].iloc[-1]
        if len(g) < recent_days + 14:
            rows.append({"product_id": pid, "product_name": name, "status": "Insufficient data",
                         "recent_avg_units": np.nan, "baseline_avg_units": np.nan, "demand_ratio": np.nan,
                         "z_score": np.nan, "price_change_pct": np.nan, "days_of_cover": np.nan,
                         "severity": "-", "emergency_adjust_pct": 0.0, "note": "Not enough history."})
            continue
        recent = g.tail(recent_days)
        base = g.iloc[:-recent_days].tail(baseline_days)
        b_mean = float(base["units_sold"].mean())
        b_std = max(float(base["units_sold"].std(ddof=0)), 0.1 * b_mean, 1.0)
        r_mean = float(recent["units_sold"].mean())
        ratio = r_mean / max(b_mean, 0.1)
        z = (r_mean - b_mean) / b_std
        price_chg = float(recent["current_price"].mean() / base["current_price"].mean() - 1)
        cover = float(g["inventory_level"].iloc[-1]) / max(r_mean, 0.1)

        if z >= z_flash and ratio >= ratio_flash:
            status = "Flash Sale"
        elif z >= z_watch and ratio >= ratio_watch:
            status = "Watch"
        else:
            status = "Normal"

        adj, severity, note = 0.0, "-", "No unusual demand detected."
        if status == "Flash Sale":
            severity = "High" if ratio >= 2.5 else "Medium"
            if price_chg <= -0.10:
                note = (f"Demand is {ratio:.1f}x normal, but our price was cut {abs(price_chg):.0%} recently; "
                        "the surge is probably promotion-driven. No emergency increase recommended.")
            else:
                adj = 10.0 if cover <= low_cover_days else 7.0 if cover <= medium_cover_days else 4.0
                if ratio >= 2.5:
                    adj *= 1.25
                note = (f"Demand is {ratio:.1f}x the {baseline_days}-day norm (z={z:.1f}) with no price cut. "
                        f"Stock covers about {cover:.1f} days at the surge rate. "
                        f"Emergency price increase of {adj:.1f}% suggested.")
        elif status == "Watch":
            severity = "Low"
            note = f"Demand is rising ({ratio:.1f}x normal) but not yet a flash-sale pattern. Monitor daily."
        rows.append({"product_id": pid, "product_name": name, "status": status,
                     "recent_avg_units": round(r_mean, 1), "baseline_avg_units": round(b_mean, 1),
                     "demand_ratio": round(ratio, 2), "z_score": round(z, 2),
                     "price_change_pct": round(price_chg * 100, 1), "days_of_cover": round(cover, 1),
                     "severity": severity, "emergency_adjust_pct": round(adj, 1), "note": note})
    return pd.DataFrame(rows)


def daily_anomalies(g: pd.DataFrame, window: int = 28, z_threshold: float = 2.5) -> pd.DataFrame:
    """Per-day z-score of units sold versus the previous `window` days (for charting)."""
    g = g.sort_values("date").copy()
    prev = g["units_sold"].shift(1)
    mean = prev.rolling(window, min_periods=14).mean()
    std = prev.rolling(window, min_periods=14).std(ddof=0)
    std = np.maximum(std, np.maximum(0.1 * mean, 1.0))
    g["baseline_mean"] = mean
    g["z_score"] = (g["units_sold"] - mean) / std
    g["is_spike"] = (g["z_score"] >= z_threshold) & (g["units_sold"] >= 1.5 * g["baseline_mean"])
    return g
