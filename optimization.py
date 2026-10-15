"""Revenue and inventory analysis: stockout risk, excess stock, price-adjustment guidance."""
import pandas as pd


def analyze_inventory_revenue(df: pd.DataFrame, recs: pd.DataFrame, stockout_days: float = 10,
                              excess_days: float = 60, window: int = 30) -> pd.DataFrame:
    median_demand = recs["forecast_daily_units"].median()
    total_rev = 0.0
    rows = []
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("date")
        cur = g.tail(window)
        prev = g.iloc[:-window].tail(window)
        rev = float((cur["current_price"] * cur["units_sold"]).sum())
        prev_rev = float((prev["current_price"] * prev["units_sold"]).sum()) if len(prev) >= window // 2 else float("nan")
        units = float(cur["units_sold"].sum())
        r = recs[recs["product_id"] == pid].iloc[0]
        inv = float(r["inventory_level"])
        high_demand = bool(r["forecast_daily_units"] >= median_demand)  # top half of products by forecast volume
        cover = float(r["days_of_cover"])
        pct = float(r["change_pct"])

        if cover <= stockout_days and high_demand:
            status = "Stockout Risk"
            action = (f"Only {cover:.1f} days of cover for a high-demand product. Recommended price change "
                      f"{pct:+.1f}%; prioritise restocking.")
        elif cover >= excess_days and not high_demand:
            status = "Excess Stock"
            action = (f"{cover:.0f} days of cover for a slow-moving product. Recommended price change "
                      f"{pct:+.1f}%; consider a clearance promotion.")
        else:
            status = "Healthy"
            action = f"No inventory action needed. Recommended price change {pct:+.1f}%."
        total_rev += rev
        rows.append({
            "product_id": pid, "product_name": r["product_name"], f"revenue_last_{window}d": round(rev, 2),
            "revenue_change_vs_prev_%": round((rev / prev_rev - 1) * 100, 1) if prev_rev and prev_rev == prev_rev else None,
            f"units_last_{window}d": int(units), "inventory_level": int(inv),
            "forecast_daily_units": r["forecast_daily_units"], "days_of_cover": cover,
            "sell_through_%": round(units / (units + inv) * 100, 1) if (units + inv) > 0 else 0.0,
            "status": status, "current_price": r["current_price"],
            "recommended_price": r["recommended_price"], "change_pct": pct, "recommendation": action,
        })
    out = pd.DataFrame(rows)
    out["revenue_share_%"] = (out[f"revenue_last_{window}d"] / total_rev * 100).round(1) if total_rev > 0 else 0.0
    return out
