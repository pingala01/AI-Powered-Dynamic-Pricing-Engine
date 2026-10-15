"""Rule-based pricing engine driven by ML demand forecasts, inventory, competitor prices and flash-sale signals."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MAX_CHANGE = 0.15
MAX_CHANGE_FLASH = 0.20
DEFAULT_ELASTICITY = -1.2


def estimate_elasticity(g: pd.DataFrame):
    """Rough log-log price elasticity (controls for seasonal factor). Falls back to a default."""
    g = g[(g["units_sold"] > 0) & (g["current_price"] > 0)]
    if len(g) < 30 or g["current_price"].nunique() < 3:
        return DEFAULT_ELASTICITY, "default"
    X = np.column_stack([np.log(g["current_price"]), np.log(g["seasonal_factor"].clip(lower=0.01))])
    coef = LinearRegression().fit(X, np.log(g["units_sold"])).coef_[0]
    if not np.isfinite(coef) or coef > -0.2:
        return DEFAULT_ELASTICITY, "default"
    return float(np.clip(coef, -3.0, -0.3)), "estimated"


def recommend_prices(df: pd.DataFrame, forecasts: dict, flash: pd.DataFrame,
                     low_cover_days: float = 10, excess_cover_days: float = 60) -> pd.DataFrame:
    flash_map = flash.set_index("product_id").to_dict("index") if len(flash) else {}
    rows = []
    for pid, g in df.groupby("product_id"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        cur, comp, inv = float(last["current_price"]), float(last["competitor_price"]), float(last["inventory_level"])
        f_daily = float(forecasts[pid].future["predicted"].head(7).mean())
        base28 = float(g["units_sold"].tail(28).mean())
        ratio = f_daily / base28 if base28 > 0 else 1.0
        cover = inv / max(f_daily, 0.1)
        gap = cur / comp - 1
        elasticity, e_src = estimate_elasticity(g)
        fl = flash_map.get(pid, {})
        parts, reasons = [], []

        # 1) Demand signal (an uplift is ignored when stock is already in excess)
        excess_stock = cover >= excess_cover_days
        if ratio >= 1.10 and excess_stock:
            reasons.append(f"Forecast demand is {ratio - 1:.0%} above the 28-day average, but stock is in excess, so no demand-based increase is applied.")
        elif ratio >= 1.10:
            parts.append(min((ratio - 1) * 0.25, 0.06))
            reasons.append(f"Forecast demand ({f_daily:.1f} units/day) is {ratio - 1:.0%} above the 28-day average, so a higher price is supported.")
        elif ratio <= 0.90:
            parts.append(max((ratio - 1) * 0.25, -0.05))
            reasons.append(f"Forecast demand ({f_daily:.1f} units/day) is {1 - ratio:.0%} below the 28-day average, so a lower price may help.")
        else:
            reasons.append(f"Forecast demand ({f_daily:.1f} units/day) is stable versus the 28-day average.")

        # 2) Inventory signal
        low_stock = cover <= low_cover_days
        if cover <= low_cover_days / 2:
            parts.append(0.06)
            reasons.append(f"Very low stock: only {cover:.1f} days of cover. Raising the price slows sell-through.")
        elif low_stock:
            parts.append(0.04)
            reasons.append(f"Low stock: {cover:.1f} days of cover. A small increase protects availability.")
        elif cover >= 2 * excess_cover_days:
            parts.append(-0.08)
            reasons.append(f"Large excess stock: {cover:.0f} days of cover. A discount can clear inventory.")
        elif cover >= excess_cover_days:
            parts.append(-0.05)
            reasons.append(f"Excess stock: {cover:.0f} days of cover. A modest discount can speed up sales.")
        else:
            reasons.append(f"Inventory is healthy ({cover:.0f} days of cover).")

        # 3) Competitor signal
        if gap > 0.05:
            if low_stock:
                reasons.append(f"Our price is {gap:.0%} above the competitor, but low stock means we do not cut the price.")
            else:
                parts.append(-min(gap * 0.5, 0.06))
                reasons.append(f"Our price is {gap:.0%} above the competitor (${comp:.2f}), so a reduction improves competitiveness.")
        elif gap < -0.05:
            parts.append(min(-gap * 0.4, 0.04))
            reasons.append(f"Our price is {abs(gap):.0%} below the competitor (${comp:.2f}), leaving room to increase.")
        else:
            reasons.append(f"Our price is within 5% of the competitor price (${comp:.2f}).")

        # 4) Flash-sale emergency adjustment
        is_flash = fl.get("status") == "Flash Sale"
        if is_flash:
            adj = float(fl.get("emergency_adjust_pct", 0.0)) / 100
            if adj > 0:
                parts.append(adj)
            reasons.append("Flash-sale pattern detected: " + str(fl.get("note", "")))
        elif fl.get("status") == "Watch":
            reasons.append("Demand is rising and is on the flash-sale watch list.")

        limit = MAX_CHANGE_FLASH if is_flash else MAX_CHANGE
        pct = float(np.clip(sum(parts), -limit, limit))
        new_price = cur * (1 + pct)
        ceiling = comp * 1.20
        if new_price > ceiling and not (low_stock or is_flash) and ceiling > cur:
            new_price = ceiling
            reasons.append("Price capped at 120% of the competitor price.")
        pct = new_price / cur - 1
        if abs(pct) < 0.01:
            new_price, pct, action = cur, 0.0, "Hold"
        else:
            action = "Increase" if pct > 0 else "Decrease"
        new_price = round(new_price, 2)

        # Model-based 7-day estimate only (uses estimated elasticity, units capped by stock); not a guarantee.
        units_now = min(7 * f_daily, inv)
        units_new = min(7 * f_daily * (new_price / cur) ** elasticity, inv)
        est_change = new_price * units_new - cur * units_now
        rows.append({
            "product_id": pid, "product_name": last["product_name"], "current_price": round(cur, 2),
            "competitor_price": round(comp, 2), "recommended_price": new_price,
            "change_pct": round((new_price / cur - 1) * 100, 1), "action": action,
            "forecast_daily_units": round(f_daily, 1), "demand_ratio": round(ratio, 2),
            "inventory_level": int(inv), "days_of_cover": round(cover, 1),
            "flash_status": fl.get("status", "Normal"), "elasticity": round(elasticity, 2),
            "elasticity_source": e_src, "est_7d_revenue_change": round(est_change, 2),
            "reasons": reasons, "reason_text": " | ".join(reasons),
        })
    return pd.DataFrame(rows)
