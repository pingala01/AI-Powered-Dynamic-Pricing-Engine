"""Generates a SYNTHETIC (fake) sample dataset for demonstration purposes only."""
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = 120
START_DATE = "2026-06-01"

# (id, name, base daily demand, reference price, competitor reference price, final inventory)
PRODUCTS = [
    ("P1001", "Wireless Earbuds", 45, 79.99, 79.99, 160),      # high demand, low stock
    ("P1002", "Smart Watch", 18, 199.00, 199.00, 500),         # normal
    ("P1003", "Yoga Mat", 6, 34.99, 34.99, 900),               # slow-moving, excess stock
    ("P1004", "Phone Case", 40, 14.99, 14.99, 300),            # flash-sale spike in last 3 days
    ("P1005", "Coffee Maker", 14, 89.99, 79.00, 350),          # competitor undercuts us
    ("P1006", "Desk Lamp", 12, 29.99, 36.00, 300),             # competitor is more expensive
    ("P1007", "Backpack", 22, 49.99, 49.99, 400),              # rising seasonal demand
    ("P1008", "Bluetooth Speaker", 25, 59.99, 59.99, 450),     # normal
]
DOW_FACTOR = np.array([0.95, 0.95, 0.97, 1.0, 1.08, 1.15, 1.05])  # Monday..Sunday


def make_sample_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
    t = np.arange(DAYS)
    frames = []
    for i, (pid, name, base, ref_price, comp_ref, end_inv) in enumerate(PRODUCTS):
        # Our price: small step changes every 14 days
        steps = rng.choice([-0.05, 0.0, 0.0, 0.04], size=DAYS // 14 + 1)
        price = np.round(ref_price * (1 + steps[t // 14]), 2)
        # Competitor price: gentle wave plus noise
        comp = np.round(comp_ref * (1 + 0.02 * np.sin(t / 9 + i) + rng.normal(0, 0.01, DAYS)), 2)
        # Seasonal factor
        if pid == "P1007":
            season = np.linspace(0.9, 1.4, DAYS)
        else:
            season = 1 + 0.10 * np.sin(2 * np.pi * t / 30 + i)
        season = np.round(season, 3)
        lam = (base * season * DOW_FACTOR[dates.dayofweek]
               * (price / ref_price) ** -1.4 * (comp / price) ** 0.3
               * rng.lognormal(0, 0.08, DAYS))
        if pid == "P1004":
            lam[-3:] *= 3.2  # flash-sale style surge
        units = rng.poisson(lam)
        # Inventory built backwards from the final level (with periodic restocks)
        inv = np.zeros(DAYS, dtype=int)
        inv[-1] = end_inv
        cap = max(end_inv * 2.2, base * 45)
        for k in range(DAYS - 1, 0, -1):
            nxt = int(inv[k] + units[k])
            if nxt > cap:
                nxt = int(cap * 0.4)
            inv[k - 1] = nxt
        frames.append(pd.DataFrame({
            "product_id": pid, "product_name": name, "date": dates,
            "current_price": price, "units_sold": units, "inventory_level": inv,
            "competitor_price": comp, "seasonal_factor": season,
        }))
    return pd.concat(frames, ignore_index=True)


def write_sample_csv(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_sample_data().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = write_sample_csv(Path(__file__).resolve().parent.parent / "data" / "sample_sales_data.csv")
    print(f"Sample (synthetic) dataset written to: {out}")
