"""AI-Powered Dynamic Pricing Engine - Streamlit dashboard.
Run from the project folder:  python -m streamlit run app.py
"""
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_utils import REQUIRED_COLUMNS, DataValidationError, load_csv  # noqa: E402
from src.flash_sale import daily_anomalies, detect_flash_sales  # noqa: E402
from src.forecasting import accuracy_table, forecast_all  # noqa: E402
from src.generate_sample_data import write_sample_csv  # noqa: E402
from src.optimization import analyze_inventory_revenue  # noqa: E402
from src.pricing import recommend_prices  # noqa: E402

SAMPLE_PATH = ROOT / "data" / "sample_sales_data.csv"
STATUS_COLORS = {"Stockout Risk": "#d62728", "Excess Stock": "#ff7f0e", "Healthy": "#2ca02c"}

st.set_page_config(page_title="AI Dynamic Pricing Engine", layout="wide")


@st.cache_data(show_spinner=False)
def load_sample():
    if not SAMPLE_PATH.exists():
        write_sample_csv(SAMPLE_PATH)
    return load_csv(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def load_uploaded(data: bytes):
    return load_csv(io.BytesIO(data))


@st.cache_data(show_spinner="Training demand forecasting models...")
def run_forecasts(df: pd.DataFrame, horizon: int):
    return forecast_all(df, horizon)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
source = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 28, 14)
low_days = st.sidebar.slider("Low-stock threshold (days of cover)", 3, 21, 10)
excess_days = st.sidebar.slider("Excess-stock threshold (days of cover)", 30, 120, 60)
st.sidebar.caption("Required CSV columns: " + ", ".join(REQUIRED_COLUMNS))

# ---------------------------------------------------------------- load data
try:
    if source == "Sample data":
        df, data_warnings = load_sample()
        is_sample, source_label = True, "sample_sales_data.csv"
    else:
        upload = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
        if upload is None:
            st.title("AI-Powered Dynamic Pricing Engine")
            st.info("Upload a CSV file in the sidebar, or switch to 'Sample data'.")
            st.stop()
        df, data_warnings = load_uploaded(upload.getvalue())
        is_sample, source_label = False, upload.name
except DataValidationError as exc:
    st.title("AI-Powered Dynamic Pricing Engine")
    st.error(f"Data problem: {exc}")
    st.stop()
except Exception as exc:  # unexpected failure: show it clearly instead of a crash
    st.title("AI-Powered Dynamic Pricing Engine")
    st.error(f"Could not load the data: {exc}")
    st.stop()

# ---------------------------------------------------------------- run the pipeline
try:
    forecasts = run_forecasts(df, horizon)
    flash = detect_flash_sales(df)
    recs = recommend_prices(df, forecasts, flash, low_cover_days=low_days, excess_cover_days=excess_days)
    inv = analyze_inventory_revenue(df, recs, stockout_days=low_days, excess_days=excess_days)
except Exception as exc:
    st.title("AI-Powered Dynamic Pricing Engine")
    st.error(f"The analysis failed: {exc}")
    st.stop()

names = df.drop_duplicates("product_id", keep="last").set_index("product_id")["product_name"].to_dict()

# ---------------------------------------------------------------- header
st.title("AI-Powered Dynamic Pricing Engine")
if is_sample:
    st.warning("SAMPLE DATA: this dashboard is showing a synthetic (artificial) demo dataset, not real sales.")
else:
    st.success(f"Using your uploaded file: {source_label}")
for w in data_warnings:
    st.warning(w)

k = st.columns(6)
k[0].metric("Products", len(recs))
k[1].metric("Price increases", int((recs["action"] == "Increase").sum()))
k[2].metric("Price decreases", int((recs["action"] == "Decrease").sum()))
k[3].metric("Flash-sale alerts", int((flash["status"] == "Flash Sale").sum()))
k[4].metric("Stockout risks", int((inv["status"] == "Stockout Risk").sum()))
k[5].metric("Excess-stock items", int((inv["status"] == "Excess Stock").sum()))

tab_price, tab_fc, tab_inv, tab_flash, tab_data = st.tabs(
    ["Dynamic Pricing", "Demand Forecast", "Revenue & Inventory", "Flash Sale Detection", "Data & Method"])

# ---------------------------------------------------------------- tab 1: pricing
with tab_price:
    st.subheader("Current vs recommended prices")
    show = recs[["product_id", "product_name", "current_price", "recommended_price", "change_pct", "action",
                 "competitor_price", "forecast_daily_units", "days_of_cover", "flash_status",
                 "est_7d_revenue_change"]]
    st.dataframe(show, hide_index=True, column_config={
        "current_price": st.column_config.NumberColumn("Current price", format="$%.2f"),
        "recommended_price": st.column_config.NumberColumn("Recommended price", format="$%.2f"),
        "competitor_price": st.column_config.NumberColumn("Competitor price", format="$%.2f"),
        "change_pct": st.column_config.NumberColumn("Change %", format="%.1f%%"),
        "forecast_daily_units": st.column_config.NumberColumn("Forecast units/day"),
        "days_of_cover": st.column_config.NumberColumn("Days of cover"),
        "est_7d_revenue_change": st.column_config.NumberColumn("Est. 7-day revenue change", format="$%.2f"),
    })
    fig = go.Figure()
    fig.add_bar(name="Current", x=recs["product_name"], y=recs["current_price"])
    fig.add_bar(name="Recommended", x=recs["product_name"], y=recs["recommended_price"])
    fig.add_bar(name="Competitor", x=recs["product_name"], y=recs["competitor_price"])
    fig.update_layout(barmode="group", yaxis_title="Price", height=380, margin=dict(t=30))
    st.plotly_chart(fig)
    st.caption("'Est. 7-day revenue change' is a model-based estimate (forecast demand and a rough price-elasticity "
               "fit, with units capped by stock). It has not been validated with real price experiments and is "
               "not a guaranteed result.")

    st.subheader("Why each price was recommended")
    for _, r in recs.iterrows():
        with st.expander(f"{r['product_name']}: {r['action']} ({r['change_pct']:+.1f}%) "
                         f"${r['current_price']:.2f} -> ${r['recommended_price']:.2f}"):
            st.markdown("\n".join(f"- {x}" for x in r["reasons"]))
            st.caption(f"Price elasticity used: {r['elasticity']} ({r['elasticity_source']}).")
    st.download_button("Download recommendations (CSV)",
                       recs.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8"),
                       file_name="price_recommendations.csv", mime="text/csv")

# ---------------------------------------------------------------- tab 2: forecast
with tab_fc:
    st.subheader("Actual vs predicted demand")
    pid = st.selectbox("Product", list(forecasts.keys()), format_func=lambda p: f"{names.get(p, p)} ({p})")
    fr = forecasts[pid]
    hist = fr.history.tail(90)
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["actual"], name="Actual", mode="lines")
    hp = fr.history.dropna(subset=["predicted"])
    fig.add_scatter(x=hp["date"], y=hp["predicted"], name="Predicted (hold-out test)", mode="lines",
                    line=dict(dash="dash"))
    fig.add_scatter(x=fr.future["date"], y=fr.future["predicted"], name="Forecast (future)", mode="lines+markers",
                    line=dict(dash="dot"))
    fig.update_layout(yaxis_title="Units per day", height=420, margin=dict(t=30))
    st.plotly_chart(fig)
    c = st.columns(3)
    c[0].metric("Model", fr.method)
    c[1].metric("Hold-out MAE (units/day)", "n/a" if pd.isna(fr.mae) else f"{fr.mae:.2f}")
    c[2].metric("Hold-out WAPE", "n/a" if pd.isna(fr.wape) else f"{fr.wape:.1%}")
    st.caption("The model is trained on older data and tested on the last 14 days it has not seen (hold-out). "
               "Future forecasts assume the latest price, competitor price and seasonality continue.")
    st.subheader("Forecast accuracy for all products")
    st.dataframe(accuracy_table(forecasts, names), hide_index=True)

# ---------------------------------------------------------------- tab 3: revenue & inventory
with tab_inv:
    rev_col = [c for c in inv.columns if c.startswith("revenue_last_")][0]
    units_col = [c for c in inv.columns if c.startswith("units_last_")][0]
    c = st.columns(3)
    c[0].metric("Revenue (last 30 days)", f"${inv[rev_col].sum():,.0f}")
    c[1].metric("Units sold (last 30 days)", f"{inv[units_col].sum():,.0f}")
    c[2].metric("Units in stock", f"{inv['inventory_level'].sum():,.0f}")

    daily = df.assign(revenue=df["current_price"] * df["units_sold"]).groupby("date", as_index=False)["revenue"].sum()
    fig = go.Figure(go.Scatter(x=daily["date"], y=daily["revenue"], mode="lines", name="Revenue"))
    fig.update_layout(title="Total daily revenue", yaxis_title="Revenue", height=330, margin=dict(t=50))
    st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=inv["product_name"], y=inv[rev_col]))
        fig.update_layout(title="Revenue by product (last 30 days)", height=350, margin=dict(t=50))
        st.plotly_chart(fig)
    with right:
        fig = go.Figure()
        for status, color in STATUS_COLORS.items():
            sub = inv[inv["status"] == status]
            fig.add_bar(x=sub["product_name"], y=sub["days_of_cover"], name=status, marker_color=color)
        fig.add_hline(y=low_days, line_dash="dash", annotation_text="low-stock threshold")
        fig.add_hline(y=excess_days, line_dash="dash", annotation_text="excess threshold")
        fig.update_layout(title="Days of inventory cover", height=350, margin=dict(t=50))
        st.plotly_chart(fig)

    cols = ["product_name", "days_of_cover", "forecast_daily_units", "current_price", "recommended_price",
            "change_pct", "recommendation"]
    st.subheader("High-demand products at risk of stockout")
    risk = inv[inv["status"] == "Stockout Risk"]
    st.dataframe(risk[cols], hide_index=True) if len(risk) else st.success("No stockout risks detected.")
    st.subheader("Slow-moving products with excess stock")
    exc = inv[inv["status"] == "Excess Stock"]
    st.dataframe(exc[cols], hide_index=True) if len(exc) else st.success("No excess-stock products detected.")
    with st.expander("Full revenue and inventory table"):
        st.dataframe(inv, hide_index=True)

# ---------------------------------------------------------------- tab 4: flash sale
with tab_flash:
    st.subheader("Flash-sale detection")
    st.caption("Compares the last 3 days of sales with the 28 days before them. A flash-sale pattern needs a large "
               "statistical jump AND at least 1.5x the normal demand.")
    alerts = flash[flash["status"] == "Flash Sale"]
    if len(alerts):
        for _, r in alerts.iterrows():
            st.error(f"{r['product_name']}: {r['note']}")
    else:
        st.success("No flash-sale patterns detected.")
    for _, r in flash[flash["status"] == "Watch"].iterrows():
        st.warning(f"{r['product_name']}: {r['note']}")
    st.dataframe(flash.drop(columns=["note"]), hide_index=True)

    pid2 = st.selectbox("Product to inspect", list(forecasts.keys()), key="flash_pid",
                        format_func=lambda p: f"{names.get(p, p)} ({p})")
    an = daily_anomalies(df[df["product_id"] == pid2]).tail(60)
    fig = go.Figure()
    fig.add_scatter(x=an["date"], y=an["units_sold"], name="Units sold", mode="lines")
    fig.add_scatter(x=an["date"], y=an["baseline_mean"], name="Prior 28-day average", mode="lines",
                    line=dict(dash="dash"))
    sp = an[an["is_spike"]]
    fig.add_scatter(x=sp["date"], y=sp["units_sold"], name="Unusual spike", mode="markers",
                    marker=dict(size=11, color="red"))
    fig.update_layout(yaxis_title="Units per day", height=400, margin=dict(t=30))
    st.plotly_chart(fig)

# ---------------------------------------------------------------- tab 5: data & method
with tab_data:
    st.subheader("Data preview")
    st.caption(("SAMPLE (synthetic) data" if is_sample else f"Uploaded file: {source_label}")
               + f" | {len(df):,} rows | {df['product_id'].nunique()} products | "
               f"{df['date'].min().date()} to {df['date'].max().date()}")
    st.dataframe(df.tail(200), hide_index=True)
    if is_sample:
        st.download_button("Download the sample CSV as a template", df.to_csv(index=False).encode("utf-8"),
                           file_name="sample_sales_data.csv", mime="text/csv")
    st.subheader("How it works")
    st.markdown("""
1. **Demand forecasting:** one scikit-learn Gradient Boosting model per product, using weekday, seasonal factor,
   price-to-competitor ratio and recent sales. Accuracy is measured on a 14-day hold-out period.
2. **Flash-sale detection:** recent 3-day sales are compared with the previous 28 days.
3. **Dynamic pricing:** combines the forecast, inventory cover, competitor price gap and flash-sale signal into
   one price change (limited to +/-15%, or +/-20% during a flash sale). Every reason is listed.
4. **Revenue and inventory optimization:** classifies products as stockout-risk, excess-stock or healthy.

**Limitations:** the engine has no cost or margin data, so it cannot protect margins. Revenue effects are
estimates only and have not been validated with real price tests. Test any recommendation before relying on it.
""")
