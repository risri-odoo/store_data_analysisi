#!/usr/bin/env python3
"""
streamlit_app.py
=================

A Streamlit UI wrapping the staffing_advisor.py analysis. Uploads a CSV,
runs the same ramp-curve + seasonality model (or the confidence-aware
fallback for sparse data) used by the CLI, and displays the scenario
comparison as a table and chart.

Run with:
    streamlit run streamlit_app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from staffing_advisor import (
    MIN_HIRES_FOR_RAMP,
    MIN_MONTHS_FOR_MODEL,
    build_tenure_table,
    detect_most_recent_departure,
    estimate_seasonality,
    fallback_recent_average_productivity,
    fit_ramp_curve,
    get_staff_tenures,
    load_data,
    project_scenarios,
)

st.set_page_config(page_title="RevInsight Staffing Advisor", layout="wide")

SCENARIO_LABELS = {"stay_as_is": "Remain at current staffing level", "hire_now": "Hire replacement immediately"}


def scenario_label(key):
    if key in SCENARIO_LABELS:
        return SCENARIO_LABELS[key]
    if key.startswith("delay_"):
        d = key.split("_")[1]
        return f"Delay hiring by {d} month(s)"
    return key


def run_analysis(df, sp_cols, horizon, salary, commission_rate, margin_rate):
    """Mirrors the model-selection logic in staffing_advisor.main()."""
    tenures, last_month_in_data = get_staff_tenures(df, sp_cols)
    departed_staff = detect_most_recent_departure(tenures, last_month_in_data)

    used_fallback = last_month_in_data < MIN_MONTHS_FOR_MODEL

    ramp_params = None
    seasonal_mult = None
    if not used_fallback:
        tenure_df = build_tenure_table(df, sp_cols)
        n_hires = tenure_df["salesperson"].nunique()
        seasonal_mult, _ = estimate_seasonality(df)
        if n_hires >= MIN_HIRES_FOR_RAMP:
            ramp_params = fit_ramp_curve(tenure_df, seasonal_mult)
        if ramp_params is None:
            used_fallback = True

    if used_fallback:
        seasonal_mult, _ = estimate_seasonality(df) if last_month_in_data >= 12 else (
            {i: 1.0 for i in range(12)}, df["Sales"].mean()
        )
        start_productivity = fallback_recent_average_productivity(df, sp_cols, tenures)
        seasonal_mult["fallback_start_productivity"] = start_productivity

    scenarios, current_headcount = project_scenarios(
        df, sp_cols, tenures, last_month_in_data, ramp_params, seasonal_mult,
        horizon, salary, commission_rate, margin_rate,
    )

    return {
        "tenures": tenures,
        "last_month_in_data": last_month_in_data,
        "departed_staff": departed_staff,
        "used_fallback": used_fallback,
        "scenarios": scenarios,
        "current_headcount": current_headcount,
    }


st.title("RevInsight Staffing Advisor")
st.caption(
    "Upload monthly sales-by-salesperson history to get a data-driven recommendation "
    "on whether (and when) to hire a replacement salesperson."
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload sales CSV", type="csv")

    st.header("Business parameters")
    salary = st.number_input(
        "Salary per person per month", min_value=0.0, value=3.0, step=0.5,
        help="Same units as the Sales column.",
    )
    commission_rate = st.number_input(
        "Commission rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f",
        help="Fraction of sales, e.g. 0.05 for 5%.",
    )
    margin_rate = st.number_input(
        "Gross margin rate", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f",
        help="Fraction of sales, e.g. 0.50 for 50%.",
    )
    horizon = st.number_input(
        "Forecast horizon (months)", min_value=1, max_value=24, value=6, step=1,
    )

if uploaded_file is None:
    st.info("Upload a CSV to run the analysis. Expected columns: Month, Sales, sale sp1, sale sp2, ...")
    st.stop()

try:
    df, sp_cols = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

result = run_analysis(df, sp_cols, horizon, salary, commission_rate, margin_rate)

tenures = result["tenures"]
last_month_in_data = result["last_month_in_data"]
departed_staff = result["departed_staff"]
used_fallback = result["used_fallback"]
scenarios = result["scenarios"]
current_headcount = result["current_headcount"]

if used_fallback:
    st.warning(
        f"**Lower-confidence result: using the fallback method.** This dataset has only "
        f"{last_month_in_data} months of history, below the {MIN_MONTHS_FOR_MODEL}-month "
        f"threshold (and/or fewer than {MIN_HIRES_FOR_RAMP} historical hires) needed to "
        "reliably fit a productivity ramp-up curve and a 12-month seasonal pattern. This "
        "report instead uses each active salesperson's own recent average sales and a "
        "conservative estimate of new-hire starting productivity. Treat this as "
        "directional, not precise, and revisit once more months of data are available.",
        icon="⚠️",
    )
else:
    st.success(
        f"Sufficient history ({last_month_in_data} months) to fit a productivity "
        "ramp-up curve and 12-month seasonality model.",
        icon="✅",
    )

col1, col2 = st.columns(2)
with col1:
    st.subheader("Current staff")
    active = {c: t for c, t in tenures.items() if t["currently_active"]}
    st.write(f"**{len(active)} active salesperson(s)**")
    for c, t in sorted(active.items(), key=lambda kv: kv[1]["first"]):
        tenure_months = last_month_in_data - t["first"] + 1
        st.write(f"- {c}: employed since month {t['first']} ({tenure_months} months tenure)")

with col2:
    st.subheader("Departure")
    if departed_staff:
        t = tenures[departed_staff]
        st.write(
            f"Most recent departure: **{departed_staff}** "
            f"(employed months {t['first']}-{t['last']}, {t['n_months']} months tenure, "
            f"left {last_month_in_data - t['last']} month(s) ago)"
        )
    else:
        st.write("No departure detected in the data. This evaluates whether to ADD a salesperson.")

st.subheader(f"Projected {horizon}-month gross margin by scenario")

ordered_keys = ["stay_as_is", "hire_now"] + sorted(
    [k for k in scenarios if k.startswith("delay_")],
    key=lambda k: int(k.split("_")[1]),
)
table_df = pd.DataFrame(
    [
        {
            "Scenario": scenario_label(key),
            "Headcount after": scenarios[key]["headcount_after"],
            "Expected Gross Margin": scenarios[key]["total_margin"],
        }
        for key in ordered_keys
    ]
).sort_values("Expected Gross Margin", ascending=False).reset_index(drop=True)

st.dataframe(table_df, use_container_width=True, hide_index=True)

fig = go.Figure()
for key in ordered_keys:
    fig.add_trace(
        go.Scatter(
            x=list(range(1, horizon + 1)),
            y=scenarios[key]["monthly_margin"],
            mode="lines+markers",
            name=scenario_label(key),
        )
    )
fig.update_layout(
    xaxis_title="Month ahead",
    yaxis_title="Projected gross margin",
    legend_title="Scenario",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

best_key = max(scenarios, key=lambda k: scenarios[k]["total_margin"])
best_margin = scenarios[best_key]["total_margin"]
baseline_margin = scenarios["stay_as_is"]["total_margin"]
improvement = best_margin - baseline_margin
pct_improvement = (improvement / abs(baseline_margin) * 100) if baseline_margin != 0 else float("nan")

st.subheader("Recommendation")
if best_key == "stay_as_is":
    st.markdown(
        f"**Do NOT hire a replacement right now.** Based on the projected ramp-up of a new "
        f"hire and the expected seasonal pattern over the next {horizon} months, staying at "
        f"{current_headcount} staff is expected to produce the highest gross margin of the "
        "scenarios tested."
    )
else:
    st.markdown(
        f"**{scenario_label(best_key)}** is expected to produce the highest gross margin "
        f"over the next {horizon} months: approximately **{best_margin:.2f}**, compared to "
        f"{baseline_margin:.2f} if staffing is left unchanged "
        f"(+{improvement:.2f}, about {pct_improvement:.1f}%)."
    )

st.caption(
    "This recommendation reflects this dataset's specific recent seasonal position and "
    "current staff tenure. Re-run whenever staffing changes are being considered."
)
