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
SCENARIO_SHORT_LABELS = {"stay_as_is": "Stay as-is", "hire_now": "Hire now"}

# Fixed categorical colors, one per scenario identity (never reassigned by rank/rerun).
# Order follows the validated dataviz palette's adjacent-pairlist slot sequence.
SCENARIO_COLORS = {
    "stay_as_is": "#2a78d6",       # blue
    "hire_now": "#eb6834",         # orange
    "delay_1_months": "#1baf7a",   # aqua
    "delay_2_months": "#eda100",   # yellow
    "delay_3_months": "#e87ba4",   # magenta
}
HIGHLIGHT_BG = "rgba(12, 163, 12, 0.18)"  # soft wash of the palette's "good" status hue


def scenario_label(key):
    if key in SCENARIO_LABELS:
        return SCENARIO_LABELS[key]
    if key.startswith("delay_"):
        d = key.split("_")[1]
        return f"Delay hiring by {d} month(s)"
    return key


def scenario_short_label(key):
    """Compact label for space-constrained widgets like st.metric."""
    if key in SCENARIO_SHORT_LABELS:
        return SCENARIO_SHORT_LABELS[key]
    if key.startswith("delay_"):
        d = key.split("_")[1]
        return f"Delay {d} mo"
    return scenario_label(key)


@st.cache_resource(show_spinner="Parsing data and fitting model...")
def analyze_file(uploaded_file):
    """
    Everything that depends only on the uploaded data, not on the business
    parameters: CSV parsing, tenure detection, and (if there's enough
    history) fitting the ramp-up curve and seasonality model. Cached so that
    adjusting salary/commission/margin/horizon in the sidebar never re-parses
    the file or re-fits the curve -- only the cheap scenario projection below
    reruns. Mirrors the model-selection logic in staffing_advisor.main().

    Uses cache_resource (cache by reference) rather than cache_data (cache by
    pickled value): fit_ramp_curve() returns a dict holding a local closure
    (its `model` function), which isn't picklable, so cache_data would fail
    on any dataset with enough history to fit the ramp curve.
    """
    df, sp_cols = load_data(uploaded_file)
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

    return {
        "df": df,
        "sp_cols": sp_cols,
        "tenures": tenures,
        "last_month_in_data": last_month_in_data,
        "departed_staff": departed_staff,
        "used_fallback": used_fallback,
        "ramp_params": ramp_params,
        "seasonal_mult": seasonal_mult,
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
    analysis = analyze_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

df = analysis["df"]
sp_cols = analysis["sp_cols"]
tenures = analysis["tenures"]
last_month_in_data = analysis["last_month_in_data"]
departed_staff = analysis["departed_staff"]
used_fallback = analysis["used_fallback"]
ramp_params = analysis["ramp_params"]
seasonal_mult = analysis["seasonal_mult"]

# Cheap: reruns on every parameter change without re-parsing or re-fitting.
scenarios, current_headcount = project_scenarios(
    df, sp_cols, tenures, last_month_in_data, ramp_params, seasonal_mult,
    horizon, salary, commission_rate, margin_rate,
)

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

ordered_keys = ["stay_as_is", "hire_now"] + sorted(
    [k for k in scenarios if k.startswith("delay_")],
    key=lambda k: int(k.split("_")[1]),
)

best_key = max(scenarios, key=lambda k: scenarios[k]["total_margin"])
worst_key = min(scenarios, key=lambda k: scenarios[k]["total_margin"])
best_margin = scenarios[best_key]["total_margin"]
worst_margin = scenarios[worst_key]["total_margin"]
baseline_margin = scenarios["stay_as_is"]["total_margin"]
improvement = best_margin - baseline_margin
pct_improvement = (improvement / abs(baseline_margin) * 100) if baseline_margin != 0 else float("nan")

# --- At-a-glance summary -----------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Recommended action", scenario_short_label(best_key), help=scenario_label(best_key))
m2.metric(
    "Projected gross margin",
    f"{best_margin:,.2f}",
    delta=f"{improvement:+,.2f} vs. staying as-is",
)
m3.metric(
    "Best vs. worst scenario",
    f"{best_margin - worst_margin:,.2f}",
    help=f"Gap between '{scenario_label(best_key)}' and '{scenario_label(worst_key)}', the "
    "widest spread among the scenarios tested.",
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

best_label = scenario_label(best_key)


def highlight_best_row(row):
    style = f"background-color: {HIGHLIGHT_BG}" if row["Scenario"] == best_label else ""
    return [style] * len(row)


st.dataframe(
    table_df.style.apply(highlight_best_row, axis=1).format({"Expected Gross Margin": "{:,.2f}"}),
    use_container_width=True,
    hide_index=True,
)
st.caption(f"Highlighted row: **{best_label}**, the highest-margin scenario tested.")

# --- Chart ---------------------------------------------------------------
fig = go.Figure()
for key in ordered_keys:
    is_best = key == best_key
    color = SCENARIO_COLORS.get(key, "#898781")
    fig.add_trace(
        go.Scatter(
            x=list(range(1, horizon + 1)),
            y=scenarios[key]["monthly_margin"],
            mode="lines+markers",
            name=scenario_label(key) + (" (recommended)" if is_best else ""),
            line=dict(color=color, width=4 if is_best else 1.75),
            marker=dict(color=color, size=9 if is_best else 5),
            opacity=1.0 if is_best else 0.35,
        )
    )
fig.update_layout(
    xaxis_title="Month ahead",
    yaxis_title="Projected gross margin",
    hovermode="x unified",
    height=560,
    font=dict(size=14),
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    margin=dict(t=80),
)
st.plotly_chart(fig, use_container_width=True)

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
