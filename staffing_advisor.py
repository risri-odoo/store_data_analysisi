#!/usr/bin/env python3
"""
staffing_advisor.py
====================

A reusable tool for retail shop owners to decide whether to replace a
departing salesperson, using their own monthly sales-by-salesperson history.

WHAT IT DOES
------------
1. Reads a CSV with columns: Month, Sales, sale sp1, sale sp2, ... sale spN
   (any number of salesperson columns; a salesperson's column is 0 in months
   they were not employed).
2. Detects each salesperson's tenure (first/last active month) and figures
   out who is currently employed and who has departed.
3. If there is enough history (>= MIN_MONTHS_FOR_MODEL months), it:
     a. Fits a productivity "ramp-up curve" describing how a new hire's
        output grows with tenure, based on every historical hire in the data.
     b. Estimates a 12-month seasonality pattern from total store Sales.
     c. Projects the next 6 months under several staffing scenarios
        (stay as-is / hire immediately / delay hiring by 1-3 months) and
        computes expected gross margin for each, using the owner's formula:
            Gross Margin = margin_rate * Sales - salary_per_person * headcount
                           - commission_rate * Sales
4. If there is NOT enough history, it falls back to a simpler "use recent
   average productivity" method, clearly flagged as lower-confidence, since
   a ramp-up curve and a 12-month seasonal cycle can't be reliably estimated
   from a short history.
5. Prints a plain-language recommendation and a comparison table, and saves
   both to a text file.

WHY IT'S BUILT THIS WAY
------------------------
- Salespeople are not interchangeable: a brand-new hire is typically far
  less productive than a veteran, and only catches up gradually. Comparing
  "headcount of 7" vs "headcount of 8" without accounting for this
  overstates the value of hiring.
- Many retail businesses have real seasonality (festivals, weather, school
  holidays). Whether a new hire clears their own breakeven point in their
  first few months depends heavily on whether those months are in a
  seasonal peak or trough -- not just on headcount.
- Every shop is different, so nothing about salary, commission rate, margin
  rate, or the forecast horizon is hard-coded; they're all parameters.

HOW TO RUN IT
-------------
Basic usage (uses default assumptions: salary=3/person/month, commission=5%,
gross margin rate=50%, 6-month forecast horizon):

    python3 staffing_advisor.py path/to/your_data.csv

With custom business parameters:

    python3 staffing_advisor.py path/to/your_data.csv \
        --salary 25000 --commission-rate 0.04 --margin-rate 0.55 \
        --horizon 6

To analyze a specific departure instead of the most recently detected one
(e.g. you want to ask "what if this other person leaves"):

    python3 staffing_advisor.py path/to/your_data.csv --departed-staff "sale sp9"

To save the report to a specific location:

    python3 staffing_advisor.py path/to/your_data.csv --output my_report.txt

REQUIREMENTS
------------
    pip install pandas numpy scipy

INPUT CSV FORMAT
-----------------
    Month, Sales, sale sp1, sale sp2, ..., sale spN
    1, 23.10, 7.23, 6.34, 9.53, 0, 0, ...
    2, 41.59, 8.90, 7.80, 11.73, 7.52, 5.64, 0, ...
    ...

  - Month: sequential integers starting at 1, one row per month, no gaps.
  - Sales: total store sales for that month (any consistent currency unit).
  - sale spN: that salesperson's individual sales for the month. A value of
    0 (or blank) means that person was not employed that month. Once a
    person's column goes to 0 and stays 0 through the end of the data,
    they are treated as having left the business.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


MIN_MONTHS_FOR_MODEL = 18  # need at least this much history to trust ramp+seasonality fits
MIN_HIRES_FOR_RAMP = 4     # need at least this many historical hires to fit a ramp curve


# ---------------------------------------------------------------------------
# Data loading & structure detection
# ---------------------------------------------------------------------------

def load_data(csv_path):
    """Load the CSV and clean up column names (strip stray whitespace)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Month" not in df.columns or "Sales" not in df.columns:
        raise ValueError(
            "CSV must contain 'Month' and 'Sales' columns. "
            f"Found columns: {list(df.columns)}"
        )

    sp_cols = [c for c in df.columns if c.strip().lower().startswith("sale sp")]
    if not sp_cols:
        raise ValueError(
            "No salesperson columns found. Expected columns named like "
            "'sale sp1', 'sale sp2', etc."
        )

    df = df.sort_values("Month").reset_index(drop=True)
    df[sp_cols] = df[sp_cols].fillna(0)
    return df, sp_cols


def get_staff_tenures(df, sp_cols):
    """
    For each salesperson column, find their first and last active month
    (sales > 0). Returns a dict: {col_name: {"first": m, "last": m, "n_months": n}}
    A person is considered "currently active" if their last active month
    is the final month in the dataset.
    """
    last_month_in_data = df["Month"].max()
    tenures = {}
    for c in sp_cols:
        active = df[df[c] > 0]
        if len(active) == 0:
            continue
        first = int(active["Month"].min())
        last = int(active["Month"].max())
        tenures[c] = {
            "first": first,
            "last": last,
            "n_months": len(active),
            "currently_active": (last == last_month_in_data),
        }
    return tenures, last_month_in_data


def detect_most_recent_departure(tenures, last_month_in_data):
    """
    Find the salesperson whose last active month is the most recent among
    everyone who is NOT currently active. Returns None if nobody has left,
    or if everyone currently active never had a predecessor leave.
    """
    departed = {c: t for c, t in tenures.items() if not t["currently_active"]}
    if not departed:
        return None
    # most recent departure = max "last" among departed staff
    most_recent = max(departed.items(), key=lambda kv: kv[1]["last"])
    return most_recent[0]  # column name


# ---------------------------------------------------------------------------
# Ramp-up curve fitting
# ---------------------------------------------------------------------------

def build_tenure_table(df, sp_cols):
    """
    Build a long-format table of (salesperson, tenure_month, sales,
    calendar_month) for every active month of every salesperson. tenure_month
    is 1 for a person's first active month, 2 for their second, etc.
    """
    records = []
    for c in sp_cols:
        s = df[c]
        active_idx = s[s > 0].index
        if len(active_idx) == 0:
            continue
        vals = s.loc[active_idx].values
        cal_months = df.loc[active_idx, "Month"].values
        for t, (v, cm) in enumerate(zip(vals, cal_months), start=1):
            records.append(
                {"salesperson": c, "tenure_month": t, "sales": v, "calendar_month": cm}
            )
    return pd.DataFrame(records)


def estimate_seasonality(df, cycle_length=12):
    """
    Estimate a multiplicative seasonal index by position in a 12-month
    cycle, based on total store Sales. Returns a dict {0..11: multiplier}
    where multiplier=1.0 means "average month".
    """
    tmp = df[["Month", "Sales"]].copy()
    tmp["cyc"] = (tmp["Month"] - 1) % cycle_length
    seasonal_avg = tmp.groupby("cyc")["Sales"].mean()
    overall_mean = tmp["Sales"].mean()
    seasonal_mult = (seasonal_avg / overall_mean).to_dict()
    # fill any missing cycle positions (possible with very short history) with 1.0
    for i in range(cycle_length):
        seasonal_mult.setdefault(i, 1.0)
    return seasonal_mult, overall_mean


def fit_ramp_curve(tenure_df, seasonal_mult, cycle_length=12):
    """
    Fit a saturating ramp-up curve: deseasonalized_sales(tenure_month) =
        B + A * (1 - exp(-k * tenure_month))
    where B = approximate starting productivity, A+B = approximate
    long-run plateau productivity, k = ramp speed.

    Deseasonalizes each observation first so that the curve reflects
    tenure-driven growth, not calendar-driven seasonal swings.
    """
    df = tenure_df.copy()
    df["cyc"] = (df["calendar_month"] - 1) % cycle_length
    df["seasonal_mult"] = df["cyc"].map(seasonal_mult)
    df["sales_deseasonalized"] = df["sales"] / df["seasonal_mult"].replace(0, np.nan)
    df = df.dropna(subset=["sales_deseasonalized"])

    def model(t, A, k, B):
        return B + A * (1 - np.exp(-k * t))

    x = df["tenure_month"].values.astype(float)
    y = df["sales_deseasonalized"].values.astype(float)

    # reasonable starting guesses: B ~ low end of observed sales, A ~ spread, k small
    p0 = [max(y.max() - y.min(), 1.0), 0.05, max(y.min(), 0.1)]
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=20000)
        A, k, B = popt
        # guard against degenerate fits (negative ramp speed, absurd plateau)
        if k <= 0 or A < 0:
            return None
        return {"A": A, "k": k, "B": B, "model": model}
    except Exception:
        return None


def ramp_value(ramp_params, tenure_month):
    """Predicted deseasonalized productivity at a given tenure month."""
    return ramp_params["model"](tenure_month, ramp_params["A"], ramp_params["k"], ramp_params["B"])


# ---------------------------------------------------------------------------
# Margin model and scenario simulation
# ---------------------------------------------------------------------------

def margin_from_sales(total_sales, headcount, salary, commission_rate, margin_rate):
    gross = margin_rate * total_sales
    salaries = salary * headcount
    commission = commission_rate * total_sales
    return gross - salaries - commission


def project_scenarios(
    df,
    sp_cols,
    tenures,
    last_month_in_data,
    ramp_params,
    seasonal_mult,
    horizon,
    salary,
    commission_rate,
    margin_rate,
    cycle_length=12,
):
    """
    Runs the core comparison:
      - "stay": keep current active staff, no new hire, over the horizon
      - "hire_now": add one new hire starting next month
      - "delay_D": add one new hire starting D months from now, for D in 1..min(3, horizon-1)

    For each scenario, projects month-by-month sales of existing staff
    using the ramp curve at their *future* tenure, applies seasonal
    multipliers for each future calendar month, sums to a total monthly
    sales figure, and computes margin.

    Returns a dict of scenario_name -> {"total_margin": x, "monthly": [...]}
    """
    active_staff = {c: t for c, t in tenures.items() if t["currently_active"]}
    current_headcount = len(active_staff)

    future_months = [last_month_in_data + i for i in range(1, horizon + 1)]
    future_cyc = [(m - 1) % cycle_length for m in future_months]
    future_seasonal_factors = [seasonal_mult.get(c, 1.0) for c in future_cyc]

    def existing_staff_sales(month_offset, seasonal_factor):
        """Projected total sales from current staff at a future month offset (1..horizon)."""
        total = 0.0
        for col, t in active_staff.items():
            tenure_so_far = last_month_in_data - t["first"] + 1
            future_tenure = tenure_so_far + month_offset
            if ramp_params is not None:
                total += ramp_value(ramp_params, future_tenure) * seasonal_factor
            else:
                # fallback: use this person's own recent average sales, scaled by
                # seasonal factor relative to the average seasonal factor (~1.0)
                total += t["recent_avg_sales"] * seasonal_factor
        return total

    scenarios = {}

    # Scenario: stay as-is
    monthly = []
    for off in range(1, horizon + 1):
        f = future_seasonal_factors[off - 1]
        s = existing_staff_sales(off, f)
        monthly.append(s)
    margins = [
        margin_from_sales(s, current_headcount, salary, commission_rate, margin_rate)
        for s in monthly
    ]
    scenarios["stay_as_is"] = {
        "headcount_after": current_headcount,
        "monthly_sales": monthly,
        "monthly_margin": margins,
        "total_margin": sum(margins),
    }

    # Scenario: hire immediately, and delayed hiring by D months
    max_delay = min(3, horizon - 1) if horizon > 1 else 0
    for D in [0] + list(range(1, max_delay + 1)):
        monthly = []
        for off in range(1, horizon + 1):
            f = future_seasonal_factors[off - 1]
            s = existing_staff_sales(off, f)
            if off > D:
                new_hire_tenure = off - D
                if ramp_params is not None:
                    s += ramp_value(ramp_params, new_hire_tenure) * f
                else:
                    # fallback: assume new hire starts at the lowest historical
                    # starting productivity observed, scaled by season
                    s += seasonal_mult.get("fallback_start_productivity", 0) * f
                headcount = current_headcount + 1
            else:
                headcount = current_headcount
            monthly.append((s, headcount))
        margins = [
            margin_from_sales(s, hc, salary, commission_rate, margin_rate) for s, hc in monthly
        ]
        key = "hire_now" if D == 0 else f"delay_{D}_months"
        scenarios[key] = {
            "headcount_after": current_headcount + 1,
            "monthly_sales": [m[0] for m in monthly],
            "monthly_margin": margins,
            "total_margin": sum(margins),
        }

    return scenarios, current_headcount


# ---------------------------------------------------------------------------
# Fallback path for short histories
# ---------------------------------------------------------------------------

def fallback_recent_average_productivity(df, sp_cols, tenures, lookback=3):
    """
    For shops with too little history to fit a ramp curve reliably, estimate
    each active salesperson's "recent average sales" using their last
    `lookback` active months, and estimate a typical new-hire starting
    productivity as the minimum first-month sales seen among any past hires
    (or, if no past hires exist at all, the minimum of any active person's
    earliest observed sales).
    """
    for c, t in tenures.items():
        s = df[df[c] > 0][c]
        recent = s.tail(lookback)
        t["recent_avg_sales"] = recent.mean() if len(recent) > 0 else s.mean()

    first_month_values = []
    for c in sp_cols:
        s = df[df[c] > 0][c]
        if len(s) > 0:
            first_month_values.append(s.iloc[0])
    start_productivity = min(first_month_values) if first_month_values else 0.0
    return start_productivity


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_report(
    csv_path,
    df,
    sp_cols,
    tenures,
    last_month_in_data,
    departed_staff,
    used_fallback,
    ramp_params,
    seasonal_mult,
    scenarios,
    current_headcount,
    horizon,
    salary,
    commission_rate,
    margin_rate,
):
    lines = []
    add = lines.append

    add("=" * 78)
    add("STAFFING RECOMMENDATION REPORT")
    add("=" * 78)
    add(f"Input file: {csv_path}")
    add(f"Months of history: {last_month_in_data}")
    add(f"Forecast horizon: {horizon} months")
    add(
        f"Business parameters: salary={salary}/person/month, "
        f"commission_rate={commission_rate:.0%}, margin_rate={margin_rate:.0%}"
    )
    add("")

    active = {c: t for c, t in tenures.items() if t["currently_active"]}
    add(f"Current active staff: {len(active)}")
    for c, t in sorted(active.items(), key=lambda kv: kv[1]["first"]):
        tenure_months = last_month_in_data - t["first"] + 1
        add(f"  - {c}: employed since month {t['first']} ({tenure_months} months tenure)")
    add("")

    if departed_staff:
        t = tenures[departed_staff]
        add(
            f"Most recent departure: {departed_staff} "
            f"(employed months {t['first']}-{t['last']}, {t['n_months']} months tenure, "
            f"left {last_month_in_data - t['last']} month(s) ago)"
        )
    else:
        add("No departure detected in the data (or none specified). "
            "This report evaluates whether to ADD a salesperson beyond current headcount.")
    add("")

    if used_fallback:
        add("-" * 78)
        add("NOTE ON DATA SUFFICIENCY")
        add("-" * 78)
        add(
            f"This dataset has only {last_month_in_data} months of history, which is below "
            f"the {MIN_MONTHS_FOR_MODEL}-month threshold this tool uses to trust a fitted "
            "ramp-up curve and a 12-month seasonal pattern. Both require enough repeated "
            "observations (multiple full hire-to-veteran journeys, multiple full yearly "
            "cycles) to estimate reliably; with limited data they would likely overfit "
            "noise rather than capture real patterns."
        )
        add(
            "This report instead uses each active salesperson's own recent average sales, "
            "and a conservative estimate of new-hire starting productivity based on the "
            "weakest first-month performance seen in the data. Treat this recommendation "
            "as directional rather than precise, and revisit it once more months of data "
            "are available."
        )
        add("")

    add("-" * 78)
    add(f"PROJECTED {horizon}-MONTH GROSS MARGIN BY SCENARIO")
    add("-" * 78)

    name_map = {"stay_as_is": "Remain at current staffing level"}
    for key in scenarios:
        if key.startswith("delay_"):
            d = key.split("_")[1]
            name_map[key] = f"Delay hiring by {d} month(s)"
    name_map["hire_now"] = "Hire replacement immediately"

    ordered_keys = ["stay_as_is", "hire_now"] + sorted(
        [k for k in scenarios if k.startswith("delay_")],
        key=lambda k: int(k.split("_")[1]),
    )

    col_w = 38
    add(f"{'Scenario':<{col_w}}{'Expected Gross Margin':>20}")
    for key in ordered_keys:
        if key not in scenarios:
            continue
        label = name_map.get(key, key)
        margin = scenarios[key]["total_margin"]
        add(f"{label:<{col_w}}{margin:>20.2f}")
    add("")

    best_key = max(scenarios, key=lambda k: scenarios[k]["total_margin"])
    best_margin = scenarios[best_key]["total_margin"]
    baseline_margin = scenarios["stay_as_is"]["total_margin"]
    improvement = best_margin - baseline_margin
    pct_improvement = (improvement / abs(baseline_margin) * 100) if baseline_margin != 0 else float("nan")

    add("-" * 78)
    add("RECOMMENDATION")
    add("-" * 78)
    if best_key == "stay_as_is":
        add(
            "Do NOT hire a replacement right now. Based on the projected ramp-up of a new "
            "hire and the expected seasonal pattern over the next "
            f"{horizon} months, staying at {current_headcount} staff is expected to produce "
            "the highest gross margin of the scenarios tested."
        )
    else:
        add(
            f"{name_map.get(best_key, best_key)} is expected to produce the highest gross "
            f"margin over the next {horizon} months: approximately {best_margin:.2f}, "
            f"compared to {baseline_margin:.2f} if staffing is left unchanged "
            f"(+{improvement:.2f}, about {pct_improvement:.1f}%)."
        )
    add("")
    add(
        "This recommendation reflects this dataset's specific recent seasonal position and "
        "current staff tenure. It is not a general rule -- re-run this analysis whenever "
        "staffing changes are being considered, since the right timing depends on where the "
        "next few months fall in the seasonal cycle and how experienced the remaining staff "
        "currently are."
    )
    add("")
    add("=" * 78)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze monthly sales-by-salesperson data and recommend "
        "whether to hire a replacement salesperson, and when."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "--salary", type=float, default=3.0,
        help="Salary cost per salesperson per month, in the same units as the Sales "
        "column (default: 3.0, matching the example dataset).",
    )
    parser.add_argument(
        "--commission-rate", type=float, default=0.05,
        help="Commission as a fraction of sales, e.g. 0.05 for 5%% (default: 0.05).",
    )
    parser.add_argument(
        "--margin-rate", type=float, default=0.50,
        help="Gross margin as a fraction of sales, e.g. 0.50 for 50%% (default: 0.50).",
    )
    parser.add_argument(
        "--horizon", type=int, default=6,
        help="Number of months ahead to forecast (default: 6).",
    )
    parser.add_argument(
        "--departed-staff", type=str, default=None,
        help="Column name of a specific salesperson to evaluate as 'departed' "
        "(e.g. 'sale sp9'), overriding auto-detection of the most recent departure.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the text report (default: <input_filename>_report.txt).",
    )
    args = parser.parse_args()

    try:
        df, sp_cols = load_data(args.csv_path)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    tenures, last_month_in_data = get_staff_tenures(df, sp_cols)

    if args.departed_staff:
        if args.departed_staff not in tenures:
            print(
                f"Error: '{args.departed_staff}' not found among detected salesperson "
                f"columns: {list(tenures.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        departed_staff = args.departed_staff
    else:
        departed_staff = detect_most_recent_departure(tenures, last_month_in_data)

    used_fallback = last_month_in_data < MIN_MONTHS_FOR_MODEL

    ramp_params = None
    if not used_fallback:
        tenure_df = build_tenure_table(df, sp_cols)
        n_hires = tenure_df["salesperson"].nunique()
        seasonal_mult, _ = estimate_seasonality(df)
        if n_hires >= MIN_HIRES_FOR_RAMP:
            ramp_params = fit_ramp_curve(tenure_df, seasonal_mult)
        if ramp_params is None:
            # not enough distinct hires to fit a ramp curve reliably, or fit failed;
            # fall back even though there's plenty of *time* history
            used_fallback = True

    if used_fallback:
        seasonal_mult, _ = estimate_seasonality(df) if last_month_in_data >= 12 else (
            {i: 1.0 for i in range(12)}, df["Sales"].mean()
        )
        start_productivity = fallback_recent_average_productivity(df, sp_cols, tenures)
        seasonal_mult["fallback_start_productivity"] = start_productivity

    scenarios, current_headcount = project_scenarios(
        df, sp_cols, tenures, last_month_in_data, ramp_params, seasonal_mult,
        args.horizon, args.salary, args.commission_rate, args.margin_rate,
    )

    report = format_report(
        args.csv_path, df, sp_cols, tenures, last_month_in_data, departed_staff,
        used_fallback, ramp_params, seasonal_mult, scenarios, current_headcount,
        args.horizon, args.salary, args.commission_rate, args.margin_rate,
    )

    print(report)

    output_path = args.output
    if output_path is None:
        import os
        base = os.path.basename(args.csv_path).rsplit(".", 1)[0]
        output_path = f"{base}_report.txt"
    try:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\n(Report saved to {output_path})")
    except OSError as e:
        print(
            f"\n(Could not save report to {output_path}: {e}. "
            "Use --output to specify a writable location.)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
