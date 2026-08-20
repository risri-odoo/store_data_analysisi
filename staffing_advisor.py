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
