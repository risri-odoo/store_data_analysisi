# RevInsight Take-Home — Submission

## Contents

- `RevInsight_Part1_Recommendation.pdf` — Part 1: written recommendation and
  step-by-step analysis for the specific dataset provided (RevI-Test.csv).
- `staffing_advisor.py` — Part 2: a reusable Python script that runs the same
  type of analysis on any future CSV with the same column format.
- `sample_output.txt` — example console output from running the script
  against the provided RevI-Test.csv, included so the recommendation can be
  verified without re-running anything.

## Running the script

Requirements:

    pip install pandas numpy scipy

Basic usage:

    python3 staffing_advisor.py path/to/your_data.csv

This uses the same default business assumptions as Part 1 (salary = 3 per
person per month, 5% commission, 50% gross margin rate, 6-month forecast
horizon). All of these can be overridden, e.g.:

    python3 staffing_advisor.py path/to/your_data.csv \
        --salary 25000 --commission-rate 0.04 --margin-rate 0.55 --horizon 6

Full usage details, including how to evaluate a specific departed employee
(`--departed-staff`) and where the report gets saved (`--output`), are
documented in the script's own header — run `python3 staffing_advisor.py -h`
or open the file directly.

## What the script does, briefly

1. Reads the CSV and detects each salesperson's employment tenure from when
   their column is nonzero.
2. Auto-detects the most recently departed employee (or uses one specified
   via `--departed-staff`).
3. If there's enough history (18+ months, 4+ historical hires), fits:
   - a productivity ramp-up curve showing how new hires' output grows with
     tenure, and
   - a 12-month seasonality pattern from total store sales.
   It then projects expected gross margin over the next N months under five
   scenarios: stay as-is, hire immediately, or delay hiring by 1-3 months.
4. If there's not enough history to fit those reliably, it falls back to a
   simpler method (recent average productivity per employee) and says so
   explicitly in the report, so the recommendation's confidence level is
   never overstated.
5. Prints a plain-language recommendation with the full scenario comparison
   table, and saves it to a text file.

This mirrors the logic used in the Part 1 write-up: a departing veteran is
not a like-for-like loss, a new hire is not a like-for-like replacement, and
whether hiring now is the right call depends on how the next several months
line up with the seasonal cycle — not on a fixed headcount target.
