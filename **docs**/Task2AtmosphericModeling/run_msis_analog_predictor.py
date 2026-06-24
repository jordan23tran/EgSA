"""
run_msis_analog_predictor.py
============================
Purpose
-------
Builds (or loads from cache) the NRLMSIS-00 kNN analog predictor, then
launches an interactive terminal query session where you can:
  • Enter solar/geomagnetic conditions (F10.7, Ap, day-of-year)
  • Get predicted atmospheric density, temperature, and species composition
  • Copy-paste the result directly into a SPARTA atmosphere input block

Dependency chain (run in this order)
-------------------------------------
  1. gfz_cleaner.py            → space_weather_daily_may_1996_2026.csv
  2. nrlmsis_runner.py         → output_daily.csv   ← this script reads it
  3. THIS SCRIPT               → msis_analog_predictor.pkl
  4. (use pkl for queries)

Auto-rebuild logic
------------------
  • If msis_analog_predictor.pkl does not exist → build from output_daily.csv.
  • If the pkl exists but was created with an older schema (schema v1: density
    + atomic-O only) → delete it and rebuild automatically.
  • If output_daily.csv is in old format (missing O_mean_m3 column) → exit
    with instructions to re-run nrlmsis_runner.py first.
  • If the pkl is up-to-date → skip rebuilding and go straight to the query.

Inputs
------
  output_daily.csv   (from nrlmsis_runner.py)
    Required columns: Year, Month, Day, F107, F107A, Ap_daily, alt_km,
                      Density_mean_kg_m3, T_mean_K,
                      O_mean_m3, N2_mean_m3, O2_mean_m3, He_mean_m3,
                      Ar_mean_m3, H_mean_m3, N_mean_m3

Outputs
-------
  msis_analog_predictor.pkl       (cached predictor — loaded next time)
  sparta_block_NRLMSIS_00_<alt>km.sparta    (if generated in session)
  analogs_NRLMSIS_00_<alt>km.csv            (if saved in session)
  analog_bars_NRLMSIS_00_<alt>km.png        (if plotted in session)

Environment
-----------
  Windows 10/11, Python 3.10+
  Required packages:  numpy pandas matplotlib
      pip install numpy pandas matplotlib

  This script does NOT need pymsis at runtime; pymsis is only needed when
  re-running nrlmsis_runner.py.

How to run
----------
  1. Confirm all four paths in USER SETTINGS below.
  2. Open Command Prompt (Win+R → cmd) and run:
       python run_msis_analog_predictor.py
  3. First run will build the predictor (takes a few seconds).
     Subsequent runs load the pkl instantly.
  4. Follow terminal prompts: enter solar indices, altitude, k, method.
  5. Choose whether to generate a SPARTA block when prompted.

Query features
--------------
  F107      Daily F10.7 solar radio flux [sfu]   (e.g. 150)
  F107A     81-day mean F10.7 [sfu]              (e.g. 145)
  Ap_daily  Daily geomagnetic Ap index           (e.g. 10)
  doy_sin   sin(2π × DOY / 365.25)  — auto-filled if you enter a date
  doy_cos   cos(2π × DOY / 365.25)  — auto-filled if you enter a date
"""

import os
import sys
import numpy as np
import pandas as pd

# ── import the shared predictor library ────────────────────────────────────────
# msis_dtm_analog_predictor.py must be in the same directory as this script.
from msis_dtm_analog_predictor import (
    AtmosphericAnalogPredictor,
    build_feature_df_msis,
    terminal_query,
    PKL_SCHEMA_VERSION,
)

# ============================================================
# USER SETTINGS  ← edit these paths before running
# ============================================================

DAILY_CSV     = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26\output_daily.csv"
)

PREDICTOR_PKL = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26\msis_analog_predictor.pkl"
)

OUTPUT_DIR    = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26"
)

# ============================================================
# LOAD DAILY CSV
# ============================================================

print(f'Loading daily CSV: {DAILY_CSV}')
if not os.path.exists(DAILY_CSV):
    sys.exit(
        f'ERROR: {DAILY_CSV} not found.\n'
        'Run nrlmsis_runner.py first to generate it.'
    )

daily = pd.read_csv(DAILY_CSV)

# ── Format version check ────────────────────────────────────────────────────
# The new nrlmsis_runner.py (v2) adds O_mean_m3 and other species columns.
# Old files have only Density_mean_kg_m3 and no species columns.
if 'O_mean_m3' not in daily.columns:
    sys.exit(
        'ERROR: output_daily.csv is in old format (missing O_mean_m3 column).\n'
        'Re-run nrlmsis_runner.py to regenerate it with all species + temperature.\n'
        'Then re-run this script.'
    )

print(f'  {len(daily):,} rows  |  columns: {list(daily.columns)}')

# ── Build daily date index ──────────────────────────────────────────────────
daily['Date'] = pd.to_datetime(daily[['Year', 'Month', 'Day']])
ALTITUDES_KM  = sorted(daily['alt_km'].unique())
print(f'  Altitudes found: {ALTITUDES_KM} km')

# ============================================================
# AUTO-DETECT & LOAD OR REBUILD THE PREDICTOR PKL
# ============================================================

predictor = None

if os.path.exists(PREDICTOR_PKL):
    try:
        predictor = AtmosphericAnalogPredictor.load(PREDICTOR_PKL)
    except (ValueError, AttributeError, Exception) as e:
        print(f'\n[AUTO-REBUILD] Stale or incompatible pkl detected:\n  {e}')
        print('Deleting old pkl and rebuilding from output_daily.csv …\n')
        os.remove(PREDICTOR_PKL)
        predictor = None

if predictor is None:
    print('Building NRLMSIS-00 analog predictor from daily CSV …')

    # ── Pivot each species/output to (n_days × n_alts) ndarray ─────────────
    # Why pivot? The predictor's species_db stores one 2-D array per output:
    # rows = historical days, columns = altitudes.  A query at a new altitude
    # interpolates between columns; a query on a known altitude uses the exact
    # column.  This structure allows a single fit() call to serve all altitudes.
    col_map = {
        'Density_mean_kg_m3': 'total_density',
        'T_mean_K'          : 'temperature',
        'O_mean_m3'         : 'O',
        'N2_mean_m3'        : 'N2',
        'O2_mean_m3'        : 'O2',
        'He_mean_m3'        : 'He',
        'Ar_mean_m3'        : 'Ar',
        'H_mean_m3'         : 'H',
        'N_mean_m3'         : 'N',
    }

    # Reference pivot to get aligned date index
    ref_pivot = (
        daily.pivot_table(index='Date', columns='alt_km', values='Density_mean_kg_m3')
        .reindex(columns=ALTITUDES_KM)
        .sort_index()
        .dropna()
    )
    dates = pd.DatetimeIndex(ref_pivot.index)

    species_dict = {}
    for csv_col, sp_name in col_map.items():
        pivot = (
            daily.pivot_table(index='Date', columns='alt_km', values=csv_col)
            .reindex(columns=ALTITUDES_KM)
            .sort_index()
            .reindex(ref_pivot.index)
        )
        species_dict[sp_name] = pivot.values   # (n_days, n_alts)

    # ── Space-weather feature DataFrame ────────────────────────────────────
    # One row per unique date (SW indices are altitude-independent)
    sw_daily = (
        daily.drop_duplicates(subset=['Year', 'Month', 'Day'])
        [['Date', 'F107', 'F107A', 'Ap_daily']]
        .set_index('Date')
        .sort_index()
        .reindex(ref_pivot.index)
    )
    feature_df = build_feature_df_msis(sw_daily)

    # ── Fit and save ────────────────────────────────────────────────────────
    p = AtmosphericAnalogPredictor(model_name='NRLMSIS-00')
    p.fit(dates, feature_df, species_dict, ALTITUDES_KM)
    p.save(PREDICTOR_PKL)
    predictor = p
    print(f'\nPredictor saved → {PREDICTOR_PKL}')

# ============================================================
# LAUNCH INTERACTIVE TERMINAL QUERY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
terminal_query(predictor, output_dir=OUTPUT_DIR)
