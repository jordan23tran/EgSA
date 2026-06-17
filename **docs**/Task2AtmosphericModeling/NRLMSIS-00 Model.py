"""
run_msis_analog_predictor.py
============================
Builds (or reloads) the NRLMSIS-00 kNN analog predictor.

Workflow
--------
  1. If a saved predictor already exists → load it, jump to query.
  2. Otherwise:
       a. Load the NRLMSIS-00 daily-average output CSV produced by
          your existing pymsis script (output_daily.csv).
       b. Build the feature DataFrame from the raw space-weather CSV.
       c. Fit the predictor, save it to disk, then launch the query.

Expected daily CSV columns  (from your pymsis aggregation step)
---------------------------------------------------------------
  Year, Month, Day, lat, lon, alt_km,
  F107, F107A, Ap_daily,
  Density_mean_kg_m3, Atomic_O_mean_m3

pymsis output-index note
------------------------
  pymsis.calculate() returns a (..., 11) array where:
    index 0  →  total mass density  [kg/m³]  ← Density_kg_m3
    index 1  →  N₂ number density   [m⁻³]
    index 3  →  O  number density   [m⁻³]   ← correct Atomic O
  If your CSV was produced with msis_output[:, 1] labelled as
  'Atomic_O_density_m3', it actually contains N₂, not O.
  To fix, re-run the pymsis script with msis_output[:, 3] for atomic O
  and regenerate output_daily.csv before rebuilding the predictor.
"""

import os
import numpy as np
import pandas as pd

from msis_dtm_analog_predictor import (
    build_feature_df_msis,
    AtmosphericAnalogPredictor,
    terminal_query,
)

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
SW_CSV    = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26\space_weather_daily_may_1996_2026.csv"
)
DAILY_CSV = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26\output_daily.csv"
)
OUTPUT_DIR     = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26"
)
PREDICTOR_PATH = os.path.join(OUTPUT_DIR, "msis_analog_predictor.pkl")
ALTITUDES_KM   = [200, 225, 250, 275, 300]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD EXISTING PREDICTOR IF AVAILABLE ──────────────────────────────────────
if os.path.exists(PREDICTOR_PATH):
    print(f"Loading existing predictor:\n  {PREDICTOR_PATH}")
    predictor = AtmosphericAnalogPredictor.load(PREDICTOR_PATH)
    terminal_query(predictor, output_dir=OUTPUT_DIR)
    raise SystemExit

# ── LOAD NRLMSIS-00 DAILY OUTPUT ──────────────────────────────────────────────
print(f"Loading NRLMSIS-00 daily output:\n  {DAILY_CSV}")
daily = pd.read_csv(DAILY_CSV)
daily["Date"] = pd.to_datetime(daily[["Year", "Month", "Day"]])
daily = daily.sort_values(["Date", "alt_km"]).reset_index(drop=True)

dates_all = pd.DatetimeIndex(sorted(daily["Date"].unique()))
print(f"  {len(dates_all)} unique dates  |  altitudes: {ALTITUDES_KM} km")

# ── PIVOT TO (n_times, n_alts) ARRAYS ─────────────────────────────────────────
#  pivot_table handles any accidental duplicate (Date, alt_km) rows by averaging
density_pivot  = daily.pivot_table(
    index="Date", columns="alt_km",
    values="Density_mean_kg_m3", aggfunc="mean"
)
atomic_O_pivot = daily.pivot_table(
    index="Date", columns="alt_km",
    values="Atomic_O_mean_m3", aggfunc="mean"
)

density_pivot  = density_pivot.reindex(index=dates_all,  columns=ALTITUDES_KM)
atomic_O_pivot = atomic_O_pivot.reindex(index=dates_all, columns=ALTITUDES_KM)

density_array  = density_pivot.values.astype(float)
atomic_O_array = atomic_O_pivot.values.astype(float)

# Sanity check – total density at 200 km should be ~1e-10 to 1e-9 kg/m³
med_rho_200 = np.nanmedian(density_array[:, ALTITUDES_KM.index(200)])
if not (1e-13 < med_rho_200 < 1e-7):
    print(f"  WARNING: median density at 200 km = {med_rho_200:.3e} kg/m³ "
          "– expected ~1e-10 kg/m³. Verify the Density_mean_kg_m3 column units.")

n_nan_rho = np.isnan(density_array).sum()
n_nan_O   = np.isnan(atomic_O_array).sum()
if n_nan_rho or n_nan_O:
    print(f"  WARNING: {n_nan_rho} NaN density values, "
          f"{n_nan_O} NaN atomic-O values after pivot.")

# ── BUILD FEATURE DATAFRAME ───────────────────────────────────────────────────
print("\nBuilding feature DataFrame from space-weather CSV...")
feature_df = build_feature_df_msis(
    dates       = dates_all,
    sw_csv_path = SW_CSV,
    f107_col    = "F107_adj",
    ap_col      = "Ap_daily",
    include_doy = True,
)
print(f"  Feature columns : {list(feature_df.columns)}")
print(f"  Shape           : {feature_df.shape}")

# ── FIT PREDICTOR ─────────────────────────────────────────────────────────────
predictor = AtmosphericAnalogPredictor(model_name="NRLMSIS-00")
predictor.fit(
    dates          = dates_all,
    feature_df     = feature_df,
    density_array  = density_array,
    atomic_O_array = atomic_O_array,
    altitudes_km   = ALTITUDES_KM,
)

# ── SAVE TO DISK ──────────────────────────────────────────────────────────────
predictor.save(PREDICTOR_PATH)
print(f"Predictor saved to:\n  {PREDICTOR_PATH}")

# ── LAUNCH INTERACTIVE QUERY ──────────────────────────────────────────────────
terminal_query(predictor, output_dir=OUTPUT_DIR)
