"""
run_dtm_analog_predictor.py
============================
Builds (or reloads) the DTM-2020 (SWAMI MCM) kNN analog predictor.

Workflow
--------
  1. If a saved predictor already exists → load it, jump to query.
  2. Otherwise:
       a. Load the MCM combined results CSV (mcm_results_all_altitudes.csv).
       b. Detect and convert the total density column.
       c. Build the feature DataFrame from the raw space-weather CSV.
       d. Fit the predictor, save, then launch the query.

Expected MCM results CSV columns
---------------------------------
  Date, Altitude_km, F107, F107m, kp1, kp2, Ap_daily, SunspotNumber,
  O_number_density_m3,  <density_col>  (e.g. 'rho')
  (plus any additional SWAMI output fields: d_O, t_exo, t, …)

Density column
--------------
  DENSITY_COL  : set to None for auto-detection from candidate names,
                 or override explicitly (e.g. 'rho').
  DENSITY_UNIT : 'g_cm3' → multiply by 1000 to convert to kg/m³.
                            SWAMI MCM typically outputs rho in g/cm³.
                 'kg_m3' → no conversion needed.
  Verify by checking that the median density at 200 km ≈ 1e-10 kg/m³.
"""

import os
import numpy as np
import pandas as pd

from msis_dtm_analog_predictor import (
    build_feature_df_dtm,
    AtmosphericAnalogPredictor,
    terminal_query,
)

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
SW_CSV    = (
    "/mnt/c/Users/jorda/OneDrive - Massachusetts Institute of Technology"
    "/Documents/Summer 26/space_weather_daily_may_1996_2026.csv"
)
MCM_CSV   = "/home/jordan23tran/mcm/output/mcm_results_all_altitudes.csv"
OUTPUT_DIR     = "/home/jordan23tran/mcm/output"
PREDICTOR_PATH = os.path.join(OUTPUT_DIR, "dtm2020_analog_predictor.pkl")
ALTITUDES_KM   = [200, 225, 250, 275, 300]

# Density column settings
DENSITY_COL  = None        # None → auto-detect; or set explicitly e.g. "rho"
DENSITY_UNIT = "g_cm3"     # "g_cm3" converts ×1000 → kg/m³  |  "kg_m3" = no-op

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD EXISTING PREDICTOR IF AVAILABLE ──────────────────────────────────────
if os.path.exists(PREDICTOR_PATH):
    print(f"Loading existing predictor:\n  {PREDICTOR_PATH}")
    predictor = AtmosphericAnalogPredictor.load(PREDICTOR_PATH)
    terminal_query(predictor, output_dir=OUTPUT_DIR)
    raise SystemExit

# ── LOAD MCM OUTPUT ───────────────────────────────────────────────────────────
print(f"Loading DTM-2020 (MCM) results:\n  {MCM_CSV}")
results = pd.read_csv(MCM_CSV)
results["Date"] = pd.to_datetime(results["Date"]).dt.normalize()
results = results.sort_values(["Date", "Altitude_km"]).reset_index(drop=True)

print(f"  Columns found: {results.columns.tolist()}")

# ── AUTO-DETECT DENSITY COLUMN ────────────────────────────────────────────────
if DENSITY_COL is None:
    _candidates = ["rho", "density", "Density", "mass_density", "rho_total"]
    DENSITY_COL = next((c for c in _candidates if c in results.columns), None)
    if DENSITY_COL is None:
        raise ValueError(
            "Could not auto-detect a density column in the MCM results CSV.\n"
            f"Available columns: {results.columns.tolist()}\n"
            "Set DENSITY_COL explicitly at the top of this script."
        )
print(f"  Using density column : '{DENSITY_COL}'")
print(f"  Density unit setting : '{DENSITY_UNIT}'")

# ── UNIT CONVERSION ───────────────────────────────────────────────────────────
if DENSITY_UNIT == "g_cm3":
    results[DENSITY_COL] = results[DENSITY_COL] * 1e3    # g/cm³ → kg/m³
    print("  Applied unit conversion: g/cm³ × 1000 → kg/m³")
elif DENSITY_UNIT != "kg_m3":
    raise ValueError(
        f"Unknown DENSITY_UNIT '{DENSITY_UNIT}'. Use 'g_cm3' or 'kg_m3'."
    )

# ── PIVOT TO (n_times, n_alts) ARRAYS ─────────────────────────────────────────
dates_all = pd.DatetimeIndex(sorted(results["Date"].unique()))
print(f"  {len(dates_all)} unique dates  |  altitudes: {ALTITUDES_KM} km")

density_pivot  = results.pivot_table(
    index="Date", columns="Altitude_km",
    values=DENSITY_COL, aggfunc="mean"
)
atomic_O_pivot = results.pivot_table(
    index="Date", columns="Altitude_km",
    values="O_number_density_m3", aggfunc="mean"
)

density_pivot  = density_pivot.reindex(index=dates_all,  columns=ALTITUDES_KM)
atomic_O_pivot = atomic_O_pivot.reindex(index=dates_all, columns=ALTITUDES_KM)

density_array  = density_pivot.values.astype(float)
atomic_O_array = atomic_O_pivot.values.astype(float)

# Sanity check – density at 200 km should be ~1e-10 to 1e-9 kg/m³
med_rho_200 = np.nanmedian(density_array[:, ALTITUDES_KM.index(200)])
if not (1e-13 < med_rho_200 < 1e-7):
    print(
        f"\n  *** UNIT WARNING ***\n"
        f"  Median density at 200 km = {med_rho_200:.3e} kg/m³\n"
        f"  Expected range: ~1e-10 to 1e-9 kg/m³.\n"
        f"  Check DENSITY_COL ('{DENSITY_COL}') and DENSITY_UNIT ('{DENSITY_UNIT}')."
    )
else:
    print(f"  Median density at 200 km = {med_rho_200:.3e} kg/m³  ✓")

n_nan_rho = np.isnan(density_array).sum()
n_nan_O   = np.isnan(atomic_O_array).sum()
if n_nan_rho or n_nan_O:
    print(f"  WARNING: {n_nan_rho} NaN density, {n_nan_O} NaN atomic-O values.")

# ── BUILD FEATURE DATAFRAME ───────────────────────────────────────────────────
print("\nBuilding feature DataFrame from space-weather CSV...")
feature_df = build_feature_df_dtm(
    dates       = dates_all,
    sw_csv_path = SW_CSV,
    f107_col    = "F107_adj",
    kp1_col     = "Kp_09_12",
    include_doy = True,
)
print(f"  Feature columns : {list(feature_df.columns)}")
print(f"  Shape           : {feature_df.shape}")

# ── FIT PREDICTOR ─────────────────────────────────────────────────────────────
predictor = AtmosphericAnalogPredictor(model_name="DTM-2020")
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