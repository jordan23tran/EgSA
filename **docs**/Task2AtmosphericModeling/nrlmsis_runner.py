"""
nrlmsis_runner.py
=================
Purpose
-------
Runs the NRLMSIS-00 empirical atmosphere model (via the `pymsis` Python
package) across a daily space-weather time series and exports a CSV of
daily-averaged atmospheric species densities and temperature at each
configured altitude.  The output is consumed directly by
run_msis_analog_predictor.py.

Why sub-daily sampling?  NRLMSIS-00 is a function of local solar time as
well as geomagnetic and solar indices.  Running the model at 4 evenly-spaced
local times (0, 6, 12, 18 h) and averaging eliminates the diurnal bias that
would otherwise corrupt the analog-day comparisons; all SPARTA simulations
represent orbit-averaged (not fixed local-time) conditions.

Inputs
------
  space_weather_daily_1932_2026.csv
    Produced by gfz_cleaner.py.
    Required columns: Year, Month, Day, Ap_daily, F107_adj

Outputs
-------
  output_daily.csv
    One row per (date, altitude).
    Columns:
      Year, Month, Day, F107, F107A, Ap_daily, lat, lon, alt_km,
      Density_mean_kg_m3, T_mean_K,
      O_mean_m3, N2_mean_m3, O2_mean_m3, He_mean_m3,
      Ar_mean_m3, H_mean_m3, N_mean_m3

    The presence of  O_mean_m3  is used by run_msis_analog_predictor.py
    to detect whether this file was produced by the new (v2) runner.
    If O_mean_m3 is missing, the predictor runner will refuse to proceed.

  O_column_<alt>km.csv   (one per altitude, for quick manual validation)

Dependencies
------------
  pip install pymsis pandas numpy

  pymsis ships the NRLMSISE-00 coefficients internally — no separate data
  download is required.

Environment
-----------
  Windows 10/11, Python 3.10+
  Run from the standard Windows Python environment (NOT WSL).

How to run
----------
  1. Edit INPUT_SW_CSV and OUTPUT_DIR in the USER SETTINGS block below.
  2. Hit run python file or Open Command Prompt (Win+R → cmd) or PowerShell and run:
       python nrlmsis_runner.py
  3. Watch the sanity-check printout and verify magnitudes match comments.
  4. After completion, run "run_msis_analog_predictor.py".

PYMSIS column mapping note
--------------------------
pymsis.calculate() returns an array of shape (..., 11).
The mapping below was confirmed on this project (see comments):
  index [0] → total mass density [kg/m³]    ← confirmed
  index [1] → N2 number density  [m⁻³]      ← confirmed (NOT atomic O)
  index [3] → O  number density  [m⁻³]      ← confirmed (bug-fix from original)

If the runtime sanity-check block prints unexpected magnitudes, adjust
PYMSIS_IDX to match your installed pymsis version and model variant.

Schema version
--------------
This script produces format v2 of output_daily.csv (includes O_mean_m3).
run_msis_analog_predictor.py rejects v1 files and will tell you to re-run
this script.
"""

import os
import numpy as np
import pandas as pd
import pymsis

# ============================================================
# USER SETTINGS  ← edit these before running
# ============================================================

INPUT_SW_CSV = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26\space_weather_daily_1932_2026.csv"
)

OUTPUT_DIR = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26"
)

# Altitudes [km] — must match what run_msis_analog_predictor.py expects
ALTITUDES_KM = [200.0, 225.0, 250.0, 275.0, 300.0]

# Geographic point: equatorial sub-satellite point for orbit-averaged density
LAT = 0.0   # deg
LON = 0.0   # deg

# Local solar times sampled per day for diurnal average (hours, 0-23)
LOCAL_TIMES = [0, 6, 12, 18]

# ============================================================
# PYMSIS OUTPUT INDEX MAPPING
# Adjust this dict if your pymsis version uses a different column order.
# Confirmed indices are marked; others are assumed from NRLMSISE-00 specs.
# ============================================================

PYMSIS_IDX = {
    'total_density': 0,   # [kg/m³]  ← confirmed on this project
    'N2':  1,             # [m⁻³]   ← confirmed (NOT atomic O)
    'O2':  2,             # [m⁻³]   ← assumed (between N2 and O)
    'O':   3,             # [m⁻³]   ← confirmed (old bug used index 1)
    'He':  4,             # [m⁻³]   ← assumed
    'Ar':  5,             # [m⁻³]   ← assumed
    'H':   6,             # [m⁻³]   ← assumed
    'N':   7,             # [m⁻³]   ← assumed
    # index 8 = anomalous O (~0 at LEO) — intentionally skipped
    'T_exo': 9,           # [K]  exospheric temperature — stored but not in daily CSV
    'T':    10,           # [K]  kinetic temperature at altitude — used for SPARTA
}

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load space-weather ─────────────────────────────────────────────────
    print("Loading space-weather CSV …")
    sw = pd.read_csv(INPUT_SW_CSV)
    sw['Date'] = pd.to_datetime(sw[['Year', 'Month', 'Day']])
    sw = sw.sort_values('Date').dropna(subset=['F107_adj', 'Ap_daily']).reset_index(drop=True)

    # 81-day centred rolling mean of F10.7 (required by NRLMSIS-00 as F107A)
    sw['F107A'] = sw['F107_adj'].rolling(window=81, center=True, min_periods=40).mean()
    # Fall back to daily value where rolling window is incomplete
    sw['F107A'] = sw['F107A'].fillna(sw['F107_adj'])

    print(f"  {len(sw)} days  |  {sw['Date'].iloc[0].date()} → {sw['Date'].iloc[-1].date()}")
    print(f"  Altitudes : {ALTITUDES_KM} km")
    print(f"  Local times: {LOCAL_TIMES} h  ({len(LOCAL_TIMES)} per day for diurnal average)")

    # ── Build sub-daily grid ───────────────────────────────────────────────
    # First-principles reason: NRLMSIS-00 is a function of local solar time.
    # Averaging over 4 equally-spaced times removes the diurnal bias.
    print("\nBuilding sub-daily grid …")
    rows = []
    for _, row in sw.iterrows():
        for lt in LOCAL_TIMES:
            ts = pd.Timestamp(int(row.Year), int(row.Month), int(row.Day), lt)
            for alt in ALTITUDES_KM:
                rows.append({
                    'datetime'  : ts,
                    'Year'      : int(row.Year),
                    'Month'     : int(row.Month),
                    'Day'       : int(row.Day),
                    'lat'       : LAT,
                    'lon'       : LON,
                    'alt_km'    : alt,
                    'F107'      : float(row.F107_adj),
                    'F107A'     : float(row.F107A),
                    'Ap_daily'  : float(row.Ap_daily),
                })
    grid = pd.DataFrame(rows)
    print(f"  Grid size : {len(grid):,} rows")

    # ── Run NRLMSIS-00 ─────────────────────────────────────────────────────
    # aps must be shape (n, 7): columns are Ap at t, t-3h, t-6h, … t-18h.
    # Using daily Ap repeated across all 7 slots is a standard approximation
    # when sub-3-hourly Ap history is not required (valid for kNN analog use).
    print("Running pymsis.calculate() — this may take a few minutes …")
    aps = np.repeat(grid['Ap_daily'].values[:, None], 7, axis=1)

    msis_out = pymsis.calculate(
        dates=grid['datetime'].values,
        lons=grid['lon'].values,
        lats=grid['lat'].values,
        alts=grid['alt_km'].values,
        f107s=grid['F107'].values,
        f107as=grid['F107A'].values,
        aps=aps,
    )
    print(f"  pymsis output shape: {msis_out.shape}")

    # ── Sanity check ───────────────────────────────────────────────────────
    # Pick a mid-dataset row at ~250 km to cross-check magnitudes.
    mid = (len(grid) // 2) - (len(grid) // 2) % len(ALTITUDES_KM)
    s = msis_out[mid]
    print("\n  pymsis sanity check (mid-dataset, middle altitude):")
    print(f"    [0] = {s[0]:.3e}   expect ~1e-11 to 1e-9 kg/m³ (total mass density)")
    print(f"    [1] = {s[1]:.3e}   expect ~1e13 to 1e16 m⁻³   (N2)")
    print(f"    [3] = {s[3]:.3e}   expect ~1e13 to 1e16 m⁻³   (O)")
    print(f"   [10] = {s[10]:.1f} K      expect 700 – 1300 K  (kinetic temperature)")
    if not (1e-13 < s[0] < 1e-6):
        print("\n  ⚠  WARNING: [0] does not look like total mass density.")
        print("  ⚠  Adjust PYMSIS_IDX in this file to match your pymsis version.")
        print("  ⚠  Run:  python -c \"import pymsis; help(pymsis.calculate)\"  for docs.\n")

    # ── Extract all species and temperature from pymsis output ─────────────
    for sp, idx in PYMSIS_IDX.items():
        grid[sp] = msis_out[:, idx]

    # ── Daily aggregation: mean over LOCAL_TIMES ───────────────────────────
    # GROUP key: date + altitude.  The F107/F107A/Ap_daily values are
    # identical for all local times on the same day, so using 'first' is safe.
    grp_keys = ['Year', 'Month', 'Day', 'lat', 'lon', 'alt_km']
    sw_keys  = ['F107', 'F107A', 'Ap_daily']
    out_keys = ['total_density', 'N2', 'O2', 'O', 'He', 'Ar', 'H', 'N', 'T']

    agg_spec = {k: 'first' for k in sw_keys}
    agg_spec.update({k: 'mean' for k in out_keys})

    daily = grid.groupby(grp_keys).agg(agg_spec).reset_index()

    # Rename to self-documenting column names
    daily = daily.rename(columns={
        'total_density' : 'Density_mean_kg_m3',
        'T'             : 'T_mean_K',
        'O'             : 'O_mean_m3',
        'N2'            : 'N2_mean_m3',
        'O2'            : 'O2_mean_m3',
        'He'            : 'He_mean_m3',
        'Ar'            : 'Ar_mean_m3',
        'H'             : 'H_mean_m3',
        'N'             : 'N_mean_m3',
    })

    # ── Export daily CSV ───────────────────────────────────────────────────
    daily_path = os.path.join(OUTPUT_DIR, 'output_daily.csv')
    daily.to_csv(daily_path, index=False)
    print(f"\n  Saved: {daily_path}  ({len(daily):,} rows)")

    # ── Export per-altitude O column files (for manual validation) ─────────
    for alt in ALTITUDES_KM:
        sub = daily[daily['alt_km'] == alt][
            ['Year', 'Month', 'Day', 'F107', 'Ap_daily',
             'Density_mean_kg_m3', 'T_mean_K', 'O_mean_m3']
        ].copy()
        col_path = os.path.join(OUTPUT_DIR, f'O_column_{int(alt)}km.csv')
        sub.to_csv(col_path, index=False)
    print(f"  Per-altitude O column CSVs saved to: {OUTPUT_DIR}")

    # ── Quick summary ──────────────────────────────────────────────────────
    sample = daily[daily['alt_km'] == 250.0].iloc[len(daily[daily['alt_km'] == 250.0]) // 2]
    print(f"\n  Mid-series snapshot at 250 km:")
    print(f"    Density  = {sample.Density_mean_kg_m3:.4e} kg/m³")
    print(f"    Temp     = {sample.T_mean_K:.1f} K")
    print(f"    O        = {sample.O_mean_m3:.4e} m⁻³")
    print(f"    N2       = {sample.N2_mean_m3:.4e} m⁻³")
    print(f"\nDone.  Next step: run  run_msis_analog_predictor.py")


if __name__ == '__main__':
    main()
