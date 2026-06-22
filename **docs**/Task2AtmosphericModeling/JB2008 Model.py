"""
JB2008 daily atmosphere from 1997-01-01 to 2026-04-27
at 200, 225, 250, 275, 300 km and (0N, 0E).
Now includes kNN analog predictor built from the JB2008 output.
1. Create a venv in python 3.10 and install numpy, pandas, matplot, pyatmos
2. Make sure the jb_2008_analog_predictor is in the same folder. 
3. The first time you run this file you create the pkl file, next time you run it, the script uses the pkl file.

HOW TO USE QUERY
---------------------------------
Inputs are self-explanatory except for:
DOY: The target date in the future you are modelling
K and method: Just hit enter for defaults

HOW TO READ OUTPUTS
---------------------------------
Dist: How different the input data is, 0.0 is perfect match, ~2.0 is small and a close analog, >>5 is large and a poor analog.
Weight: The higher the more accurate, weights add up to one for all the analogs.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timezone

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
lat_deg      = 0.0
lon_deg      = 0.0
altitudes_km = [200, 225, 250, 275, 300]

input_csv  = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\space_weather_daily_may_1996_2026.csv"
output_dir = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26"

SOLFSMY_PATH   = r"C:\Users\jorda\src\sw-data\SOLFSMY.TXT"
DTCFILE_PATH   = r"C:\Users\jorda\src\sw-data\DTCFILE.TXT"
PREDICTOR_PATH = os.path.join(output_dir, "jb2008_analog_predictor.pkl")

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from pyatmos import jb2008
try:
    from pyatmos import download_sw_jb2008, read_sw_jb2008
except ImportError:
    raise ImportError(
        "Could not import JB2008 space-weather helper functions from pyatmos.\n"
        "Try updating pyatmos, or inspect the package API for the exact helper names."
    )
from jb2008_analog_predictor import (
    build_feature_df,
    JB2008AnalogPredictor,
    terminal_query
)

# ── PKL GUARD ─────────────────────────────────────────────────────────────────
print(f"PKL exists : {os.path.exists(PREDICTOR_PATH)}")
print(f"Looking at : {PREDICTOR_PATH}")

if os.path.exists(PREDICTOR_PATH):
    # ── FAST PATH: predictor already built, just load and query ───────────────
    print(f"Loading existing predictor from:\n  {PREDICTOR_PATH}")
    predictor = JB2008AnalogPredictor.load(PREDICTOR_PATH)
    terminal_query(predictor, output_dir=output_dir)

else:
    # ── SLOW PATH: run JB2008, build predictor, save everything ───────────────

    os.makedirs(output_dir, exist_ok=True)

    # Read user CSV
    sw_user = pd.read_csv(input_csv)
    sw_user["Date"] = pd.to_datetime(
        dict(year=sw_user["Year"], month=sw_user["Month"], day=sw_user["Day"]),
        utc=True
    )

    # Build date range
    start_date = pd.Timestamp("1997-01-01 12:00:00", tz="UTC")
    end_date   = pd.Timestamp("2026-04-27 12:00:00", tz="UTC")
    dates      = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")

    n_times = len(dates)
    n_alts  = len(altitudes_km)

    # Download / read JB2008 drivers
    print("Downloading/updating official JB2008 space-weather driver files...")
    download_sw_jb2008()

    print("Reading official JB2008 driver data...")
    jb_swdata = read_sw_jb2008((SOLFSMY_PATH, DTCFILE_PATH))

    # Storage arrays
    rho = np.full((n_times, n_alts), np.nan)
    T   = np.full((n_times, n_alts), np.nan)

    # Run JB2008
    print(f"Running JB2008 for {n_times} days x {n_alts} altitudes...")
    for i, t in enumerate(dates):
        if i % 500 == 0:
            print(f"  Progress: {i:5d} / {n_times}")

        t_py = t.to_pydatetime().replace(tzinfo=timezone.utc)

        for j, alt_km in enumerate(altitudes_km):
            result    = jb2008(t_py, (lat_deg, lon_deg, alt_km), jb_swdata)
            rho[i, j] = result.rho
            T[i, j]   = result.T

    print("JB2008 calculations complete.")

    # Build feature DataFrame
    print("\nBuilding feature DataFrame from SOLFSMY / DTCFILE...")
    feature_df = build_feature_df(
        dates        = dates,
        solfsmy_path = SOLFSMY_PATH,
        dtcfile_path = DTCFILE_PATH,
        include_dtc  = True,
        include_doy  = True,
    )
    print(f"Feature columns : {list(feature_df.columns)}")
    print(f"Feature matrix  : {feature_df.shape}")

    # Fit predictor
    predictor = JB2008AnalogPredictor()
    predictor.fit(
        dates        = dates,
        feature_df   = feature_df,
        rho_array    = rho,
        T_array      = T,
        altitudes_km = altitudes_km
    )

    # Save predictor
    predictor.save(PREDICTOR_PATH)
    print(f"Predictor saved to:\n  {PREDICTOR_PATH}")

    # Build and save CSV output tables
    rho_df = pd.DataFrame({"Date": dates})
    T_df   = pd.DataFrame({"Date": dates})

    for j, alt_km in enumerate(altitudes_km):
        rho_df[f"Density_{alt_km}km_kg_m^-3"] = rho[:, j]
        T_df[f"Temperature_{alt_km}km_K"]     = T[:, j]

    rho_csv_path = os.path.join(output_dir, "jb2008_atmospheric_density_daily.csv")
    T_csv_path   = os.path.join(output_dir, "jb2008_neutral_temperature_daily.csv")

    rho_df.to_csv(rho_csv_path, index=False)
    T_df.to_csv(T_csv_path,     index=False)

    print(f"Saved atmospheric density CSV to:\n  {rho_csv_path}")
    print(f"Saved neutral temperature CSV to:\n  {T_csv_path}")

    # Plots
    plt.rcParams["figure.figsize"] = (12, 5)
    plt.rcParams["axes.grid"]      = True

    for j, alt_km in enumerate(altitudes_km):
        plt.figure()
        plt.plot(dates, rho[:, j], linewidth=1.0)
        plt.title(f"JB2008 Atmospheric Density at {alt_km} km (0°N, 0°E)")
        plt.xlabel("Date")
        plt.ylabel("Atmospheric Density [kg/m³]")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"atmospheric_density_{alt_km}km.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

        plt.figure()
        plt.plot(dates, T[:, j], linewidth=1.0, color="tomato")
        plt.title(f"JB2008 Neutral Temperature at {alt_km} km (0°N, 0°E)")
        plt.xlabel("Date")
        plt.ylabel("Neutral Temperature [K]")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"neutral_temperature_{alt_km}km.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    print("Saved 10 plot files to output directory.")

    # Launch interactive terminal query
    terminal_query(predictor, output_dir=output_dir)

    print("Done.")
