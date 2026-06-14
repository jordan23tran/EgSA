"""
JB2008 daily atmosphere from 1996-05-01 to 2026-05-01
at 200, 225, 250, 275, 300 km and (0N, 0E).

Outputs:
1. Atomic oxygen number density [1/m^3]
2. Total atmospheric mass density [kg/m^3]

Saves:
- jb2008_AO_number_density_daily.csv
- jb2008_atmospheric_density_daily.csv

Creates and saves 10 plots:
- 5 AO number density plots
- 5 atmospheric density plots

Important:
Your CSV has F107 and Ap, but a full JB2008 run requires additional
solar proxies beyond that. Therefore this script uses the official
JB2008 driver download/read tools from pyatmos for the model forcing,
while using your CSV for the requested date span and file organization.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timezone

# -----------------------------
# USER SETTINGS
# -----------------------------
lat_deg = 0.0
lon_deg = 0.0
altitudes_km = [200, 225, 250, 275, 300]

input_csv = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\space_weather_daily_may_1996_2026.csv"
output_dir = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26"

# -----------------------------
# IMPORT JB2008 TOOLS
# -----------------------------
# pip install pyatmos pandas matplotlib
from pyatmos import jb2008

try:
    from pyatmos import download_sw_jb2008, read_sw_jb2008
except ImportError:
    raise ImportError(
        "Could not import JB2008 space-weather helper functions from pyatmos.\n"
        "Try updating pyatmos, or inspect the package API for the exact helper names."
    )

# -----------------------------
# CREATE OUTPUT DIRECTORY IF NEEDED
# -----------------------------
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# READ USER CSV
# -----------------------------
sw_user = pd.read_csv(input_csv)

# Build datetime column from Year, Month, Day
sw_user["Date"] = pd.to_datetime(
    dict(year=sw_user["Year"], month=sw_user["Month"], day=sw_user["Day"]),
    utc=True
)

# Restrict exactly to requested range
start_date = pd.Timestamp("1997-01-01 12:00:00", tz="UTC")
end_date   = pd.Timestamp("2026-04-27 12:00:00", tz="UTC")
# Daily noon timestamps
dates = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")

n_times = len(dates)
n_alts = len(altitudes_km)

# -----------------------------
# DOWNLOAD / READ OFFICIAL JB2008 DRIVERS
# -----------------------------
print("Downloading/updating official JB2008 space-weather driver files...")
download_sw_jb2008()

print("Reading official JB2008 driver data...")

jb_swdata = read_sw_jb2008((
    r"C:\Users\jorda\src\sw-data\SOLFSMY.TXT",
    r"C:\Users\jorda\src\sw-data\DTCFILE.TXT"
))

# -----------------------------
# STORAGE
# -----------------------------
rho = np.full((n_times, n_alts), np.nan)   # atmospheric mass density [kg/m^3]
T   = np.full((n_times, n_alts), np.nan)   # neutral temperature [K]

# -----------------------------
# RUN JB2008
# -----------------------------
print(f"Running JB2008 for {n_times} days x {n_alts} altitudes...")

for i, t in enumerate(dates):
    if i % 500 == 0:
        print(f"  Progress: {i:5d} / {n_times}")

    t_py = t.to_pydatetime().replace(tzinfo=timezone.utc)

    for j, alt_km in enumerate(altitudes_km):
        # Typical pyatmos call signature
        result = jb2008(t_py, (lat_deg, lon_deg, alt_km), jb_swdata)
        rho[i, j] = result.rho
        T[i, j]   = result.T

print("JB2008 calculations complete.")

# -----------------------------
# BUILD OUTPUT TABLES
# -----------------------------
rho_df = pd.DataFrame({"Date": dates})
T_df   = pd.DataFrame({"Date": dates})

for j, alt_km in enumerate(altitudes_km):
    rho_df[f"Density_{alt_km}km_kg_m^-3"] = rho[:, j]
    T_df[f"Temperature_{alt_km}km_K"]     = T[:, j]

# -----------------------------
# SAVE CSV FILES
# -----------------------------
rho_csv_path = os.path.join(output_dir, "jb2008_atmospheric_density_daily.csv")
T_csv_path   = os.path.join(output_dir, "jb2008_neutral_temperature_daily.csv")

rho_df.to_csv(rho_csv_path, index=False)
T_df.to_csv(T_csv_path,     index=False)

print(f"Saved atmospheric density CSV to:\n  {rho_csv_path}")
print(f"Saved neutral temperature CSV to:\n  {T_csv_path}")

# -----------------------------
# PLOTTING
# -----------------------------
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True

for j, alt_km in enumerate(altitudes_km):
    # Density plot
    plt.figure()
    plt.plot(dates, rho[:, j], linewidth=1.0)
    plt.title(f"JB2008 Atmospheric Density at {alt_km} km (0°N, 0°E)")
    plt.xlabel("Date")
    plt.ylabel("Atmospheric Density [kg/m³]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"atmospheric_density_{alt_km}km.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Temperature plot
    plt.figure()
    plt.plot(dates, T[:, j], linewidth=1.0, color="tomato")
    plt.title(f"JB2008 Neutral Temperature at {alt_km} km (0°N, 0°E)")
    plt.xlabel("Date")
    plt.ylabel("Neutral Temperature [K]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"neutral_temperature_{alt_km}km.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("Saved 10 plot files to output directory.")
print("Done.")