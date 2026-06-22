import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pymsis
import os

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_CSV = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\space_weather_daily_1996_2025.csv"

# Output files
OUTPUT_SUBDAILY = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\output_subdaily.csv"
OUTPUT_DAILY = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\output_daily.csv"
OUTPUT_MONTHLY = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\output_monthly.csv"

# Fixed geographic location
LATITUDE = 0.0      # degrees
LONGITUDE = 0.0     # degrees

# Altitudes to evaluate [km]
ALTITUDES_KM = [200, 225, 250, 275, 300]

# Times per day to sample [UTC hours]
HOURS_UTC = [0, 6, 12, 18]


# Ensure output directory exists
output_dir = os.path.dirname(OUTPUT_SUBDAILY)
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory confirmed: {output_dir}")

# ============================================================
# LOAD INPUT DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

# Standardize column names if needed
df.columns = [c.strip() for c in df.columns]

# Create datetime column
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# ============================================================
# COMPUTE 81-DAY F10.7 AVERAGE
# ============================================================

# Use centered rolling mean if enough data exists on both sides.
# Near edges, min_periods=1 prevents NaNs, but is less ideal physically.
df["F107A"] = df["F107_adj"].rolling(window=81, center=True, min_periods=1).mean()

# ============================================================
# BUILD SUB-DAILY EVALUATION GRID
# ============================================================

rows = []

for _, row in df.iterrows():
    base_date = row["Date"]

    for hour in HOURS_UTC:
        dt = base_date + pd.Timedelta(hours=hour)

        for alt in ALTITUDES_KM:
            rows.append({
                "datetime": dt,
                "Year": row["Year"],
                "Month": row["Month"],
                "Day": row["Day"],
                "Hour_UTC": hour,
                "lat": LATITUDE,
                "lon": LONGITUDE,
                "alt_km": alt,
                "Ap_daily": row["Ap_daily"],
                "F107": row["F107_adj"],
                "F107A": row["F107A"]
            })

grid = pd.DataFrame(rows)

# ============================================================
# PREPARE INPUTS FOR PYMSIS
# ============================================================

dates = pd.to_datetime(grid["datetime"]).to_numpy()
lons = grid["lon"].to_numpy()
lats = grid["lat"].to_numpy()
alts = grid["alt_km"].to_numpy()
f107 = grid["F107"].to_numpy(dtype=float)
f107a = grid["F107A"].to_numpy(dtype=float)

# pymsis expects aps as an (n, 7) array for each timestamp.
# If you only have one daily Ap value per row, repeat it across the
# seven MSIS Ap columns as a simple approximation.
aps = np.repeat(grid["Ap_daily"].to_numpy(dtype=float)[:, None], 7, axis=1)

# ============================================================
# RUN NRLMSISE-00
# ============================================================

# pymsis.run returns an array of atmospheric outputs.
# One of the outputs is total mass density.
# Depending on pymsis version, output indexing may differ slightly.
# In many versions:
#   output[:, 0] = total mass density [kg/m^3]
# Check package docs if needed.

msis_output = pymsis.calculate(
    dates=dates,
    lons=lons,
    lats=lats,
    alts=alts,
    f107s=f107,
    f107as=f107a,
    aps=aps
)

# Grab the outputs from msis
grid["Density_kg_m3"] = msis_output[:, 0]
grid["Atomic_O_density_m3"] = msis_output[:, 1]

# ============================================================
# SAVE SUB-DAILY RESULTS
# ============================================================

grid.to_csv(OUTPUT_SUBDAILY, index=False)
print(f"Saved sub-daily results to {OUTPUT_SUBDAILY}")

# ============================================================
# DAILY AVERAGES
# ============================================================

daily = (
    grid.groupby(["Year", "Month", "Day", "lat", "lon", "alt_km"], as_index=False)
        .agg({
            "F107": "first",
            "F107A": "first",
            "Ap_daily": "first",
            "Density_kg_m3": ["mean", "std", "min", "max"],
            "Atomic_O_density_m3": ["mean", "std", "min", "max"]
        })
)

daily.columns = [
    "Year", "Month", "Day", "lat", "lon", "alt_km",
    "F107", "F107A", "Ap_daily",
    "Density_mean_kg_m3", "Density_std_kg_m3", "Density_min_kg_m3", "Density_max_kg_m3",
    "Atomic_O_mean_m3", "Atomic_O_std_m3", "Atomic_O_min_m3", "Atomic_O_max_m3"
]

daily["Date"] = pd.to_datetime(daily[["Year", "Month", "Day"]])

daily.to_csv(OUTPUT_DAILY, index=False)
print(f"Saved daily averages to {OUTPUT_DAILY}")

# ============================================================
# MONTHLY AVERAGES
# ============================================================

monthly = (
    daily.groupby(["Year", "Month", "lat", "lon", "alt_km"], as_index=False)
         .agg({
             "F107": "mean",
             "F107A": "mean",
             "Ap_daily": "mean",
             "Density_mean_kg_m3": ["mean", "std", "min", "max"]
         })
)

monthly.columns = [
    "Year", "Month", "lat", "lon", "alt_km",
    "F107_monthly_mean",
    "F107A_monthly_mean",
    "Ap_monthly_mean",
    "Density_monthly_mean_kg_m3",
    "Density_monthly_std_kg_m3",
    "Density_monthly_min_kg_m3",
    "Density_monthly_max_kg_m3"
]
# Save daily results
print(f"Saved daily averages to {OUTPUT_DAILY}")

# ============================================================
# EXPORT ATOMIC OXYGEN DAILY AVERAGES AS SEPARATE EXCEL FILES
# ============================================================
# ============================================================

altitudes_to_export = [200, 225, 250, 275, 300]

for alt in altitudes_to_export:
    df_alt = daily[daily["alt_km"] == alt][["Date", "Atomic_O_mean_m3"]].copy()
    df_alt["Date"] = df_alt["Date"].dt.strftime("%m/%d/%Y")
    
    output_file = os.path.join(output_dir, f"atomic_oxygen_{alt}km.csv")
    df_alt.to_csv(output_file, index=False)
    
    print(f"Saved atomic oxygen file: {output_file}")

# Optional: make a month-start datetime column for plotting
monthly["Date"] = pd.to_datetime(
    monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01"
)

monthly.to_csv(OUTPUT_MONTHLY, index=False)
print(f"Saved monthly averages to {OUTPUT_MONTHLY}")