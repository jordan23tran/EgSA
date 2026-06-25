import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from swami import MCM
import time

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = "/mnt/c/Users/jorda/OneDrive - Massachusetts Institute of Technology/Documents/Summer 26/space_weather_daily_may_1996_2026.csv"
OUTPUT_DIR = "/home/jordan23tran/mcm/output"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "mcm_results_all_altitudes.csv")

LATITUDE = 0.0
LONGITUDE = 0.0
LOCAL_TIME = 12.0

ALTITUDES_KM = [200, 225, 250, 275, 300]

START_DATE = "1996-05-01"
END_DATE   = "2026-05-01"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD INPUT DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

df["Date_dt"] = pd.to_datetime(df[["Year", "Month", "Day"]])
df = df[(df["Date_dt"] >= START_DATE) & (df["Date_dt"] <= END_DATE)].copy()
df = df.sort_values("Date_dt").reset_index(drop=True)

numeric_cols = [
    "Kp_00_03", "Kp_03_06", "Kp_06_09", "Kp_09_12",
    "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
    "ap_00_03", "ap_03_06", "ap_06_09", "ap_09_12",
    "ap_12_15", "ap_15_18", "ap_18_21", "ap_21_24",
    "Ap_daily", "SunspotNumber", "F107_adj"
]

missing = [c for c in numeric_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# MODEL INPUTS
# ============================================================

df["F107"] = df["F107_adj"]
df["F107m"] = df["F107"].rolling(window=81, center=True, min_periods=1).mean()
df["DOY"] = df["Date_dt"].dt.dayofyear.astype(float)

# For local noon, use 09-12 Kp
df["kp1"] = df["Kp_09_12"]

# Daily mean Kp as approximation for previous 24-hour mean
kp_cols = [
    "Kp_00_03", "Kp_03_06", "Kp_06_09", "Kp_09_12",
    "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24"
]
df["kp2"] = df[kp_cols].mean(axis=1)

# Drop rows with essential missing values
df = df.dropna(subset=["Date_dt", "F107", "F107m", "kp1", "kp2", "DOY"]).copy()

print("Input rows after filtering:", len(df))
print(df[["Date_dt", "F107", "F107m", "kp1", "kp2"]].head())

# ============================================================
# RUN MCM
# ============================================================

# Atomic oxygen atom mass
AMU_TO_G = 1.66053906660e-24   # grams
O_ATOMIC_MASS = 15.999
M_O_G = O_ATOMIC_MASS * AMU_TO_G   # grams per O atom

model = MCM()
results = []

total_runs = len(df) * len(ALTITUDES_KM)
run_count = 0
start_time = time.time()

for i, row in enumerate(df.itertuples(index=False), start=1):
    for alt_km in ALTITUDES_KM:
        run_count += 1

        try:
            out = model.run(
                altitude=float(alt_km),
                day_of_year=float(row.DOY),
                local_time=float(LOCAL_TIME),
                latitude=float(LATITUDE),
                longitude=float(LONGITUDE),
                f107=float(row.F107),
                f107m=float(row.F107m),
                kp1=float(row.kp1),
                kp2=float(row.kp2),
                get_uncertainty=False,
                get_winds=False
            )
            out_dict = out._asdict()
        except Exception as e:
            print(f"Model failed on {row.Date_dt.date()} at {alt_km} km: {e}")
            out_dict = {}

        d_O = out_dict.get("d_O", np.nan)

        if pd.notna(d_O):
            O_number_density_cm3 = d_O / M_O_G
            O_number_density_m3 = O_number_density_cm3 * 1e6
        else:
            O_number_density_cm3 = np.nan
            O_number_density_m3 = np.nan

        result_row = {
            "Date": row.Date_dt,
            "Altitude_km": alt_km,
            "Latitude": LATITUDE,
            "Longitude": LONGITUDE,
            "LocalTime": LOCAL_TIME,
            "DOY": row.DOY,
            "F107": row.F107,
            "F107m": row.F107m,
            "kp1": row.kp1,
            "kp2": row.kp2,
            "Ap_daily": row.Ap_daily,
            "SunspotNumber": row.SunspotNumber,
            "O_number_density_cm3": O_number_density_cm3,
            "O_number_density_m3": O_number_density_m3,
        }

        result_row.update(out_dict)
        results.append(result_row)

        if run_count % 500 == 0 or run_count == 1:
            elapsed = time.time() - start_time
            avg_time_per_run = elapsed / run_count
            remaining_runs = total_runs - run_count
            eta_hours = (avg_time_per_run * remaining_runs) / 3600.0

            print(
                f"Run {run_count}/{total_runs} | "
                f"{100*run_count/total_runs:.2f}% complete | "
                f"Elapsed: {elapsed/60:.1f} min | "
                f"ETA: {eta_hours:.2f} hr"
            )
results_df = pd.DataFrame(results)

# Save combined CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved combined CSV:\n{OUTPUT_CSV}")

# Optional: also save one CSV per altitude
for alt in ALTITUDES_KM:
    sub = results_df[results_df["Altitude_km"] == alt].copy()
    out_file = os.path.join(OUTPUT_DIR, f"mcm_results_{alt}km.csv")
    sub.to_csv(out_file, index=False)

print("Saved per-altitude CSV files too.")

# ============================================================
# IDENTIFY MAIN OUTPUT COLUMN FOR PLOTTING
# ============================================================

print("Model output columns:")
print(results_df.columns.tolist())

candidate_density_cols = ["rho", "density", "Density", "mass_density"]
plot_col = None

for c in candidate_density_cols:
    if c in results_df.columns:
        plot_col = c
        break

if plot_col is None:
    numeric_output_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = ["Altitude_km", "Latitude", "Longitude", "LocalTime", "DOY", "F107", "F107m", "kp1", "kp2", "Ap_daily", "SunspotNumber"]
    numeric_output_cols = [c for c in numeric_output_cols if c not in excluded]
    if numeric_output_cols:
        plot_col = numeric_output_cols[0]

if plot_col is None:
    raise ValueError("Could not identify a numeric MCM output column to plot.")

print(f"Using '{plot_col}' for plotting.")


print(f"AI RETURN {results_df.columns.tolist()}")
print(results_df.head())

# ============================================================
# PLOTS
# ============================================================

# plot_df = results_df.copy()
# plot_df["Date"] = pd.to_datetime(plot_df["Date"])

# # 1. All altitudes on one figure
# plt.figure(figsize=(12, 6))
# for alt in ALTITUDES_KM:
#     sub = plot_df[plot_df["Altitude_km"] == alt]
#     plt.plot(sub["Date"], sub[plot_col], label=f"{alt} km", linewidth=1)

# plt.xlabel("Date")
# plt.ylabel(plot_col)
# plt.title(f"{plot_col} vs Time at Multiple Altitudes")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, f"{plot_col}_vs_time_all_altitudes.png"), dpi=200)
# plt.show()

# # 2. One subplot per altitude
# fig, axes = plt.subplots(len(ALTITUDES_KM), 1, figsize=(12, 3 * len(ALTITUDES_KM)), sharex=True)

# if len(ALTITUDES_KM) == 1:
#     axes = [axes]

# for ax, alt in zip(axes, ALTITUDES_KM):
#     sub = plot_df[plot_df["Altitude_km"] == alt]
#     ax.plot(sub["Date"], sub[plot_col], linewidth=1)
#     ax.set_ylabel(f"{alt} km")
#     ax.grid(True, alpha=0.3)

# axes[-1].set_xlabel("Date")
# fig.suptitle(f"{plot_col} Time Series by Altitude", y=0.995)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, f"{plot_col}_vs_time_subplots.png"), dpi=200)
# plt.show()

# # 3. Monthly means
# plot_df["YearMonth"] = plot_df["Date"].dt.to_period("M").astype(str)
# monthly = (
#     plot_df.groupby(["YearMonth", "Altitude_km"], as_index=False)[plot_col]
#     .mean()
# )
# monthly["YearMonth"] = pd.to_datetime(monthly["YearMonth"])

# plt.figure(figsize=(12, 6))
# for alt in ALTITUDES_KM:
#     sub = monthly[monthly["Altitude_km"] == alt]
#     plt.plot(sub["YearMonth"], sub[plot_col], label=f"{alt} km", linewidth=1.5)

# plt.xlabel("Date")
# plt.ylabel(f"Monthly Mean {plot_col}")
# plt.title(f"Monthly Mean {plot_col} vs Time")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, f"monthly_mean_{plot_col}.png"), dpi=200)
# plt.show()