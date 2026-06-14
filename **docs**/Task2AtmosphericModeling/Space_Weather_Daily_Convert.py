import pandas as pd

# Path to your downloaded raw GFZ text file
INPUT_TXT = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\gfz_raw.txt"

# Output CSV path
OUTPUT_CSV = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\space_weather_daily_may_1996_2026.csv"

# Read whitespace-delimited file
raw = pd.read_csv(INPUT_TXT, sep=r"\s+", header=None)

# Assign column names based on GFZ format
raw.columns = [
    "Year", "Month", "Day", "MJD", "MJD_mid", "BartelsRot", "DayOfBartelsRot",
    "Kp_00_03", "Kp_03_06", "Kp_06_09", "Kp_09_12", "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
    "ap_00_03", "ap_03_06", "ap_06_09", "ap_09_12", "ap_12_15", "ap_15_18", "ap_18_21", "ap_21_24",
    "Ap_daily", "SunspotNumber", "F107_obs", "F107_adj", "DataFlag"
]

# Keep only the columns needed for the first MSIS script
clean = raw[[
    "Year", "Month", "Day", "Kp_00_03", "Kp_03_06", "Kp_06_09", "Kp_09_12", "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
    "ap_00_03", "ap_03_06", "ap_06_09", "ap_09_12",
    "ap_12_15", "ap_15_18", "ap_18_21", "ap_21_24",
    "Ap_daily", "SunspotNumber", "F107_adj"
]]

# Save to CSV
clean.to_csv(OUTPUT_CSV, index=False)

print(f"Saved cleaned file to: {OUTPUT_CSV}")
print(clean.head())
