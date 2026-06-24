"""
gfz_cleaner.py
==============
Purpose
-------
Parses the raw GFZ (German Research Centre for Geosciences) Kp / Ap / F10.7
space-weather text file and converts it to a clean, model-ready CSV that is
consumed as the sole solar/geomagnetic input by all three atmospheric model
pipelines in this project.

Inputs
------
  gfz_raw.txt
    Downloaded from https://kp.gfz.de/en/data
    (Scroll down to Download via HTTPS → "Kp, ap, Ap, SN and F10.7 since 1932, daily updated" 
    then Copy onto Notepad)
    Format: Delete everything header row and above. First line should say 1932 01 01...
    Expected column layout (37 columns total):
      Year  Month  Day  MJD  MJD_mid  BartelsRot  DayOfBartelsRot
      Kp_00_03  Kp_03_06 … Kp_21_24   (8 three-hourly Kp)
      ap_00_03  ap_03_06 … ap_21_24   (8 three-hourly ap)
      Ap_daily  SunspotNumber  F107_obs  F107_adj  DataFlag

Outputs
-------
  space_weather_daily_1932_2026.csv
    Columns retained for modelling:
      Year, Month, Day,
      Kp_00_03 … Kp_21_24   (8 columns)
      ap_00_03 … ap_21_24   (8 columns)
      Ap_daily, SunspotNumber, F107_adj

Scripts that read this CSV
--------------------------
  nrlmsis_runner.py            → NRLMSIS-00 model run
  run_msis_analog_predictor.py → NRLMSIS-00 analog predictor
  run_dtm_analog_predictor.py  → DTM-2020 analog predictor (WSL)
  run_jb2008.py                → JB2008 model run + analog predictor

Environment
-----------
  Windows 10/11, standard Python 3.10+
  Required package: pandas  (pip install pandas)
  No virtual environment needed; any pandas installation works.

How to run (Windows Command Prompt or PowerShell)
-------------------------------------------------
  1. Download the GFZ file:
       https://www.gfz-potsdam.de/en/kp-index/
     Save the .txt file to INPUT_TXT below.
  2. Edit the two path constants in the USER SETTINGS block below.
  3. Open Command Prompt, navigate to the folder containing this script, and run:
       python gfz_cleaner.py
  4. Confirm the output CSV was created at OUTPUT_CSV and check the preview.

Notes
-----
  • Run this once before running any of the three atmospheric model scripts.
  • The file covers Jan 1 1932 – Jun 23 2026 in the default path naming, but the
    cleaner works for any date range in the GFZ format.
  • F107_adj is the F10.7 solar radio flux adjusted to 1 AU; this is the
    physically correct value to use with NRLMSIS-00 and DTM-2020.
"""

import pandas as pd

# ============================================================
# USER SETTINGS  ← edit these two paths before running
# ============================================================

INPUT_TXT  = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26\gfz_raw.txt"
)

OUTPUT_CSV = (
    r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology"
    r"\Documents\Summer 26\space_weather_daily_1932_2026.csv"
)

# ============================================================
# PARSE
# ============================================================

raw = pd.read_csv(INPUT_TXT, sep=r"\s+", header=None)

raw.columns = [
    "Year", "Month", "Day", "MJD", "MJD_mid", "BartelsRot", "DayOfBartelsRot",
    "Kp_00_03", "Kp_03_06", "Kp_06_09", "Kp_09_12",
    "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
    "ap_00_03", "ap_03_06", "ap_06_09", "ap_09_12",
    "ap_12_15", "ap_15_18", "ap_18_21", "ap_21_24",
    "Ap_daily", "SunspotNumber", "F107_obs", "F107_adj", "DataFlag",
]

clean = raw[[
    "Year", "Month", "Day",
    "Kp_00_03", "Kp_03_06", "Kp_06_09", "Kp_09_12",
    "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
    "ap_00_03", "ap_03_06", "ap_06_09", "ap_09_12",
    "ap_12_15", "ap_15_18", "ap_18_21", "ap_21_24",
    "Ap_daily", "SunspotNumber", "F107_adj",
]]

clean.to_csv(OUTPUT_CSV, index=False)

print(f"Saved cleaned space-weather CSV to:\n  {OUTPUT_CSV}")
print(f"Rows: {len(clean)}  |  Columns: {list(clean.columns)}")
print()
print(clean.head())
