import pandas as pd
import numpy as np
import os

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_CSV = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26\space_weather_daily_1996_2025.csv"
OUTPUT_DIR = r"C:\Users\jorda\OneDrive - Massachusetts Institute of Technology\Documents\Summer 26"

LATITUDE = 0.0
LONGITUDE = 0.0

ALTITUDES_KM = [200, 225, 250, 275, 300]

# ============================================================
# LOAD INPUT DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
df.columns = [c.strip() for c in df.columns]

# ---- Adjust these column names if needed ----
# If your file uses F107_adj, map that to F107 here
if "F107" not in df.columns and "F107_adj" in df.columns:
    df["F107"] = df["F107_adj"]

# Create Date column
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

# Keep only what we need
required_cols = ["Date", "Year", "Month", "Day", "F107", "Ap_daily"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in input CSV: {missing}")

# Optional Kp handling
if "Kp" not in df.columns:
    print("Warning: 'Kp' column not found. Proceeding with Ap_daily only.")
    df["Kp"] = np.nan

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# ============================================================
# OPTIONAL: COMPUTE 81-DAY F10.7A IF NEEDED BY DTM2020
# ============================================================

df["F107A"] = df["F107"].rolling(window=81, center=True, min_periods=1).mean()

# ============================================================
# IMPORT SWAMI / DTM2020
# ============================================================

# IMPORTANT:
# You may need to adjust this import depending on your swami version.
# Examples could look like:
#   from swami import DTM2020
#   from swami.models import DTM2020
#   from swami.atmosphere import DTM2020

try:
    from swami import DTM2020
except ImportError:
    try:
        from swami.models import DTM2020
    except ImportError:
        raise ImportError(
            "Could not import DTM2020 from swami. "
            "Check your swami installation and the correct import path."
        )

# ============================================================
# INITIALIZE MODEL
# ============================================================

# Depending on swami, this may be:
#   model = DTM2020()
# or something else.
model = DTM2020()

# ============================================================
# HELPER FUNCTION TO EXTRACT DTM2020 OUTPUTS
# ============================================================

def get_dtm2020_outputs(model, date, alt_km, lat, lon, f107, f107a, ap_daily, kp):
    """
    Runs DTM2020 for one date/location/altitude and returns:
    - air density
    - AO composition

    You may need to modify the internals of this function to match the exact
    swami API and output field names on your machine.
    """

    # --------------------------------------------------------
    # EXAMPLE CALL PATTERN
    # --------------------------------------------------------
    # The actual call signature may differ.
    # Common possibilities include:
    #
    # result = model.calculate(
    #     time=date,
    #     alt_km=alt_km,
    #     lat=lat,
    #     lon=lon,
    #     f107=f107,
    #     f107a=f107a,
    #     ap=ap_daily,
    #     kp=kp
    # )
    #
    # or:
    #
    # result = model.run(...)
    #
    # or:
    #
    # result = model(...)
    #
    # --------------------------------------------------------

    result = model.calculate(
        time=date,
        alt_km=alt_km,
        lat=lat,
        lon=lon,
        f107=f107,
        f107a=f107a,
        ap=ap_daily,
        kp=kp
    )

    # --------------------------------------------------------
    # EXTRACT AIR DENSITY
    # --------------------------------------------------------
    # Common possible field names:
    #   result["rho"]
    #   result["density"]
    #   result["mass_density"]
    #
    # Adjust as needed.
    # --------------------------------------------------------

    if isinstance(result, dict):
        # Try likely keys
        if "density" in result:
            air_density = result["density"]
        elif "rho" in result:
            air_density = result["rho"]
        elif "mass_density" in result:
            air_density = result["mass_density"]
        else:
            raise KeyError("Could not find density field in DTM2020 result.")

        # ----------------------------------------------------
        # EXTRACT ATOMIC OXYGEN COMPOSITION
        # ----------------------------------------------------
        # Possible keys:
        #   result["O"]
        #   result["atomic_oxygen"]
        #   result["ao"]
        #   result["O_fraction"]
        #
        # If the model gives only atomic oxygen density, use that.
        # If it gives both O and total density/composition, compute a fraction.
        # ----------------------------------------------------

        if "O_fraction" in result:
            ao_composition = result["O_fraction"]
        elif "atomic_oxygen_fraction" in result:
            ao_composition = result["atomic_oxygen_fraction"]
        elif "O" in result:
            ao_composition = result["O"]
        elif "atomic_oxygen" in result:
            ao_composition = result["atomic_oxygen"]
        elif "ao" in result:
            ao_composition = result["ao"]
        else:
            raise KeyError("Could not find atomic oxygen output in DTM2020 result.")

    else:
        # If result is an object rather than a dict, try attribute access
        if hasattr(result, "density"):
            air_density = result.density
        elif hasattr(result, "rho"):
            air_density = result.rho
        elif hasattr(result, "mass_density"):
            air_density = result.mass_density
        else:
            raise AttributeError("Could not find density attribute in DTM2020 result.")

        if hasattr(result, "O_fraction"):
            ao_composition = result.O_fraction
        elif hasattr(result, "atomic_oxygen_fraction"):
            ao_composition = result.atomic_oxygen_fraction
        elif hasattr(result, "O"):
            ao_composition = result.O
        elif hasattr(result, "atomic_oxygen"):
            ao_composition = result.atomic_oxygen
        elif hasattr(result, "ao"):
            ao_composition = result.ao
        else:
            raise AttributeError("Could not find atomic oxygen output in DTM2020 result.")

    return float(air_density), float(ao_composition)

# ============================================================
# RUN DTM2020 FOR EACH ALTITUDE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for alt in ALTITUDES_KM:
    output_rows = []

    print(f"Processing altitude: {alt} km")

    for _, row in df.iterrows():
        date = row["Date"]
        f107 = float(row["F107"])
        f107a = float(row["F107A"])
        ap_daily = float(row["Ap_daily"])
        kp = row["Kp"]

        if pd.isna(kp):
            kp = None
        else:
            kp = float(kp)

        try:
            air_density, ao_composition = get_dtm2020_outputs(
                model=model,
                date=date,
                alt_km=alt,
                lat=LATITUDE,
                lon=LONGITUDE,
                f107=f107,
                f107a=f107a,
                ap_daily=ap_daily,
                kp=kp
            )

            output_rows.append({
                "Date": date.strftime("%m/%d/%Y"),
                "Air_Density": air_density,
                "AO_Composition": ao_composition
            })

        except Exception as e:
            print(f"Error on {date.strftime('%Y-%m-%d')} at {alt} km: {e}")
            output_rows.append({
                "Date": date.strftime("%m/%d/%Y"),
                "Air_Density": np.nan,
                "AO_Composition": np.nan
            })

    out_df = pd.DataFrame(output_rows)

    output_file = os.path.join(OUTPUT_DIR, f"DTM2020_{alt}km.csv")
    out_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("All DTM2020 altitude files created.")