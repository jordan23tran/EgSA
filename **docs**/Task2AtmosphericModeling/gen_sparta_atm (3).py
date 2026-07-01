"""
gen_sparta_atm.py

===============================================================================
HOW TO USE THIS SCRIPT
===============================================================================
1. Edit the two settings in the "USER CONFIG" section directly below this
   docstring:
       CSV_PATH   -> full path to your NRLMSIS-00 daily output CSV
       OUTPUT_DIR -> folder where the generated .atm file should be saved
   You only need to do this once (or whenever those locations change).

2. Run the script. In VS Code: open this file, click the ▶ Run button
   (top-right) or press F5/Ctrl+F5. No terminal typing required.

3. The script will prompt you, in the terminal, for:
       - a date   (format: YYYY-MM-DD, e.g. 1974-01-01)
       - an altitude in km (must be an exact value present in the CSV,
         e.g. 200 -- it will NOT interpolate)
   If either doesn't match a row in the CSV, it tells you what's wrong
   (e.g. lists the valid altitudes for that date) and asks again.

4. On success it writes:
       OUTPUT_DIR/sparta_atm_<date>_<altitude>km.atm
   and prints the full path so you know exactly where to find it.

===============================================================================
WHAT THIS SCRIPT COMPUTES AND WHY (first-principles basis)
===============================================================================
  - nrho   = sum of species number densities (Dalton's law, number-density
             form: total number density of a mixture = sum of the partial
             number densities of each species)
  - frac_i = n_i / nrho, then renormalized a second time to force the
             fractions to sum to EXACTLY 1.000000 (kills floating-point
             division drift)
  - vstream = circular-orbit speed from the vis-viva equation reduced to
              eccentricity = 0:
                v_orb = sqrt(GM_earth / (R_earth + alt))
              recomputed fresh from whatever altitude you query, since a
              stored/looked-up value would only be valid for altitudes
              already present as columns in the CSV.
  - doy_cos/doy_sin = seasonal phase encoding,
              doy_frac = 2*pi*day_of_year/365.25

This script does an EXACT match lookup (date + altitude both must exist
verbatim in the CSV). It does not interpolate or nearest-match altitude --
if the requested altitude isn't on the grid, it errors and shows you the
valid altitudes for that date, per your explicit instruction.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ==============================================================================
# USER CONFIG -- edit these two lines, then just hit Run.
# ==============================================================================
CSV_PATH = r"C:\Summer 26\Task2\output_daily_updated.csv"
OUTPUT_DIR = r"C:\Summer 26\Task2"
# ==============================================================================

# --- Physical constants (IAG 1980 mean spherical Earth, matches the header
#     comments already used in your SPARTA files) ---
GM_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH = 6371.0e3          # m

# Species order locked in by the existing pipeline schema
SPECIES_ORDER = ["O", "N2", "O2", "N", "He", "Ar", "H"]
SPECIES_COLUMNS = {
    "O": "O_mean_m3",
    "N2": "N2_mean_m3",
    "O2": "O2_mean_m3",
    "N": "N_mean_m3",
    "He": "He_mean_m3",
    "Ar": "Ar_mean_m3",
    "H": "H_mean_m3",
}


REQUIRED_COLUMNS = ["Year", "Month", "Day", "alt_km", "F107", "F107A", "Ap_daily", "T_mean_K"] + list(SPECIES_COLUMNS.values())


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Strip incidental whitespace from header names -- a common cause of
    # KeyErrors when a CSV was exported/edited by a different tool than the
    # one that originally defined the schema.
    df.columns = df.columns.str.strip()

    # Build a case-insensitive, whitespace-insensitive lookup so a header
    # like " o_mean" or "O_MEAN" still resolves to "O_mean" instead of
    # crashing -- but we NEVER guess a semantically different column, only
    # correct for formatting.
    normalized = {c.strip().lower(): c for c in df.columns}

    missing = []
    rename_map = {}
    for required in REQUIRED_COLUMNS:
        key = required.strip().lower()
        if required in df.columns:
            continue
        elif key in normalized:
            rename_map[normalized[key]] = required
        else:
            missing.append(required)

    if missing:
        raise ValueError(
            "The CSV is missing expected column(s): "
            f"{missing}\n"
            f"Columns actually found in the file: {list(df.columns)}\n"
            "Fix: either rename the columns in the CSV to match the "
            "expected names above, or update REQUIRED_COLUMNS / "
            "SPECIES_COLUMNS at the top of this script to match your "
            "real header names."
        )

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_row(csv_path: Path, year: int, month: int, day: int, alt_km: float) -> pd.Series:
    df = load_and_validate_csv(csv_path)

    date_mask = (df["Year"] == year) & (df["Month"] == month) & (df["Day"] == day)
    date_df = df[date_mask]

    if date_df.empty:
        raise ValueError(
            f"No data found for date {year:04d}-{month:02d}-{day:02d} in {csv_path}"
        )

    alt_mask = date_df["alt_km"] == alt_km
    row_df = date_df[alt_mask]

    if row_df.empty:
        valid_alts = sorted(date_df["alt_km"].unique().tolist())
        raise ValueError(
            f"Altitude {alt_km} km not found for date "
            f"{year:04d}-{month:02d}-{day:02d}. "
            f"Valid altitudes for this date are: {valid_alts}"
        )

    if len(row_df) > 1:
        # Shouldn't happen given lat/lon are fixed columns, but don't silently
        # pick one if the data has an unexpected duplicate.
        raise ValueError(
            f"Multiple rows matched date {year:04d}-{month:02d}-{day:02d} "
            f"and altitude {alt_km} km — data ambiguity, refusing to guess."
        )

    return row_df.iloc[0]


def compute_fractions(row: pd.Series) -> tuple[dict, float]:
    raw_n = {s: float(row[SPECIES_COLUMNS[s]]) for s in SPECIES_ORDER}
    nrho = sum(raw_n.values())

    frac = {s: raw_n[s] / nrho for s in SPECIES_ORDER}

    # Renormalize to kill floating-point drift (matches the ~0.01% gap
    # already documented in the pipeline's prior debugging).
    frac_sum = sum(frac.values())
    frac = {s: frac[s] / frac_sum for s in SPECIES_ORDER}

    return frac, nrho


def compute_vstream(alt_km: float) -> float:
    alt_m = alt_km * 1000.0
    v_orb = math.sqrt(GM_EARTH / (R_EARTH + alt_m))
    return v_orb


def compute_doy_phase(year: int, month: int, day: int) -> tuple[int, float, float]:
    doy = datetime(year, month, day).timetuple().tm_yday
    doy_frac = 2.0 * math.pi * doy / 365.25
    return doy, math.cos(doy_frac), math.sin(doy_frac)


def build_atm_text(row: pd.Series, year, month, day, alt_km, frac, nrho, v_orb, doy, doy_cos, doy_sin) -> str:
    temp = float(row["T_mean_K"])
    f107 = float(row["F107"])
    f107a = float(row["F107A"])
    ap_daily = float(row["Ap_daily"])

    lines = []
    lines.append("# " + "=" * 74)
    lines.append("# SPARTA ATMOSPHERE BLOCK")
    lines.append(f"# Model        : NRLMSIS-00 (direct CSV match, exact date/altitude lookup)")
    lines.append(f"# Date         : {year:04d}-{month:02d}-{day:02d}   |   Altitude: {alt_km:.1f} km")
    lines.append(
        f"# Query        : Ap_daily={ap_daily:g}   F107={f107:g}   F107A={f107a:g}   "
        f"doy_cos={doy_cos:.4f}   doy_sin={doy_sin:.4f}"
    )
    lines.append("#")
    lines.append("# PHYSICAL ASSUMPTIONS:")
    lines.append("#   \u2022 Circular orbit (eccentricity = 0)")
    lines.append("#   \u2022 v_orb = sqrt(GM_earth / (R_earth + alt))")
    lines.append(f"#       GM_earth = {GM_EARTH:.8e} m\u00b3/s\u00b2")
    lines.append(f"#       R_earth  = {R_EARTH/1000:.2f} km  (spherical mean radius, IAG 1980)")
    lines.append("#   \u2022 No atmospheric co-rotation correction (~460 m/s at equator)")
    lines.append("#   \u2022 No J2 oblateness correction (~0.1 % at LEO)")
    lines.append("#   \u2022 'global temp' = neutral kinetic temperature at altitude")
    lines.append("#       NOT the exospheric temperature (tinf / T_exo)")
    lines.append("#   \u2022 Species fractions = number-density fractions (NOT mass fractions)")
    lines.append("#       frac_i = n_i / nrho   where nrho = \u03a3 n_i over all species")
    lines.append("#   \u2022 vstream = (0, 0, -v_orb): satellite frame; flow arrives from +z")
    lines.append("# " + "=" * 74)
    lines.append("")
    lines.append(f"global      nrho    {nrho:.6e}")
    lines.append(f"global      temp    {temp:.2f}")
    lines.append(f"global      vstream 0.0 0.0 -{v_orb:.2f}    # circular orbit at {alt_km:.2f} km")
    lines.append("")
    for s in SPECIES_ORDER:
        lines.append(f"mixture     air     {s:<3} frac {frac[s]:.6f}")
    lines.append(f"# fraction sum = {sum(frac.values()):.6f}")
    lines.append("")
    return "\n".join(lines)


def generate(csv_path: str, date_str: str, alt_km: float, output_dir: str) -> Path:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year, month, day = dt.year, dt.month, dt.day

    row = load_row(csv_path, year, month, day, alt_km)
    frac, nrho = compute_fractions(row)
    v_orb = compute_vstream(alt_km)
    doy, doy_cos, doy_sin = compute_doy_phase(year, month, day)

    text = build_atm_text(row, year, month, day, alt_km, frac, nrho, v_orb, doy, doy_cos, doy_sin)

    out_path = output_dir / f"sparta_atm_{date_str}_{alt_km:.0f}km.atm"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def prompt_for_date() -> str:
    """Keep asking until the user gives a string that actually parses as
    YYYY-MM-DD. Fail fast and re-ask here rather than letting a bad string
    reach pandas filtering, where it would just silently match zero rows."""
    while True:
        raw = input("Enter query date (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return raw
        except ValueError:
            print(f"  '{raw}' isn't a valid YYYY-MM-DD date. Try again.")


def prompt_for_altitude() -> float:
    while True:
        raw = input("Enter altitude in km (must be exact, e.g. 200): ").strip()
        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' isn't a number. Try again.")


def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        print(f"ERROR: CSV_PATH does not point to a real file:\n  {csv_path}")
        print("Edit CSV_PATH at the top of this script and re-run.")
        sys.exit(1)

    print("=" * 60)
    print("SPARTA Atmosphere Block Generator")
    print(f"CSV source : {CSV_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")
    print("=" * 60)

    while True:
        date_str = prompt_for_date()
        alt_km = prompt_for_altitude()

        try:
            out_path = generate(CSV_PATH, date_str, alt_km, OUTPUT_DIR)
        except ValueError as e:
            # Wrong date, wrong altitude, or ambiguous match -- explain
            # exactly what went wrong (including valid alternatives) and
            # let the user try a different query instead of exiting.
            print(f"\nERROR: {e}\n")
            again = input("Try a different date/altitude? (y/n): ").strip().lower()
            if again != "y":
                sys.exit(1)
            continue

        print(f"\nDone. Wrote: {out_path}")

        again = input("\nGenerate another? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
