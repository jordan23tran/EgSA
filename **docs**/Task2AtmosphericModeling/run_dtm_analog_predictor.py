"""
run_dtm_analog_predictor.py
===========================
Purpose
-------
Builds (or loads from cache) the DTM-2020 kNN analog predictor from the
SWAMI MCM output CSV, then launches an interactive terminal query session.

The predictor outputs:
  • Total mass density [kg/m³] and neutral temperature [K]
  • Per-species number densities [m⁻³]: O, N2, O2, He, H, N
    (Argon is NOT modeled by DTM-2020; it is set to 0.0 in the SPARTA block
     with an explicit warning note)
  • A copy-paste–ready SPARTA atmosphere block

Dependency chain
-----------------
  1. SWAMI MCM (WSL Ubuntu)  → mcm_results_all_altitudes.csv
  2. THIS SCRIPT             → dtm2020_analog_predictor.pkl
  3. (use pkl for queries)

Auto-rebuild logic
------------------
  Same as run_msis_analog_predictor.py:
  • Missing pkl → rebuild from MCM CSV.
  • Stale pkl (schema v1) → auto-delete and rebuild.
  • Pkl up-to-date → load and go straight to query.

Unit conversion performed by this script
-----------------------------------------
The SWAMI MCM outputs species partial mass densities in g/cm³:
  d_O, d_N2, d_O2, d_N, d_He, d_H  [g/cm³]

These are converted to number densities [m⁻³] using:
  n_X = d_X [g/cm³] × N_A [mol⁻¹] / M_X [g/mol] × 1e6 [cm³/m³]

where N_A = 6.02214076e23 mol⁻¹ and M_X is the molar mass.

This formula follows directly from:
  n_X = (ρ_X [kg/m³]) / (m_X [kg/molecule])
where ρ_X [kg/m³] = d_X [g/cm³] × 1000 [g/kg] / 1e-6 [m³/cm³]
and m_X [kg] = M_X [g/mol] / N_A / 1000.

Inputs
------
  mcm_results_all_altitudes.csv   (from SWAMI MCM — already in WSL)
    Required columns:
      Date, Altitude_km, dens [g/cm³], temp [K],
      d_H, d_He, d_O, d_N2, d_O2, d_N [g/cm³],
      F107, F107m, kp1, kp2, Ap_daily

Outputs
-------
  dtm2020_analog_predictor.pkl
  sparta_block_DTM_2020_<alt>km.sparta  (if generated in session)
  analogs_DTM_2020_<alt>km.csv          (if saved in session)

Environment
-----------
  WSL Ubuntu 20.04+, Python 3.10+
  Required packages:  numpy pandas matplotlib
      pip install numpy pandas matplotlib

  The SWAMI MCM itself runs on Linux and was run separately to produce
  mcm_results_all_altitudes.csv.  This predictor script only reads that CSV;
  it does NOT invoke the MCM executable again.

How to access WSL and run this script
--------------------------------------
  Step 1 — Open WSL Ubuntu terminal
    In Windows, press Win+R, type:
       wsl
    Press Enter.  A Linux bash prompt appears.

  Step 2 — Navigate to the script directory
       cd /home/jordan23tran/mcm

  Step 3 — Confirm the MCM CSV exists
       ls output/mcm_results_all_altitudes.csv

  Step 4 — Run the script
       python3 run_dtm_analog_predictor.py

  Step 5 — On first run, the predictor is built and saved as a .pkl file.
    Subsequent runs load the pkl instantly; no MCM re-run is needed.

    If you want to force a rebuild (e.g. after adding new MCM data):
       rm dtm2020_analog_predictor.pkl
       python3 run_dtm_analog_predictor.py

  Step 6 — Follow the terminal prompts to enter a query and optionally
    generate a SPARTA block.

    Note: msis_dtm_analog_predictor.py must also be in this directory.
    Copy it from Windows if needed:
       cp /mnt/c/Users/jorda/OneDrive*/Documents/"Summer 26"/msis_dtm_analog_predictor.py .

DTM-2020 note on Argon
-----------------------
DTM-2020 does not model argon.  The SPARTA block will show:
  mixture  air  Ar  frac  0.000000
with a warning note in the header.  At 200–300 km, Ar constitutes roughly
0.003–0.02 % of number density (per NRLMSIS-00); this omission introduces
less than 0.02 % error in nrho and is acceptable for VLEO simulations.
"""

import os
import sys
import numpy as np
import pandas as pd

from msis_dtm_analog_predictor import (
    AtmosphericAnalogPredictor,
    build_feature_df_dtm,
    terminal_query,
    PKL_SCHEMA_VERSION,
)

# ============================================================
# USER SETTINGS  ← edit these paths before running
# ============================================================

MCM_CSV       = '/home/jordan23tran/mcm/output/mcm_results_all_altitudes.csv'
PREDICTOR_PKL = '/home/jordan23tran/mcm/dtm2020_analog_predictor.pkl'
OUTPUT_DIR    = '/home/jordan23tran/mcm/output'

# ============================================================
# SPECIES UNIT CONVERSION CONSTANTS
# ============================================================

_N_A = 6.02214076e23   # Avogadro's number [mol⁻¹]

# Molar masses [g/mol] — used to convert g/cm³ → m⁻³
_M = {
    'O':  15.9994,
    'N2': 28.0134,
    'O2': 31.9988,
    'N':  14.0067,
    'He':  4.0026,
    'H':   1.00794,
}

# MCM column name → predictor species name
_MCM_COL = {
    'd_O'  : 'O',
    'd_N2' : 'N2',
    'd_O2' : 'O2',
    'd_N'  : 'N',
    'd_He' : 'He',
    'd_H'  : 'H',
}

def _gcm3_to_m3(rho_gcm3: np.ndarray, species: str) -> np.ndarray:
    """
    Convert partial mass density [g/cm³] to number density [m⁻³].

    First principles:
      n = ρ [g/cm³] × N_A [mol⁻¹] / M [g/mol] × 1e6 [cm³/m³]

    The factor 1e6 converts cm³ in the denominator to m³
    (1 m³ = 1e6 cm³, so dividing by cm³ and multiplying by 1e6 gives m⁻³).
    """
    return rho_gcm3 * _N_A / _M[species] * 1e6

# ============================================================
# LOAD MCM CSV
# ============================================================

print(f'Loading MCM CSV: {MCM_CSV}')
if not os.path.exists(MCM_CSV):
    sys.exit(
        f'ERROR: {MCM_CSV} not found.\n'
        'Confirm the SWAMI MCM was run and output saved to this path.'
    )

mcm = pd.read_csv(MCM_CSV)
mcm['Date'] = pd.to_datetime(mcm['Date'])

ALTITUDES_KM = sorted(mcm['Altitude_km'].unique())
print(f'  {len(mcm):,} rows  |  dates: {mcm["Date"].min().date()} → {mcm["Date"].max().date()}')
print(f'  Altitudes: {ALTITUDES_KM} km')

# Quick unit verification (compare computed n_O vs CSV O_number_density_m3)
row0 = mcm.iloc[0]
n_O_check  = _gcm3_to_m3(row0['d_O'], 'O')
n_O_csv    = row0.get('O_number_density_m3', np.nan)
rel_err    = abs(n_O_check - n_O_csv) / n_O_csv if not np.isnan(n_O_csv) else 0
print(f'  Unit check: n_O computed={n_O_check:.4e} m⁻³  CSV={n_O_csv:.4e} m⁻³'
      f'  rel_err={rel_err:.2e}'
      + ('  ✓' if rel_err < 0.005 else '  ⚠ WARNING — check conversion'))

# ============================================================
# AUTO-DETECT & LOAD OR REBUILD THE PREDICTOR PKL
# ============================================================

predictor = None

if os.path.exists(PREDICTOR_PKL):
    try:
        predictor = AtmosphericAnalogPredictor.load(PREDICTOR_PKL)
    except (ValueError, AttributeError, Exception) as e:
        print(f'\n[AUTO-REBUILD] Stale or incompatible pkl detected:\n  {e}')
        print('Deleting old pkl and rebuilding from MCM CSV …\n')
        os.remove(PREDICTOR_PKL)
        predictor = None

if predictor is None:
    print('Building DTM-2020 analog predictor from MCM CSV …')

    # ── Feature DataFrame (one row per unique date) ─────────────────────
    feature_df = build_feature_df_dtm(mcm)
    dates      = feature_df.index   # pd.DatetimeIndex, one date per row

    # ── Pivot total density and temperature (g/cm³ → kg/m³) ────────────
    # 1 g/cm³ = 1000 kg/m³  (1e3 factor, not 1e6 — see notes in docstring)
    dens_pivot = (
        mcm.pivot_table(index='Date', columns='Altitude_km', values='dens')
        .reindex(columns=ALTITUDES_KM).sort_index()
        .reindex(dates)
    ) * 1e3   # g/cm³ → kg/m³

    temp_pivot = (
        mcm.pivot_table(index='Date', columns='Altitude_km', values='temp')
        .reindex(columns=ALTITUDES_KM).sort_index()
        .reindex(dates)
    )

    # ── Pivot and convert each species  d_X [g/cm³] → n_X [m⁻³] ───────
    species_dict = {
        'total_density': dens_pivot.values,
        'temperature'  : temp_pivot.values,
    }

    for mcm_col, sp_name in _MCM_COL.items():
        pivot = (
            mcm.pivot_table(index='Date', columns='Altitude_km', values=mcm_col)
            .reindex(columns=ALTITUDES_KM).sort_index()
            .reindex(dates)
        )
        species_dict[sp_name] = _gcm3_to_m3(pivot.values, sp_name)

    # Note: Ar is NOT in DTM-2020; it will be 0.0 in the SPARTA block.
    # The model_notes string below is displayed in the SPARTA block header.

    # ── Fit and save ─────────────────────────────────────────────────────
    p = AtmosphericAnalogPredictor(model_name='DTM-2020')
    p.fit(pd.DatetimeIndex(dates), feature_df, species_dict, ALTITUDES_KM)
    p.save(PREDICTOR_PKL)
    predictor = p
    print(f'\nPredictor saved → {PREDICTOR_PKL}')

# ============================================================
# LAUNCH INTERACTIVE TERMINAL QUERY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

DTM_NOTES = (
    'Ar (argon) is NOT modeled by DTM-2020; Ar fraction = 0.000000.\n'
    'At 200-300 km Ar is ~0.003-0.02 % of number density (negligible).\n'
    'Total density from DTM-2020 converted: dens [g/cm³] × 1e3 → kg/m³.\n'
    'Species densities from d_X [g/cm³] × N_A / M_X × 1e6 → n_X [m⁻³].'
)

terminal_query(predictor, output_dir=OUTPUT_DIR, model_notes=DTM_NOTES)
