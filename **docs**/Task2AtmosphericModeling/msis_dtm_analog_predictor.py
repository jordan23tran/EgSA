"""
msis_dtm_analog_predictor.py
============================
Purpose
-------
Shared kNN analog-forecasting engine and SPARTA output utilities for the
NRLMSIS-00 and DTM-2020 atmospheric models.

This file is a pure library — do NOT run it directly.

It is imported by:
  • run_msis_analog_predictor.py  (Windows — NRLMSIS-00 predictor)
  • run_dtm_analog_predictor.py   (WSL Ubuntu — DTM-2020 / SWAMI MCM predictor)
  • run_jb2008.py                 (Windows — JB2008 hybrid, uses NRLMSIS species)

The library provides:

  AtmosphericAnalogPredictor      kNN predictor class (all species + temp)
  build_feature_df_msis()         feature builder for NRLMSIS-00
  build_feature_df_dtm()          feature builder for DTM-2020
  terminal_query()                interactive terminal loop with SPARTA output
  generate_sparta_block()         format a copy-paste SPARTA atmosphere block
  circular_orbital_velocity()     v = sqrt(GM / (R+alt))
  mean_molecular_mass_kg()        Σ(n_i × m_i) / Σ(n_i)
  SPARTA_SPECIES_ORDER            ['O','N2','O2','N','He','Ar','H']
  PARTICLE_MASS_KG                per-particle mass for each species [kg]
  PKL_SCHEMA_VERSION              current pkl schema version (int)

PKL Schema Version
------------------
PKL_SCHEMA_VERSION = 2

Version 2 adds: per-species number density databases, neutral temperature
database, SPARTA block generation.
Version 1 pkl files (density + atomic-O only, no _PKL_VERSION attribute)
are auto-detected by runner scripts, deleted, and rebuilt.

Environment
-----------
  • Windows 10/11 (for MSIS and JB2008 runners), Python 3.10+
  • WSL Ubuntu 20.04+ (for DTM runner), Python 3.10+
  Required packages: numpy, pandas, matplotlib
  Place this file in the same directory as whichever runner imports it.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Schema version ─────────────────────────────────────────────────────────────
PKL_SCHEMA_VERSION = 2

# ── SPARTA species order (must match your <name>.species file) ─────────────────
SPARTA_SPECIES_ORDER = ['O', 'N2', 'O2', 'N', 'He', 'Ar', 'H']

# ── Atomic / molecular masses [g/mol] ─────────────────────────────────────────
_MOLAR_MASS_G_PER_MOL = {
    'O':  15.9994,
    'N2': 28.0134,
    'O2': 31.9988,
    'N':  14.0067,
    'He':  4.0026,
    'Ar': 39.948,
    'H':   1.00794,
}
_N_A = 6.02214076e23   # Avogadro [mol⁻¹]

# Mass per particle [kg]:  m = M [g/mol] / N_A [mol⁻¹] / 1000 [g/kg]
PARTICLE_MASS_KG = {k: v / _N_A / 1000.0 for k, v in _MOLAR_MASS_G_PER_MOL.items()}


# =============================================================================
# PHYSICAL UTILITY FUNCTIONS
# =============================================================================

def circular_orbital_velocity(alt_km: float) -> float:
    """
    Circular orbital speed at altitude alt_km above Earth's surface [m/s].

        v = sqrt(GM_earth / (R_earth + alt))

    Physical assumptions (document these in SPARTA file comments):
      • Perfectly circular orbit  (eccentricity = 0)
      • Spherical Earth           (R_earth = 6371.0 km, IAG 1980 mean radius)
      • Standard GM_earth         = 3.986004418e14 m³/s²
      • No atmospheric co-rotation correction
          (~460 m/s at equator; ~6 % of orbital speed; increases nrho by ~3 %)
      • No J2 oblateness correction  (~0.1 % error at LEO)
      • No drag history or altitude decay

    At 250 km this gives v ≈ 7784 m/s.  The SPARTA vstream convention used
    here is (0, 0, −v_orb): satellite frame, flow arriving from +z.
    """
    GM_earth = 3.986004418e14  # m³/s²
    R_earth  = 6.371e6         # m
    return float(np.sqrt(GM_earth / (R_earth + alt_km * 1e3)))


def mean_molecular_mass_kg(species_num_den: dict) -> float:
    """
    Mass-weighted mean molecular mass from species number densities.

        m_mean = Σ(n_i × m_i) / Σ(n_i)

    Used by run_jb2008.py to convert JB2008 total mass density [kg/m³] into
    total number density [m⁻³] using NRLMSIS-00 species composition.

    Parameters
    ----------
    species_num_den : {species_name: float [m⁻³]}

    Returns
    -------
    float  [kg / molecule]
    """
    n_total = sum(species_num_den.values())
    if n_total < 1e-30:
        return PARTICLE_MASS_KG['N2']   # fallback: shouldn't happen in practice
    m_num = sum(
        species_num_den[s] * PARTICLE_MASS_KG[s]
        for s in species_num_den if s in PARTICLE_MASS_KG
    )
    return m_num / n_total


# =============================================================================
# SPARTA BLOCK GENERATOR
# =============================================================================

def generate_sparta_block(
    model_name: str,
    alt_km: float,
    result: dict,
    query_features: dict = None,
    output_dir: str = '.',
    save: bool = True,
    model_notes: str = '',
) -> str:
    """
    Format and optionally save a SPARTA atmosphere input block.

    The generated text replaces the global / mixture lines in a .sparta input
    file.  It is ready to copy-paste.

    Parameters
    ----------
    model_name    : str   e.g. 'NRLMSIS-00' or 'DTM-2020'
    alt_km        : float altitude [km]
    result        : dict  output of AtmosphericAnalogPredictor.predict()
    query_features: dict  raw query features (annotated in the block header)
    output_dir    : str   directory for the saved .sparta snippet file
    save          : bool  write sparta_block_<model>_<alt>km.sparta
    model_notes   : str   optional extra assumptions appended to the header
                          (one note per line — each becomes a bullet point)

    Returns
    -------
    str   complete SPARTA block text
    """
    v_orb    = circular_orbital_velocity(alt_km)
    preds    = result.get('outputs_pred', {})
    nrho     = result.get('nrho', float('nan'))
    temp     = preds.get('temperature', float('nan'))
    fracs    = result.get('species_fracs', {s: 0.0 for s in SPARTA_SPECIES_ORDER})
    frac_sum = sum(fracs.values())
    k_used   = result.get('k', '?')
    method   = result.get('method', '?')

    q_str = ('  '.join(f"{k}={v:.4g}" for k, v in sorted(query_features.items()))
             if query_features else '(no query summary)')

    lines = [
        '# ============================================================',
        '# SPARTA ATMOSPHERE BLOCK',
        f'# Model       : {model_name} Analog Predictor',
        f'# Altitude    : {alt_km:.1f} km  |  k={k_used}  |  method={method}',
        f'# Query       : {q_str}',
        '#',
        '# PHYSICAL ASSUMPTIONS:',
        '#   • Circular orbit (eccentricity = 0)',
        '#   • v_orb = sqrt(GM_earth / (R_earth + alt))',
        '#       GM_earth = 3.986004418e14 m³/s²',
        '#       R_earth  = 6371.0 km  (spherical mean radius, IAG 1980)',
        '#   • No atmospheric co-rotation correction (~460 m/s at equator)',
        '#   • No J2 oblateness correction (~0.1 % at LEO)',
        "#   • 'global temp' = neutral kinetic temperature at altitude",
        '#       NOT the exospheric temperature (tinf / T_exo)',
        '#   • Species fractions = number-density fractions (NOT mass fractions)',
        '#       frac_i = n_i / n_total   where n_total = Σ n_i over all species',
        '#   • vstream = (0, 0, −v_orb): satellite frame; flow arrives from +z',
    ]

    if model_notes:
        for note in model_notes.strip().split('\n'):
            lines.append(f'#   • {note.strip()}')

    lines += [
        '# ============================================================',
        '',
        f'global          nrho    {nrho:.6e}',
        f'global          temp    {temp:.2f}',
        (f'global          vstream 0.0  0.0  {-v_orb:.2f}'
         f'   # circular orbit at {alt_km:.1f} km'),
        '',
        f'mixture         air  O  N2  O2  N  He  Ar  H  temp  {temp:.2f}',
    ]

    for sp in SPARTA_SPECIES_ORDER:
        f = fracs.get(sp, 0.0)
        lines.append(f'mixture         air  {sp:<4s} frac  {f:.6f}')

    ok_str = '  ✓' if abs(frac_sum - 1.0) < 1e-3 else '  ← WARNING: does not sum to 1'
    lines.append(f'# fraction sum = {frac_sum:.6f}{ok_str}')
    if abs(frac_sum - 1.0) >= 1e-3:
        lines.append(
            '# Check that all species arrays were provided to predictor.fit().'
        )

    text = '\n'.join(lines)
    bar  = '─' * 66
    print(f'\n{bar}')
    print('  SPARTA ATMOSPHERE BLOCK — copy-paste into your .sparta file')
    print(bar)
    print(text)
    print(bar + '\n')

    if save:
        safe  = model_name.replace(' ', '_').replace('-', '_').replace('+', '_')
        fname = f'sparta_block_{safe}_{int(alt_km)}km.sparta'
        path  = os.path.join(output_dir, fname)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
        print(f'  Saved SPARTA block → {path}\n')

    return text


# =============================================================================
# FEATURE BUILDERS
# =============================================================================

def build_feature_df_msis(sw_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature DataFrame for NRLMSIS-00 analog predictor.

    Input  : daily space-weather DataFrame indexed by date.
             Required columns: F107, F107A, Ap_daily
    Output : DataFrame  (same index)
             columns: F107, F107A, Ap_daily, doy_sin, doy_cos
    """
    idx = pd.DatetimeIndex(sw_daily_df.index)
    doy = idx.day_of_year.astype(float)
    return pd.DataFrame({
        'F107'    : sw_daily_df['F107'].values.astype(float),
        'F107A'   : sw_daily_df['F107A'].values.astype(float),
        'Ap_daily': sw_daily_df['Ap_daily'].values.astype(float),
        'doy_sin' : np.sin(2.0 * np.pi * doy / 365.25),
        'doy_cos' : np.cos(2.0 * np.pi * doy / 365.25),
    }, index=sw_daily_df.index)


def build_feature_df_dtm(mcm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature DataFrame for DTM-2020 analog predictor.

    Input  : MCM results DataFrame (from mcm_results_all_altitudes.csv).
             One row per (date, altitude); SW indices are date-constant.
             Required columns: Date, F107, F107m, kp1, kp2, Ap_daily
    Output : DataFrame indexed by Date (one row per unique date)
             columns: F107, F107m, kp1, kp2, Ap_daily, doy_sin, doy_cos
    """
    daily = (
        mcm_df.drop_duplicates(subset='Date')
        [['Date', 'F107', 'F107m', 'kp1', 'kp2', 'Ap_daily']]
        .copy()
    )
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.set_index('Date').sort_index()

    doy = daily.index.day_of_year.astype(float)
    daily['doy_sin'] = np.sin(2.0 * np.pi * doy / 365.25)
    daily['doy_cos'] = np.cos(2.0 * np.pi * doy / 365.25)

    return daily[['F107', 'F107m', 'kp1', 'kp2', 'Ap_daily', 'doy_sin', 'doy_cos']]


# =============================================================================
# ANALOG PREDICTOR CLASS
# =============================================================================

class AtmosphericAnalogPredictor:
    """
    k-nearest-neighbour (kNN) analog forecaster for atmospheric properties.

    Works for both NRLMSIS-00 and DTM-2020: pass whichever species are
    available via ``species_dict`` to ``fit()``.

    Core idea (first principles)
    ----------------------------
    The solar-terrestrial system is highly non-linear.  Instead of fitting a
    parametric model, we search the historical record for the k past days
    whose solar/geomagnetic conditions most closely match today's (Euclidean
    distance in normalised feature space).  The atmosphere on those days is
    a direct physics-based estimate of today's atmosphere — no regression
    assumptions required.

    Stored attributes (pkl schema v2)
    ----------------------------------
    species_db       dict {name: ndarray (n_days, n_alts)}
                     Keys: 'total_density' [kg/m³], 'temperature' [K],
                           'O', 'N2', 'O2', 'He', 'Ar', 'H', 'N'  [m⁻³]
                     Not every key is required; keys present depend on model.
    dates            ndarray of np.datetime64, shape (n_days,)
    features_raw     ndarray (n_days, n_features) — unscaled feature vectors
    features_sc      ndarray (n_days, n_features) — z-score normalised
    feature_cols     list[str]
    altitudes_km     list[float]
    available_species list[str] — species keys from SPARTA_SPECIES_ORDER present
    model_name       str
    _PKL_VERSION     int = PKL_SCHEMA_VERSION
    """

    _PKL_VERSION = PKL_SCHEMA_VERSION

    def __init__(self, model_name: str = 'NRLMSIS-00'):
        self.model_name        = model_name
        self._PKL_VERSION      = PKL_SCHEMA_VERSION
        self.feature_cols      = None
        self.altitudes_km      = None
        self.dates             = None
        self.features_raw      = None
        self.features_sc       = None
        self._mean             = None
        self._std              = None
        self.species_db        = None
        self.available_species = []
        self._fitted           = False

    # ── fit ───────────────────────────────────────────────────────────────
    def fit(
        self,
        dates,
        feature_df: pd.DataFrame,
        species_dict: dict,
        altitudes_km: list,
    ) -> None:
        """
        Train the analog predictor on historical data.

        Parameters
        ----------
        dates        : array-like of datetime-like values (length n_days)
        feature_df   : DataFrame (n_days × n_features) with feature columns.
                       Rows with any NaN are dropped automatically.
        species_dict : dict {name: ndarray (n_days, n_alts)}
                       Required keys: 'total_density' [kg/m³], 'temperature' [K]
                       Recommended  : 'O', 'N2', 'O2', 'He', 'Ar', 'H', 'N' [m⁻³]
                       Missing species get fraction = 0 in the SPARTA block.
        altitudes_km : list of floats, length n_alts (must match 2nd dim of arrays)
        """
        for req in ('total_density', 'temperature'):
            if req not in species_dict:
                raise ValueError(
                    f"species_dict must contain '{req}'. "
                    f"Got: {list(species_dict.keys())}"
                )

        valid = ~feature_df.isna().any(axis=1).values
        n_bad = (~valid).sum()
        if n_bad:
            print(f'[{self.model_name}.fit] Dropping {n_bad} rows with NaN features.')

        self.feature_cols = list(feature_df.columns)
        self.altitudes_km = [float(a) for a in altitudes_km]
        self.dates        = np.array(pd.DatetimeIndex(dates))[valid]
        self.features_raw = feature_df.values[valid].astype(float)

        # z-score normalise so that features with different scales contribute
        # equally to the Euclidean distance (prevents F10.7 swamping Ap)
        self._mean = self.features_raw.mean(axis=0)
        self._std  = self.features_raw.std(axis=0)
        self._std[self._std < 1e-12] = 1.0   # guard against constant features
        self.features_sc = (self.features_raw - self._mean) / self._std

        self.species_db = {
            name: arr[valid].astype(float)
            for name, arr in species_dict.items()
        }
        self.available_species = [
            s for s in SPARTA_SPECIES_ORDER if s in self.species_db
        ]

        self._fitted = True
        print(f'[{self.model_name} AnalogPredictor.fit]'
              f'  {len(self.dates)} days  |  {len(self.feature_cols)} features'
              f'  |  alts = {self.altitudes_km} km')
        print(f'  Feature cols    : {self.feature_cols}')
        print(f'  Species stored  : {list(self.species_db.keys())}')
        print(f'  SPARTA species  : {self.available_species}')

    # ── internal helpers ───────────────────────────────────────────────────
    def _normalise(self, query_dict: dict) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in query_dict]
        if missing:
            raise ValueError(
                f'Query is missing features: {missing}\n'
                f'Expected: {self.feature_cols}'
            )
        raw = np.array([query_dict[c] for c in self.feature_cols], dtype=float)
        return (raw - self._mean) / self._std

    def _distances(self, q_sc: np.ndarray) -> np.ndarray:
        diff = self.features_sc - q_sc
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    def _outputs_at_alt(self, indices: np.ndarray, alt_km: float) -> dict:
        """
        Extract database values at alt_km for the given row indices.
        Interpolates linearly if alt_km is not an exact stored altitude.

        Returns dict {name: ndarray (len(indices),)}
        """
        alts  = np.array(self.altitudes_km)
        exact = np.where(np.abs(alts - alt_km) < 1e-6)[0]
        out   = {}

        if exact.size > 0:
            ai = int(exact[0])
            for name, db in self.species_db.items():
                out[name] = db[indices, ai]
        else:
            if alt_km < alts.min() or alt_km > alts.max():
                print(
                    f'[{self.model_name}] WARNING: {alt_km} km is outside the '
                    f'trained range [{alts.min()}, {alts.max()}] km — extrapolating.'
                )
            for name, db in self.species_db.items():
                out[name] = np.array([
                    np.interp(alt_km, alts, db[i, :]) for i in indices
                ])
        return out

    # ── predict ───────────────────────────────────────────────────────────
    def predict(
        self,
        query_features: dict,
        alt_km: float,
        k: int = 20,
        method: str = 'inverse_distance',
    ) -> dict:
        """
        Predict atmospheric state at alt_km for the given query features.

        Parameters
        ----------
        query_features : dict {feature_name: float}
        alt_km         : float
        k              : int   number of analog days
        method         : 'inverse_distance' | 'mean' | 'best'
                         inverse_distance: w_i = 1/dist_i (best for smooth output)
                         mean: equal weights (simple average)
                         best: use only the single closest analog

        Returns
        -------
        dict with keys:
          outputs_pred     {name: float}   weighted prediction for each stored output
          outputs_sigma    {name: float}   1-σ spread across k analogs
          nrho             float [m⁻³]    Σ species number densities → for SPARTA
          species_fracs    {sp: float}     number-density fraction for each SPARTA species
          available_species list[str]      species present in this predictor
          analog_vals      {name: ndarray} raw k-analog values (for plots)
          analog_dates     ndarray
          distances        ndarray
          weights          ndarray
          k, method, alt_km
          density_pred, density_sigma, atomic_O_pred, atomic_O_sigma  (aliases)
        """
        if not self._fitted:
            raise RuntimeError('Call .fit() before .predict()')

        k       = min(k, len(self.dates))
        q_sc    = self._normalise(query_features)
        dists   = self._distances(q_sc)
        top_idx = np.argsort(dists)[:k]
        top_d   = dists[top_idx]

        # Combination weights
        eps = 1e-12
        if method == 'best':
            w = np.zeros(k); w[0] = 1.0
        elif method == 'mean':
            w = np.ones(k) / k
        elif method == 'inverse_distance':
            w = 1.0 / (top_d + eps)
            w /= w.sum()
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose: 'inverse_distance', 'mean', 'best'."
            )

        analog_vals = self._outputs_at_alt(top_idx, alt_km)

        # Weighted predictions and 1-σ spread
        preds  = {name: float(np.dot(w, vals)) for name, vals in analog_vals.items()}
        sigmas = {name: float(np.std(vals))     for name, vals in analog_vals.items()}

        # SPARTA-ready values
        num_sp  = [s for s in SPARTA_SPECIES_ORDER if s in preds]
        nrho    = sum(preds[s] for s in num_sp)
        fracs   = {}
        for s in SPARTA_SPECIES_ORDER:
            fracs[s] = (preds[s] / nrho if (s in preds and nrho > 0) else 0.0)

        return {
            'outputs_pred'    : preds,
            'outputs_sigma'   : sigmas,
            'nrho'            : nrho,
            'species_fracs'   : fracs,
            'available_species': num_sp,
            'analog_vals'     : analog_vals,
            'analog_dates'    : self.dates[top_idx],
            'distances'       : top_d,
            'weights'         : w,
            'k'               : k,
            'method'          : method,
            'alt_km'          : alt_km,
            # ── backward-compat aliases ────────────────────────────────────
            'density_pred'    : preds.get('total_density', np.nan),
            'density_sigma'   : sigmas.get('total_density', np.nan),
            'atomic_O_pred'   : preds.get('O', np.nan),
            'atomic_O_sigma'  : sigmas.get('O', np.nan),
            'density_analogs' : analog_vals.get('total_density', np.zeros(k)),
            'atomic_O_analogs': analog_vals.get('O', np.zeros(k)),
        }

    # ── rank_analogs ──────────────────────────────────────────────────────
    def rank_analogs(
        self,
        query_features: dict,
        alt_km: float,
        top_n: int = None,
    ) -> pd.DataFrame:
        """
        Return a ranked DataFrame of the top_n closest historical analog days.

        Columns: Rank, Date, Distance, <feature cols>, <all outputs at alt_km>
        """
        if not self._fitted:
            raise RuntimeError('Call .fit() first.')

        n     = len(self.dates) if top_n is None else min(top_n, len(self.dates))
        q_sc  = self._normalise(query_features)
        dists = self._distances(q_sc)
        idx   = np.argsort(dists)[:n]
        alts  = np.array(self.altitudes_km)

        rows = []
        for rank, i in enumerate(idx, start=1):
            row = {
                'Rank'    : rank,
                'Date'    : pd.Timestamp(self.dates[i]),
                'Distance': float(dists[i]),
            }
            for ci, col in enumerate(self.feature_cols):
                row[col] = float(self.features_raw[i, ci])
            for name, db in self.species_db.items():
                val  = float(np.interp(alt_km, alts, db[i, :]))
                unit = ('kg/m³' if name == 'total_density'
                        else 'K'   if name == 'temperature'
                        else 'm⁻³')
                row[f'{name} [{unit}]'] = val
            rows.append(row)
        return pd.DataFrame(rows)

    # ── plots ─────────────────────────────────────────────────────────────
    def plot_analog_bars(
        self,
        query_features: dict,
        alt_km: float,
        k: int = 20,
        output_dir: str = '.',
    ) -> None:
        """
        Two-panel horizontal bar chart: total mass density and neutral temperature
        for the k closest analog days, with the weighted prediction overlaid.
        """
        res     = self.predict(query_features, alt_km, k=k)
        labels  = [str(pd.Timestamp(d))[:10] for d in res['analog_dates']]
        colours = plt.cm.viridis_r(np.linspace(0.1, 0.9, k))
        avs     = res['analog_vals']

        pairs = [
            (avs.get('total_density', np.zeros(k)),
             res['outputs_pred'].get('total_density', np.nan),
             res['outputs_sigma'].get('total_density', np.nan),
             'Total Mass Density', 'kg/m³'),
            (avs.get('temperature', np.zeros(k)),
             res['outputs_pred'].get('temperature', np.nan),
             res['outputs_sigma'].get('temperature', np.nan),
             'Neutral Temperature', 'K'),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, k * 0.38)))
        for ax, (vals, pred, sig, title, unit) in zip(axes, pairs):
            y = np.arange(k)
            ax.barh(y, vals, color=colours, edgecolor='grey', linewidth=0.3)
            ax.set_yticks(y)
            ax.set_yticklabels(
                [f'#{r+1:3d}  {lbl}  d={d:.3f}'
                 for r, (lbl, d) in enumerate(zip(labels, res['distances']))],
                fontsize=7.5,
            )
            ax.invert_yaxis()
            ax.axvline(pred, color='crimson', lw=2.5, ls='--',
                       label=f'Prediction: {pred:.4g} {unit}\n±{sig:.3g} (1σ)')
            ax.set_xlabel(f'{title} [{unit}]')
            ax.set_title(f'{title} | {alt_km} km | top-{k}')
            ax.legend(fontsize=9)
            ax.grid(axis='x', alpha=0.3)

        safe = self.model_name.replace(' ', '_').replace('-', '_')
        plt.suptitle(
            f'{self.model_name} | alt={alt_km} km | method={res["method"]}',
            fontweight='bold',
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f'analog_bars_{safe}_{int(alt_km)}km.png')
        plt.savefig(path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f'Saved bar chart → {path}')

    def plot_analog_timeseries(
        self,
        query_features: dict,
        alt_km: float,
        k: int = 20,
        output_dir: str = '.',
    ) -> None:
        """
        Two-panel time series: total mass density and neutral temperature.
        Full historical record in blue; k analog dates highlighted in red.
        """
        if not self._fitted:
            raise RuntimeError('Call .fit() first.')

        res  = self.predict(query_features, alt_km, k=k)
        alts = np.array(self.altitudes_km)

        def _full(name):
            return np.array([
                np.interp(alt_km, alts, self.species_db[name][i, :])
                for i in range(len(self.dates))
            ]) if name in self.species_db else np.full(len(self.dates), np.nan)

        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        for ax, (name, label, unit) in zip(
            axes,
            [('total_density', 'Total Mass Density', 'kg/m³'),
             ('temperature',   'Neutral Temperature', 'K')],
        ):
            full_rec = _full(name)
            a_vals   = res['analog_vals'].get(name, np.zeros(k))
            pred     = res['outputs_pred'].get(name, np.nan)

            ax.plot(pd.DatetimeIndex(self.dates), full_rec,
                    color='steelblue', lw=0.5, alpha=0.7, label='Historical')
            sc = ax.scatter(
                pd.DatetimeIndex(res['analog_dates']), a_vals,
                c=np.arange(k), cmap='Reds_r', s=70, zorder=5,
                edgecolors='black', linewidths=0.4,
                label=f'Top-{k} analogs',
            )
            plt.colorbar(sc, ax=ax, label='Analog rank (1 = closest)')
            ax.axhline(pred, color='crimson', lw=2, ls='--',
                       label=f'Prediction: {pred:.4g} {unit}')
            ax.set_ylabel(f'{label} [{unit}]')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

        axes[-1].set_xlabel('Date')
        safe = self.model_name.replace(' ', '_').replace('-', '_')
        plt.suptitle(f'{self.model_name} Analog Locations | {alt_km} km',
                     fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'analog_ts_{safe}_{int(alt_km)}km.png')
        plt.savefig(path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f'Saved timeseries → {path}')

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)
        print(f'[{self.model_name} AnalogPredictor] Saved → {path}')

    @staticmethod
    def load(path: str) -> 'AtmosphericAnalogPredictor':
        with open(path, 'rb') as fh:
            obj = pickle.load(fh)
        stored = getattr(obj, '_PKL_VERSION', 1)
        if stored != PKL_SCHEMA_VERSION:
            raise ValueError(
                f'Stale pkl: schema v{stored} on disk, '
                f'but current code requires v{PKL_SCHEMA_VERSION}. '
                f'Delete the .pkl file and re-run the runner script.'
            )
        print(f'[AnalogPredictor] Loaded ← {path}  (schema v{stored})')
        return obj


# =============================================================================
# INTERACTIVE TERMINAL QUERY
# =============================================================================

def terminal_query(
    predictor: AtmosphericAnalogPredictor,
    output_dir: str = '.',
    model_notes: str = '',
) -> None:
    """
    Interactive terminal loop: prompt for query features, show prediction,
    offer SPARTA block, CSV export, and plots.

    Parameters
    ----------
    predictor    : fitted AtmosphericAnalogPredictor
    output_dir   : directory for saved files
    model_notes  : model-specific notes passed to generate_sparta_block()
                   (e.g. 'Ar not modeled by DTM-2020; Ar fraction = 0.')
    """
    bar  = '=' * 72
    bar2 = '─' * 68

    print(f'\n{bar}')
    print(f'  {predictor.model_name} ANALOG FORECASTER — Interactive Query')
    print(bar)
    print(f'  Database   : {len(predictor.dates)} days  '
          f'({str(predictor.dates[0])[:10]} → {str(predictor.dates[-1])[:10]})')
    print(f'  Features   : {predictor.feature_cols}')
    print(f'  Altitudes  : {predictor.altitudes_km} km  '
          f'(other values are linearly interpolated)')
    print(f'  Species    : {list(predictor.species_db.keys())}')
    print("  Type 'q' at any prompt to exit.\n")

    while True:
        print('\n--- New Query ---')
        query = {}

        # Collect each feature; offer date shortcut for DOY sin/cos
        doy_filled = False
        for col in predictor.feature_cols:
            if col == 'doy_sin' and not doy_filled:
                raw = input(
                    '  Target date [YYYY-MM-DD for auto doy_sin/cos,  '
                    'or press Enter to type manually]: '
                ).strip()
                if raw.lower() == 'q':
                    print('Exiting.'); return
                if raw:
                    try:
                        ts  = pd.Timestamp(raw)
                        doy = float(ts.day_of_year)
                        query['doy_sin'] = float(np.sin(2.0 * np.pi * doy / 365.25))
                        query['doy_cos'] = float(np.cos(2.0 * np.pi * doy / 365.25))
                        print(f'  doy_sin = {query["doy_sin"]:.6f},  '
                              f'doy_cos = {query["doy_cos"]:.6f}')
                        doy_filled = True
                        continue
                    except Exception:
                        print('  Could not parse date — enter doy_sin manually.')
            if col == 'doy_cos' and doy_filled:
                continue

            while True:
                raw = input(f'  {col:12s}: ').strip()
                if raw.lower() == 'q':
                    print('Exiting.'); return
                try:
                    query[col] = float(raw); break
                except ValueError:
                    print('  ⚠  Please enter a numeric value.')

        while True:
            raw = input(f'\n  alt_km  [trained: {predictor.altitudes_km} km]: ').strip()
            if raw.lower() == 'q':
                print('Exiting.'); return
            try:
                alt_km = float(raw); break
            except ValueError:
                print('  ⚠  Please enter a number.')

        k_raw = input('  k       [analogs, default=20]: ').strip()
        k     = int(k_raw) if k_raw.isdigit() else 20
        m_raw = input('  method  [inverse_distance / mean / best, default=inverse_distance]: ').strip()
        method = m_raw if m_raw in ('inverse_distance', 'mean', 'best') else 'inverse_distance'

        # ── Run prediction ─────────────────────────────────────────────────
        result = predictor.predict(query, alt_km, k=k, method=method)
        preds  = result['outputs_pred']
        sigmas = result['outputs_sigma']
        fracs  = result['species_fracs']

        print(f'\n{bar2}')
        print(f'  RESULT  |  alt={alt_km} km  |  k={k}  |  method={method}')
        print(bar2)
        rho = preds.get('total_density', float('nan'))
        T   = preds.get('temperature',   float('nan'))
        print(f'  Total mass density  ρ  = {rho:.6e} kg/m³'
              f'  ± {sigmas.get("total_density", float("nan")):.3e}  (1σ)')
        print(f'  Neutral temperature T  = {T:.2f} K'
              f'              ± {sigmas.get("temperature", float("nan")):.2f}  (1σ)')
        print(f'  SPARTA nrho (Σ nᵢ)    = {result["nrho"]:.6e} m⁻³\n')

        print(f'  {"Species":<6}  {"n [m⁻³]":>14}  {"frac":>9}')
        print('  ' + '─' * 35)
        for sp in SPARTA_SPECIES_ORDER:
            ni  = preds.get(sp, 0.0)
            fi  = fracs.get(sp, 0.0)
            tag = '' if sp in result['available_species'] else '  (not in model)'
            print(f'  {sp:<6}  {ni:>14.4e}  {fi:>9.6f}{tag}')
        frac_sum = sum(fracs.values())
        ok       = '✓' if abs(frac_sum - 1.0) < 1e-3 else '⚠  WARNING'
        print(f'  {"SUM":<6}  {"":>14}  {frac_sum:>9.6f}  {ok}')

        v = circular_orbital_velocity(alt_km)
        print(f'\n  Orbital velocity (circular, {alt_km:.0f} km): {v:.2f} m/s')

        print(f'\n  Top-{k} analogs:')
        rho_a = result['analog_vals'].get('total_density', np.zeros(k))
        T_a   = result['analog_vals'].get('temperature',   np.zeros(k))
        print(f'  {"#":>4}  {"Date":>12}  {"Dist":>8}  {"ρ [kg/m³]":>14}  {"T [K]":>8}  {"w":>8}')
        print('  ' + '─' * 64)
        for i, (d, dist, rho_i, Ti, wi) in enumerate(
            zip(result['analog_dates'], result['distances'],
                rho_a, T_a, result['weights']), start=1
        ):
            print(f'  {i:>4d}  {str(pd.Timestamp(d))[:10]:>12}  '
                  f'{dist:>8.4f}  {rho_i:>14.4e}  {Ti:>8.2f}  {wi:>8.4f}')

        # ── Optional outputs ───────────────────────────────────────────────
        if input('\n  Generate SPARTA atmosphere block? [y/n]: ').strip().lower() == 'y':
            generate_sparta_block(
                model_name=predictor.model_name,
                alt_km=alt_km,
                result=result,
                query_features=query,
                output_dir=output_dir,
                save=True,
                model_notes=model_notes,
            )

        if input('  Save ranked analog table to CSV? [y/n]: ').strip().lower() == 'y':
            table = predictor.rank_analogs(query, alt_km, top_n=k)
            safe  = predictor.model_name.replace(' ', '_').replace('-', '_')
            path  = os.path.join(output_dir, f'analogs_{safe}_{int(alt_km)}km.csv')
            table.to_csv(path, index=False)
            print(f'  Saved → {path}')

        if input('  Plot analog bar chart? [y/n]: ').strip().lower() == 'y':
            predictor.plot_analog_bars(query, alt_km, k=k, output_dir=output_dir)

        if input('  Plot analog timeseries? [y/n]: ').strip().lower() == 'y':
            predictor.plot_analog_timeseries(query, alt_km, k=k, output_dir=output_dir)

        if input('\n  Another query? [y/n]: ').strip().lower() != 'y':
            print('Done.'); return
