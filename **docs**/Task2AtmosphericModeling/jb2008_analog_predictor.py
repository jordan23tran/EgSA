"""
jb2008_analog_predictor.py
==========================
kNN analog forecasting surrogate for JB2008 outputs.

Dependencies: numpy, pandas, matplotlib
(No scikit-learn required – normalisation and distance computed manually
 so the underlying maths stays visible.)

Usage
-----
  from jb2008_analog_predictor import (
      build_feature_df, JB2008AnalogPredictor, terminal_query
  )
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1 – DRIVER FILE PARSERS
# =============================================================================

def parse_solfsmy(path: str) -> pd.DataFrame:
    """
    Parse SOLFSMY.TXT → daily solar proxy DataFrame.

    Observed format:
        YEAR DOY JD F10 F10B S10 S10B XM10 XM10B Y10 Y10B FLAG
    """
    names = [
        'YEAR', 'DOY', 'JD',
        'F10', 'F10B', 'S10', 'S10B',
        'XM10', 'XM10B', 'Y10', 'Y10B',
        'FLAG'
    ]

    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=names,
        comment='#',
        engine='python'
    )

    # Convert numeric fields
    numeric_cols = [
        'YEAR', 'DOY', 'JD',
        'F10', 'F10B', 'S10', 'S10B',
        'XM10', 'XM10B', 'Y10', 'Y10B'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only valid rows
    df = df.dropna(subset=['YEAR', 'DOY']).copy()
    df['YEAR'] = df['YEAR'].astype(int)
    df['DOY']  = df['DOY'].astype(int)

    df = df[(df['YEAR'] >= 1900) & (df['YEAR'] <= 2100)]
    df = df[(df['DOY'] >= 1) & (df['DOY'] <= 366)].copy()

    if df.empty:
        raise ValueError(f"No valid SOLFSMY data rows found in {path}")

    # Build date from year + day-of-year
    df['Date'] = (
        pd.to_datetime(df['YEAR'].astype(str), format='%Y', utc=True)
        + pd.to_timedelta(df['DOY'] - 1, unit='D')
    ).dt.normalize()

    df = df.set_index('Date')

    return df[['F10', 'F10B', 'S10', 'S10B', 'XM10', 'XM10B', 'Y10', 'Y10B']]
def parse_dtcfile(path: str) -> pd.DataFrame:
    """
    Parse DTCFILE.TXT → daily geomagnetic temperature correction (dTc).

    Observed format:
        DTC YEAR DOY v1 v2 v3 ... vN

    where the dTc values are stored in fixed-width fields, and some adjacent
    values may appear merged if parsed only with whitespace splitting.
    """

    rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                continue
            if not line.startswith("DTC"):
                continue

            # First parse the prefix safely using whitespace split
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue

            tag, year_str, doy_str, rest = parts
            if tag != "DTC":
                continue

            try:
                year = int(year_str)
                doy  = int(doy_str)
            except ValueError:
                continue

            # Remaining values are fixed-width 4-character fields
            vals = []
            for i in range(0, len(rest), 4):
                chunk = rest[i:i+4].strip()
                if not chunk:
                    continue
                try:
                    vals.append(float(chunk))
                except ValueError:
                    pass

            # Skip rows with no valid dTc values
            if not vals:
                continue

            rows.append({
                "YEAR": year,
                "DOY": doy,
                "dTc_mean": float(np.mean(vals)),
                "dTc_max": float(np.max(vals)),
            })

    if not rows:
        raise ValueError(
            f"No valid DTC rows found in {path}. "
            "Check the file format."
        )

    df = pd.DataFrame(rows)

    # Keep only plausible rows
    df = df[(df["YEAR"] >= 1900) & (df["YEAR"] <= 2100)]
    df = df[(df["DOY"] >= 1) & (df["DOY"] <= 366)].copy()

    if df.empty:
        raise ValueError(
            f"No valid DTC data rows remained after filtering in {path}."
        )

    df["Date"] = (
        pd.to_datetime(df["YEAR"].astype(str), format="%Y", utc=True)
        + pd.to_timedelta(df["DOY"] - 1, unit="D")
    ).dt.normalize()

    df = df.set_index("Date")
    return df[["dTc_mean", "dTc_max"]]
def build_feature_df(
    dates,
    solfsmy_path: str,
    dtcfile_path: str = None,
    include_dtc:  bool = True,
    include_doy:  bool = True,
) -> pd.DataFrame:
    """
    Assemble the feature DataFrame aligned to `dates`.

    Parameters
    ----------
    dates         : array-like of tz-aware Timestamps  (e.g. from pd.date_range)
    solfsmy_path  : path to SOLFSMY.TXT
    dtcfile_path  : path to DTCFILE.TXT  (optional but recommended)
    include_dtc   : append dTc_mean / dTc_max columns if file is provided
    include_doy   : append sin/cos day-of-year to capture seasonal density variation

    Returns
    -------
    pd.DataFrame  shape (len(dates), n_features), indexed by dates
    """
    sol = parse_solfsmy(solfsmy_path)

    # Align by normalised (midnight) date using nearest-day matching
    idx_norm = pd.DatetimeIndex(dates).normalize()
    feat = sol.reindex(idx_norm, method='nearest', tolerance=pd.Timedelta('1D'))
    feat.index = pd.DatetimeIndex(dates)

    if include_dtc and dtcfile_path:
        dtc = parse_dtcfile(dtcfile_path)
        dtc_aligned = dtc.reindex(
            idx_norm, method='nearest', tolerance=pd.Timedelta('1D')
        )
        dtc_aligned.index = pd.DatetimeIndex(dates)
        feat = pd.concat([feat, dtc_aligned], axis=1)

    if include_doy:
        # Encode day-of-year as (sin, cos) so 31 Dec ≈ 1 Jan in feature space
        doy = pd.DatetimeIndex(dates).day_of_year.astype(float)
        feat['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
        feat['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    n_nan = feat.isna().any(axis=1).sum()
    if n_nan:
        print(f"[build_feature_df] WARNING: {n_nan} rows contain NaN "
              "(dates outside driver-file coverage). "
              "These will be dropped during predictor fitting.")
    return feat


# =============================================================================
# SECTION 2 – ANALOG PREDICTOR CLASS
# =============================================================================

class JB2008AnalogPredictor:
    """
    k-Nearest Neighbours analog predictor for JB2008 density and temperature.

    Feature vector (default, all 12 scalars)
    -----------------------------------------
        Solar proxies : F10  F10B  S10  S10B  XM10  XM10B  Y10  Y10B
        Geomagnetic   : dTc_mean  dTc_max        (requires DTCFILE.TXT)
        Seasonal      : doy_sin  doy_cos

    Normalisation
    -------------
    Each feature is independently z-score normalised across the database
    (zero mean, unit variance) so that no single proxy dominates the
    Euclidean distance purely due to its magnitude.

    Altitude handling
    -----------------
    The database stores outputs at the discrete altitudes used in your
    JB2008 run.  Predictions at intermediate altitudes are obtained by
    linear interpolation between the two bracketing stored levels.
    """

    def __init__(self):
        # Set by fit()
        self.feature_cols = None   # list[str]
        self.altitudes_km = None   # list[float]
        self.dates        = None   # np.ndarray of Timestamps, shape (n,)
        self.features_raw = None   # (n, p)  float64  un-normalised
        self.features_sc  = None   # (n, p)  float64  normalised
        self.rho_db       = None   # (n, a)  kg m⁻³
        self.T_db         = None   # (n, a)  K
        self._mean        = None   # (p,)    for manual z-score
        self._std         = None   # (p,)
        self._fitted      = False

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        dates,
        feature_df: pd.DataFrame,
        rho_array:  np.ndarray,
        T_array:    np.ndarray,
        altitudes_km: list
    ):
        """
        Populate the analog database and fit the feature normaliser.

        Parameters
        ----------
        dates        : array-like of datetime-like, length n_days
        feature_df   : DataFrame (n_days × n_features) aligned to dates
        rho_array    : ndarray (n_days × n_alts)  atmospheric density [kg/m³]
        T_array      : ndarray (n_days × n_alts)  neutral temperature  [K]
        altitudes_km : list of n_alts altitude values [km]
        """
        valid = ~feature_df.isna().any(axis=1).values
        n_bad = (~valid).sum()
        if n_bad:
            print(f"[fit] Dropping {n_bad} rows with NaN features.")

        self.feature_cols = list(feature_df.columns)
        self.altitudes_km = [float(a) for a in altitudes_km]
        self.dates        = np.array(pd.DatetimeIndex(dates))[valid]
        self.features_raw = feature_df.values[valid].astype(float)
        self.rho_db       = rho_array[valid].astype(float)
        self.T_db         = T_array[valid].astype(float)

        # z-score normalisation fitted on the database
        self._mean       = self.features_raw.mean(axis=0)
        self._std        = self.features_raw.std(axis=0)
        self._std[self._std < 1e-12] = 1.0     # guard against constant cols
        self.features_sc = (self.features_raw - self._mean) / self._std
        self._fitted     = True

        print(
            f"[JB2008AnalogPredictor.fit]  "
            f"{len(self.dates)} days  |  "
            f"{len(self.feature_cols)} features  |  "
            f"alts = {self.altitudes_km} km"
        )
        print(f"  Features: {self.feature_cols}")

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _normalise_query(self, query_dict: dict) -> np.ndarray:
        """
        Convert a {feature: value} dict → z-scored 1-D array using the
        normalisation parameters fitted on the database.
        """
        missing = [c for c in self.feature_cols if c not in query_dict]
        if missing:
            raise ValueError(
                f"Query is missing required features: {missing}\n"
                f"Expected: {self.feature_cols}"
            )
        raw = np.array([query_dict[c] for c in self.feature_cols], dtype=float)
        return (raw - self._mean) / self._std    # shape (p,)

    def _euclidean_distances(self, q_sc: np.ndarray) -> np.ndarray:
        """
        Euclidean distance from a single scaled query vector to every
        row in the scaled database.  Shape: (n_days,)
        """
        diff = self.features_sc - q_sc          # broadcasting (n, p) - (p,)
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    def _get_analogs_at_alt(
        self, indices: np.ndarray, alt_km: float
    ):
        """
        Extract rho and T for the selected database rows at alt_km.
        Interpolates linearly if alt_km falls between stored levels.
        """
        alts = np.array(self.altitudes_km)
        if alt_km in alts:
            ai = int(np.argwhere(alts == alt_km)[0])
            return self.rho_db[indices, ai], self.T_db[indices, ai]

        if alt_km < alts.min() or alt_km > alts.max():
            print(f"[predict] WARNING: {alt_km} km is outside the trained "
                  f"altitude range [{alts.min()}, {alts.max()}] km.  "
                  "Extrapolating – results may be less accurate.")
        rho = np.array([np.interp(alt_km, alts, self.rho_db[i, :])
                        for i in indices])
        T   = np.array([np.interp(alt_km, alts, self.T_db[i, :])
                        for i in indices])
        return rho, T

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(
        self,
        query_features: dict,
        alt_km: float,
        k: int = 20,
        method: str = 'inverse_distance'
    ) -> dict:
        """
        Predict atmospheric density and neutral temperature.

        Parameters
        ----------
        query_features : dict  {feature_name: float}
            Values for every feature the predictor was fitted on.
            Example (with dTc and doy features active):
              {
                'F10': 152.0, 'F10B': 145.0,
                'S10': 149.0, 'S10B': 143.0,
                'XM10': 148.0, 'XM10B': 144.0,
                'Y10': 138.0, 'Y10B': 135.0,
                'dTc_mean': 5.0, 'dTc_max': 18.0,
                'doy_sin': 0.5,  'doy_cos': 0.866
              }
        alt_km  : float  target altitude in km
        k       : int    number of nearest analogs to use
        method  : 'inverse_distance'  1/d weighted mean  ← recommended
                  'mean'              simple mean of top-k
                  'best'              single nearest analog only

        Returns
        -------
        dict  – all values needed for display, saving, and plotting:
          rho_pred      float   kg/m³   point estimate
          T_pred        float   K       point estimate
          rho_sigma     float   kg/m³   spread of k analogs (uncertainty proxy)
          T_sigma       float   K
          analog_dates  ndarray top-k matched dates
          distances     ndarray corresponding normalised Euclidean distances
          rho_analogs   ndarray density value at each analog day
          T_analogs     ndarray temperature at each analog day
          weights       ndarray combination weights (sum to 1)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        k = min(k, len(self.dates))
        q_sc  = self._normalise_query(query_features)
        dists = self._euclidean_distances(q_sc)

        top_idx   = np.argsort(dists)[:k]
        top_dists = dists[top_idx]

        rho_analogs, T_analogs = self._get_analogs_at_alt(top_idx, alt_km)

        # Combination weights
        eps = 1e-12
        if method == 'best':
            weights = np.zeros(k);  weights[0] = 1.0
        elif method == 'mean':
            weights = np.ones(k) / k
        elif method == 'inverse_distance':
            weights = 1.0 / (top_dists + eps)
            weights /= weights.sum()
        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Choose 'inverse_distance', 'mean', or 'best'.")

        return {
            'rho_pred'    : float(np.dot(weights, rho_analogs)),
            'T_pred'      : float(np.dot(weights, T_analogs)),
            'rho_sigma'   : float(np.std(rho_analogs)),
            'T_sigma'     : float(np.std(T_analogs)),
            'analog_dates': self.dates[top_idx],
            'distances'   : top_dists,
            'rho_analogs' : rho_analogs,
            'T_analogs'   : T_analogs,
            'weights'     : weights,
            'k'           : k,
            'method'      : method,
            'alt_km'      : alt_km,
        }

    # ------------------------------------------------------------------
    # RANK ALL ANALOGS
    # ------------------------------------------------------------------

    def rank_analogs(
        self,
        query_features: dict,
        top_n: int = None
    ) -> pd.DataFrame:
        """
        Return all (or top_n) historical days sorted by distance to query.

        Columns: Rank, Date, Distance,
                 <all feature columns>,
                 rho_<alt>km [kg/m³], T_<alt>km [K]
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        n     = len(self.dates) if top_n is None else min(top_n, len(self.dates))
        q_sc  = self._normalise_query(query_features)
        dists = self._euclidean_distances(q_sc)
        idx   = np.argsort(dists)[:n]

        rows = []
        for rank, i in enumerate(idx, start=1):
            row = {
                'Rank'    : rank,
                'Date'    : pd.Timestamp(self.dates[i]),
                'Distance': float(dists[i]),
            }
            for ci, col in enumerate(self.feature_cols):
                row[col] = float(self.features_raw[i, ci])
            for j, alt in enumerate(self.altitudes_km):
                row[f'rho_{alt}km [kg/m³]'] = float(self.rho_db[i, j])
                row[f'T_{alt}km [K]']        = float(self.T_db[i, j])
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    def plot_analog_bars(
        self,
        query_features: dict,
        alt_km: float,
        k: int = 20,
        output_dir: str = '.'
    ):
        """
        Horizontal bar chart of the top-k analog values for ρ and T,
        ordered by distance (best match at top), with prediction in red.
        """
        res = self.predict(query_features, alt_km, k=k)
        labels  = [str(pd.Timestamp(d))[:10] for d in res['analog_dates']]
        colours = plt.cm.viridis_r(np.linspace(0.1, 0.9, k))

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, k * 0.4)))
        for ax, vals, pred, sigma, title, unit in zip(
            axes,
            [res['rho_analogs'], res['T_analogs']],
            [res['rho_pred'],    res['T_pred']],
            [res['rho_sigma'],   res['T_sigma']],
            ['Atmospheric Density', 'Neutral Temperature'],
            ['kg/m³', 'K']
        ):
            y = np.arange(k)
            ax.barh(y, vals, color=colours, edgecolor='grey', linewidth=0.3)
            ax.set_yticks(y)
            ax.set_yticklabels(
                [f"#{r+1:3d}  {lbl}  d={d:.3f}"
                 for r, (lbl, d) in enumerate(zip(labels, res['distances']))],
                fontsize=8
            )
            ax.invert_yaxis()
            ax.axvline(pred, color='crimson', lw=2.5, ls='--',
                       label=f'Prediction: {pred:.4g} {unit}\n'
                             f'±{sigma:.3g} {unit}  (1σ across analogs)')
            ax.set_xlabel(f'{title} [{unit}]', fontsize=11)
            ax.set_title(f'{title} at {alt_km} km | Top-{k} Analogs', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(axis='x', alpha=0.35)

        plt.suptitle(
            f"JB2008 Analog Forecast  |  alt={alt_km} km  |  "
            f"method={res['method']}",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"analog_bars_{int(alt_km)}km.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.show()
        print(f"Saved: {path}")

    def plot_analog_timeseries(
        self,
        query_features: dict,
        alt_km: float,
        k: int = 20,
        output_dir: str = '.'
    ):
        """
        Plot the full historical density record and highlight where the
        top-k analog dates fall (coloured by rank / distance).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        res = self.predict(query_features, alt_km, k=k)

        # Full-database density at requested altitude
        alts = np.array(self.altitudes_km)
        if alt_km in alts:
            ai = int(np.argwhere(alts == alt_km)[0])
            rho_all = self.rho_db[:, ai]
        else:
            rho_all = np.array([
                np.interp(alt_km, alts, self.rho_db[i, :])
                for i in range(len(self.dates))
            ])

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(pd.DatetimeIndex(self.dates), rho_all,
                color='steelblue', lw=0.6, alpha=0.75, label='Historical record')
        sc = ax.scatter(
            pd.DatetimeIndex(res['analog_dates']),
            res['rho_analogs'],
            c=np.arange(k),          # rank-coloured: dark = closest
            cmap='Reds_r',
            s=80, zorder=5,
            edgecolors='black', linewidths=0.5,
            label=f'Top-{k} analogs'
        )
        plt.colorbar(sc, ax=ax, label='Analog rank (1 = closest)')
        ax.axhline(res['rho_pred'], color='crimson', lw=2, ls='--',
                   label=f"Prediction: {res['rho_pred']:.4g} kg/m³")
        ax.set_xlabel('Date')
        ax.set_ylabel('Atmospheric Density [kg/m³]')
        ax.set_title(f'Analog Locations in Historical Record | {alt_km} km')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"analog_timeseries_{int(alt_km)}km.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.show()
        print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # SAVE / LOAD  (avoids re-running JB2008 every session)
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Serialise the fitted predictor to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[JB2008AnalogPredictor] Saved → {path}")

    @staticmethod
    def load(path: str) -> 'JB2008AnalogPredictor':
        """Deserialise a previously saved predictor."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"[JB2008AnalogPredictor] Loaded ← {path}")
        return obj


# =============================================================================
# SECTION 3 – INTERACTIVE TERMINAL QUERY
# =============================================================================

# Descriptions shown to the user at the prompt
_FEATURE_DESCRIPTIONS = {
    'F10'     : "F10.7 solar radio flux – current day (sfu)",
    'F10B'    : "F10.7B – 81-day centred mean of F10 (sfu)",
    'S10'     : "S10 – solar EUV proxy (sfu)",
    'S10B'    : "S10B – 81-day mean of S10 (sfu)",
    'XM10'    : "M10.7 – Mg II UV proxy (sfu)",
    'XM10B'   : "XM10B – 81-day mean of XM10 (sfu)",
    'Y10'     : "Y10 – solar FUV proxy (sfu)",
    'Y10B'    : "Y10B – 81-day mean of Y10 (sfu)",
    'dTc_mean': "dTc_mean – daily-mean geomagnetic temperature correction (K)",
    'dTc_max' : "dTc_max  – daily-max  geomagnetic temperature correction (K)",
    'doy_sin' : "sin(2π × DOY / 365.25)  e.g. 1 Jan → 0.0171",
    'doy_cos' : "cos(2π × DOY / 365.25)  e.g. 1 Jan → 0.9999",
}


def _doy_sincos(date_str: str):
    """Helper: compute (sin, cos) of day-of-year for a date string 'YYYY-MM-DD'."""
    ts  = pd.Timestamp(date_str)
    doy = float(ts.day_of_year)
    return np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)


def terminal_query(predictor: JB2008AnalogPredictor, output_dir: str = '.'):
    """
    Interactive command-line interface for querying the analog predictor.

    The user is prompted to enter each driver value.
    For doy_sin / doy_cos the user can either type a date (YYYY-MM-DD)
    or enter the values directly.
    Results are printed and optionally saved / plotted.
    """
    print("\n" + "=" * 72)
    print("  JB2008 ANALOG FORECASTER  –  Interactive Terminal Query")
    print("=" * 72)
    print(f"  Database : {len(predictor.dates)} days "
          f"({str(predictor.dates[0])[:10]} → {str(predictor.dates[-1])[:10]})")
    print(f"  Features : {predictor.feature_cols}")
    print(f"  Altitudes: {predictor.altitudes_km} km  "
          "(other values will be interpolated)")
    print("  Type 'q' at any prompt to exit.\n")

    while True:
        print("\n--- New Query ---")
        query = {}

        for col in predictor.feature_cols:
            desc = _FEATURE_DESCRIPTIONS.get(col, col)

            # Special case: accept a date string for doy_sin / doy_cos
            if col == 'doy_sin':
                raw = input(
                    f"  {'Target date':12s} [YYYY-MM-DD  OR  enter separately]"
                    f"\n  → date (auto-computes doy_sin and doy_cos): "
                ).strip()
                if raw.lower() == 'q':
                    print("Exiting."); return
                try:
                    s, c = _doy_sincos(raw)
                    query['doy_sin'] = s
                    query['doy_cos'] = c
                    print(f"  doy_sin = {s:.6f},  doy_cos = {c:.6f}")
                    continue
                except Exception:
                    pass   # fall through to numeric entry

            if col == 'doy_cos' and 'doy_cos' in query:
                continue   # already set by date entry above

            while True:
                raw = input(f"  {col:12s}  [{desc}]: ").strip()
                if raw.lower() == 'q':
                    print("Exiting."); return
                try:
                    query[col] = float(raw)
                    break
                except ValueError:
                    print("  ⚠  Please enter a numeric value.")

        # Altitude
        while True:
            raw = input(
                f"\n  alt_km   [available: {predictor.altitudes_km} km, "
                "or any value for interpolation]: "
            ).strip()
            if raw.lower() == 'q':
                print("Exiting."); return
            try:
                alt_km = float(raw); break
            except ValueError:
                print("  ⚠  Please enter a number.")

        # k and method
        k_raw = input("  k        [analogs to use, default=20]: ").strip()
        k     = int(k_raw) if k_raw.isdigit() else 20

        m_raw = input(
            "  method   [inverse_distance / mean / best, default=inverse_distance]: "
        ).strip()
        method = m_raw if m_raw in ('inverse_distance', 'mean', 'best') \
                 else 'inverse_distance'

        # ── Run prediction ────────────────────────────────────────────
        result = predictor.predict(query, alt_km, k=k, method=method)

        print("\n" + "-" * 60)
        print(f"  RESULT  |  alt = {alt_km} km  |  k = {k}  |  method = {method}")
        print("-" * 60)
        print(f"  Atmospheric Density  ρ = {result['rho_pred']:.6e} kg/m³"
              f"   ±{result['rho_sigma']:.3e} (1σ)")
        print(f"  Neutral Temperature  T = {result['T_pred']:.2f} K"
              f"           ±{result['T_sigma']:.2f} K (1σ)")
        print(f"\n  Top-{k} Analog Dates:")
        print(f"  {'#':>4}  {'Date':>12}  {'Dist':>8}  "
              f"{'ρ [kg/m³]':>14}  {'T [K]':>8}  {'Weight':>8}")
        print("  " + "-" * 60)
        for i, (d, dist, rho, T, w) in enumerate(zip(
            result['analog_dates'], result['distances'],
            result['rho_analogs'], result['T_analogs'], result['weights']
        ), start=1):
            print(f"  {i:>4d}  {str(pd.Timestamp(d))[:10]:>12}  "
                  f"{dist:>8.4f}  {rho:>14.4e}  {T:>8.2f}  {w:>8.4f}")

        # Optional outputs
        if input("\n  Save ranked table to CSV? [y/n]: ").strip().lower() == 'y':
            table    = predictor.rank_analogs(query, top_n=k)
            csv_path = os.path.join(output_dir,
                                    f"analog_ranked_{int(alt_km)}km.csv")
            table.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

        if input("  Plot analog bar chart? [y/n]: ").strip().lower() == 'y':
            predictor.plot_analog_bars(query, alt_km, k=k, output_dir=output_dir)

        if input("  Plot analog positions on historical record? [y/n]: ").strip().lower() == 'y':
            predictor.plot_analog_timeseries(query, alt_km, k=k,
                                             output_dir=output_dir)

        if input("\n  Another query? [y/n]: ").strip().lower() != 'y':
            print("Done."); break