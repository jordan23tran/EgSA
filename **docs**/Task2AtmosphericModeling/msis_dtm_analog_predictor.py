"""
msis_dtm_analog_predictor.py
============================
kNN analog forecasting surrogate for NRLMSIS-00 and DTM-2020 atmospheric
model outputs.

Predicts
--------
  • Atmospheric mass density         [kg/m³]
  • Atomic oxygen number density     [m⁻³]

at any stored or interpolated altitude.

Feature vectors
---------------
NRLMSIS-00                          DTM-2020 (SWAMI MCM)
──────────────────────────────      ──────────────────────────────
F107    F10.7 daily         [sfu]   F107    F10.7 daily         [sfu]
F107A   81-day mean F10.7   [sfu]   F107m   81-day mean F10.7   [sfu]
Ap_daily  daily Ap index  [2 nT]   kp1     instantaneous Kp
doy_sin  sin(2π DOY/365.25)         kp2     daily mean Kp
doy_cos  cos(2π DOY/365.25)         doy_sin / doy_cos

Normalisation
-------------
All features are z-score normalised (zero mean, unit variance) across the
historical database so no single proxy dominates Euclidean distance purely
by magnitude.

Altitude handling
-----------------
Outputs are stored at the discrete altitudes used in the model run.
Predictions at intermediate values use linear interpolation.

Usage
-----
  from msis_dtm_analog_predictor import (
      build_feature_df_msis,
      build_feature_df_dtm,
      AtmosphericAnalogPredictor,
      terminal_query,
  )
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# UTILITY
# =============================================================================

def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return a tz-naive copy of a DatetimeIndex.
    Needed because MSIS/DTM CSVs produce tz-naive dates while
    JB2008 uses tz-aware UTC dates; this keeps reindex calls consistent.
    """
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


# =============================================================================
# SECTION 1 – FEATURE BUILDERS
# =============================================================================

def build_feature_df_msis(
    dates,
    sw_csv_path: str,
    f107_col:    str  = "F107_adj",
    ap_col:      str  = "Ap_daily",
    include_doy: bool = True,
) -> pd.DataFrame:
    """
    Build a feature DataFrame aligned to `dates` for NRLMSIS-00.

    Parameters
    ----------
    dates        : array-like of Timestamps (tz-aware or tz-naive)
    sw_csv_path  : path to the daily space-weather CSV.
                   Required columns: Year, Month, Day, <f107_col>, <ap_col>
    f107_col     : name of the F10.7 column in the CSV   (default 'F107_adj')
    ap_col       : name of the daily Ap column           (default 'Ap_daily')
    include_doy  : if True, appends sin/cos(DOY) for seasonal variation

    Returns
    -------
    pd.DataFrame  shape (len(dates), n_features), indexed by dates
    Feature columns: F107, F107A, Ap_daily, [doy_sin, doy_cos]
    """
    sw = pd.read_csv(sw_csv_path)
    sw.columns = sw.columns.str.strip()
    sw["_date"] = pd.to_datetime(sw[["Year", "Month", "Day"]])
    sw = sw.sort_values("_date").reset_index(drop=True)

    # 81-day centred rolling mean of F10.7
    sw["F107A"] = (
        sw[f107_col].rolling(window=81, center=True, min_periods=1).mean()
    )
    sw = sw.set_index("_date")   # tz-naive index

    dates_idx = pd.DatetimeIndex(dates)
    idx_norm  = _strip_tz(dates_idx).normalize()

    def _reindex(series):
        return series.reindex(
            idx_norm, method="nearest", tolerance=pd.Timedelta("1D")
        ).values

    feat = pd.DataFrame(index=dates_idx)
    feat["F107"]     = _reindex(sw[f107_col])
    feat["F107A"]    = _reindex(sw["F107A"])
    feat["Ap_daily"] = _reindex(sw[ap_col])

    if include_doy:
        doy = dates_idx.day_of_year.astype(float)
        feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    n_nan = feat.isna().any(axis=1).sum()
    if n_nan:
        print(f"[build_feature_df_msis] WARNING: {n_nan} rows have NaN "
              "(dates outside CSV coverage). "
              "They will be dropped during predictor fitting.")
    return feat


def build_feature_df_dtm(
    dates,
    sw_csv_path: str,
    f107_col:    str   = "F107_adj",
    kp1_col:     str   = "Kp_09_12",
    kp_cols:     list  = None,
    include_doy: bool  = True,
) -> pd.DataFrame:
    """
    Build a feature DataFrame aligned to `dates` for DTM-2020 (SWAMI MCM).

    Parameters
    ----------
    dates        : array-like of Timestamps (tz-aware or tz-naive)
    sw_csv_path  : path to the daily space-weather CSV.
                   Required columns: Year, Month, Day, <f107_col>,
                                     <kp1_col>, and the 8 three-hourly Kp cols
    f107_col     : F10.7 column name              (default 'F107_adj')
    kp1_col      : instantaneous Kp column name   (default 'Kp_09_12')
    kp_cols      : list of 8 three-hourly Kp columns for computing kp2.
                   Defaults to the standard OMNI Kp_HH_HH column set.
    include_doy  : add sin/cos(DOY) for seasonal variation

    Returns
    -------
    pd.DataFrame  shape (len(dates), n_features)
    Feature columns: F107, F107m, kp1, kp2, [doy_sin, doy_cos]
    """
    if kp_cols is None:
        kp_cols = [
            "Kp_00_03", "Kp_03_06", "Kp_06_09", "Kp_09_12",
            "Kp_12_15", "Kp_15_18", "Kp_18_21", "Kp_21_24",
        ]

    sw = pd.read_csv(sw_csv_path)
    sw.columns = sw.columns.str.strip()
    sw["_date"] = pd.to_datetime(sw[["Year", "Month", "Day"]])
    sw = sw.sort_values("_date").reset_index(drop=True)

    # 81-day centred mean of F10.7
    sw["F107m"] = (
        sw[f107_col].rolling(window=81, center=True, min_periods=1).mean()
    )

    # kp2 = daily mean Kp across all 8 three-hourly slots
    for col in kp_cols:
        sw[col] = pd.to_numeric(sw[col], errors="coerce")
    sw["kp2"] = sw[kp_cols].mean(axis=1)

    sw = sw.set_index("_date")

    dates_idx = pd.DatetimeIndex(dates)
    idx_norm  = _strip_tz(dates_idx).normalize()

    def _reindex(series):
        return series.reindex(
            idx_norm, method="nearest", tolerance=pd.Timedelta("1D")
        ).values

    feat = pd.DataFrame(index=dates_idx)
    feat["F107"]  = _reindex(sw[f107_col])
    feat["F107m"] = _reindex(sw["F107m"])
    feat["kp1"]   = _reindex(sw[kp1_col])
    feat["kp2"]   = _reindex(sw["kp2"])

    if include_doy:
        doy = dates_idx.day_of_year.astype(float)
        feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    n_nan = feat.isna().any(axis=1).sum()
    if n_nan:
        print(f"[build_feature_df_dtm] WARNING: {n_nan} rows have NaN "
              "(dates outside CSV coverage). "
              "They will be dropped during predictor fitting.")
    return feat


# =============================================================================
# SECTION 2 – ANALOG PREDICTOR CLASS
# =============================================================================

class AtmosphericAnalogPredictor:
    """
    k-Nearest Neighbours analog predictor for atmospheric mass density and
    atomic oxygen number density.

    Compatible with NRLMSIS-00 and DTM-2020 (MCM) output databases.

    Normalisation
    -------------
    All features are z-score normalised (zero mean, unit variance) using the
    statistics of the historical database so no single proxy dominates
    the Euclidean distance purely by magnitude.

    Altitude handling
    -----------------
    Outputs are stored at the discrete altitudes used in the model run.
    Predictions at intermediate altitudes use linear interpolation.
    """

    def __init__(self, model_name: str = "NRLMSIS-00"):
        self.model_name   = model_name
        self.feature_cols = None
        self.altitudes_km = None
        self.dates        = None
        self.features_raw = None   # (n, p)  un-normalised
        self.features_sc  = None   # (n, p)  z-score normalised
        self.density_db   = None   # (n, a)  kg/m³
        self.atomic_O_db  = None   # (n, a)  m⁻³
        self._mean        = None   # (p,)    normalisation mean
        self._std         = None   # (p,)    normalisation std
        self._fitted      = False

    # ── FIT ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        dates,
        feature_df:     pd.DataFrame,
        density_array:  np.ndarray,
        atomic_O_array: np.ndarray,
        altitudes_km:   list,
    ):
        """
        Populate the analog database and compute normalisation parameters.

        Parameters
        ----------
        dates          : array-like, length n_days
        feature_df     : DataFrame  (n_days × n_features)
        density_array  : ndarray    (n_days × n_alts)  mass density [kg/m³]
        atomic_O_array : ndarray    (n_days × n_alts)  O number density [m⁻³]
        altitudes_km   : list of n_alts altitude values [km]
        """
        valid = ~feature_df.isna().any(axis=1).values
        n_bad = (~valid).sum()
        if n_bad:
            print(f"[{self.model_name}.fit] Dropping {n_bad} rows with NaN features.")

        self.feature_cols = list(feature_df.columns)
        self.altitudes_km = [float(a) for a in altitudes_km]
        self.dates        = np.array(pd.DatetimeIndex(dates))[valid]
        self.features_raw = feature_df.values[valid].astype(float)
        self.density_db   = density_array[valid].astype(float)
        self.atomic_O_db  = atomic_O_array[valid].astype(float)

        # z-score normalisation fitted on the database
        self._mean = self.features_raw.mean(axis=0)
        self._std  = self.features_raw.std(axis=0)
        self._std[self._std < 1e-12] = 1.0        # guard against constant cols
        self.features_sc = (self.features_raw - self._mean) / self._std
        self._fitted = True

        print(
            f"[{self.model_name} AnalogPredictor.fit]  "
            f"{len(self.dates)} days  |  "
            f"{len(self.feature_cols)} features  |  "
            f"alts = {self.altitudes_km} km"
        )
        print(f"  Feature columns : {self.feature_cols}")

    # ── INTERNAL HELPERS ──────────────────────────────────────────────────────

    def _normalise_query(self, query_dict: dict) -> np.ndarray:
        """Convert a {feature: value} dict → z-scored 1-D array."""
        missing = [c for c in self.feature_cols if c not in query_dict]
        if missing:
            raise ValueError(
                f"Query is missing required features: {missing}\n"
                f"Expected: {self.feature_cols}"
            )
        raw = np.array([query_dict[c] for c in self.feature_cols], dtype=float)
        return (raw - self._mean) / self._std

    def _euclidean_distances(self, q_sc: np.ndarray) -> np.ndarray:
        """
        Euclidean distance from a single scaled query vector to every
        row in the scaled database.  Shape: (n_days,)
        """
        diff = self.features_sc - q_sc          # broadcasting (n, p) - (p,)
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    def _outputs_at_alt(self, indices: np.ndarray, alt_km: float):
        """
        Extract density and atomic O for selected database rows at alt_km.
        Linearly interpolates if alt_km falls between stored altitudes.
        """
        alts = np.array(self.altitudes_km)
        if alt_km in alts:
            ai = int(np.argwhere(alts == alt_km)[0])
            return self.density_db[indices, ai], self.atomic_O_db[indices, ai]

        if alt_km < alts.min() or alt_km > alts.max():
            print(f"[predict] WARNING: {alt_km} km is outside the trained "
                  f"altitude range [{alts.min()}, {alts.max()}] km.  "
                  "Extrapolating – results may be less accurate.")
        density  = np.array([np.interp(alt_km, alts, self.density_db[i, :])
                              for i in indices])
        atomic_O = np.array([np.interp(alt_km, alts, self.atomic_O_db[i, :])
                              for i in indices])
        return density, atomic_O

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def predict(
        self,
        query_features: dict,
        alt_km:  float,
        k:       int  = 20,
        method:  str  = "inverse_distance",
    ) -> dict:
        """
        Predict atmospheric mass density and atomic oxygen number density.

        Parameters
        ----------
        query_features : dict  {feature_name: float}
            Must contain every feature column the predictor was fitted on.

            NRLMSIS-00 example:
              {'F107': 152.0, 'F107A': 145.0, 'Ap_daily': 12.0,
               'doy_sin': 0.50, 'doy_cos': 0.866}

            DTM-2020 example:
              {'F107': 152.0, 'F107m': 145.0,
               'kp1': 1.3, 'kp2': 1.7,
               'doy_sin': 0.50, 'doy_cos': 0.866}

        alt_km  : float   target altitude [km]
        k       : int     number of nearest analogs to combine
        method  : 'inverse_distance'   1/d weighted mean  ← recommended
                  'mean'               simple mean of top-k
                  'best'               single nearest analog only

        Returns
        -------
        dict – all values needed for display, saving, and plotting:
          density_pred     float   kg/m³   point estimate
          atomic_O_pred    float   m⁻³     point estimate
          density_sigma    float   kg/m³   1σ spread across k analogs
          atomic_O_sigma   float   m⁻³
          analog_dates     ndarray top-k matched dates
          distances        ndarray normalised Euclidean distances
          density_analogs  ndarray density at each analog day
          atomic_O_analogs ndarray atomic O at each analog day
          weights          ndarray combination weights (sum to 1)
          k, method, alt_km
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        k = min(k, len(self.dates))
        q_sc  = self._normalise_query(query_features)
        dists = self._euclidean_distances(q_sc)

        top_idx   = np.argsort(dists)[:k]
        top_dists = dists[top_idx]

        density_a, atomic_O_a = self._outputs_at_alt(top_idx, alt_km)

        # Combination weights
        eps = 1e-12
        if method == "best":
            weights = np.zeros(k); weights[0] = 1.0
        elif method == "mean":
            weights = np.ones(k) / k
        elif method == "inverse_distance":
            weights = 1.0 / (top_dists + eps)
            weights /= weights.sum()
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose 'inverse_distance', 'mean', or 'best'."
            )

        return {
            "density_pred"    : float(np.dot(weights, density_a)),
            "atomic_O_pred"   : float(np.dot(weights, atomic_O_a)),
            "density_sigma"   : float(np.std(density_a)),
            "atomic_O_sigma"  : float(np.std(atomic_O_a)),
            "analog_dates"    : self.dates[top_idx],
            "distances"       : top_dists,
            "density_analogs" : density_a,
            "atomic_O_analogs": atomic_O_a,
            "weights"         : weights,
            "k"               : k,
            "method"          : method,
            "alt_km"          : alt_km,
        }

    # ── RANK ALL ANALOGS ──────────────────────────────────────────────────────

    def rank_analogs(
        self, query_features: dict, top_n: int = None
    ) -> pd.DataFrame:
        """
        Return all (or top_n) database days sorted by Euclidean distance.

        Columns: Rank, Date, Distance,
                 <all feature columns>,
                 density_<alt>km [kg/m³], atomic_O_<alt>km [m⁻³]
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        n = len(self.dates) if top_n is None else min(top_n, len(self.dates))
        q_sc  = self._normalise_query(query_features)
        dists = self._euclidean_distances(q_sc)
        idx   = np.argsort(dists)[:n]

        rows = []
        for rank, i in enumerate(idx, start=1):
            row = {
                "Rank"    : rank,
                "Date"    : pd.Timestamp(self.dates[i]),
                "Distance": float(dists[i]),
            }
            for ci, col in enumerate(self.feature_cols):
                row[col] = float(self.features_raw[i, ci])
            for j, alt in enumerate(self.altitudes_km):
                row[f"density_{alt}km [kg/m³]"]  = float(self.density_db[i, j])
                row[f"atomic_O_{alt}km [m⁻³]"]   = float(self.atomic_O_db[i, j])
            rows.append(row)
        return pd.DataFrame(rows)

    # ── PLOTS ─────────────────────────────────────────────────────────────────

    def plot_analog_bars(
        self,
        query_features: dict,
        alt_km:     float,
        k:          int = 20,
        output_dir: str = ".",
    ):
        """
        Horizontal bar chart of the top-k analog values for both outputs,
        ordered by distance (best match at top), with prediction in red.
        """
        res     = self.predict(query_features, alt_km, k=k)
        labels  = [str(pd.Timestamp(d))[:10] for d in res["analog_dates"]]
        colours = plt.cm.viridis_r(np.linspace(0.1, 0.9, k))

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, k * 0.4)))
        pairs = [
            (res["density_analogs"],  res["density_pred"],  res["density_sigma"],
             "Mass Density",            "kg/m³"),
            (res["atomic_O_analogs"], res["atomic_O_pred"], res["atomic_O_sigma"],
             "Atomic O Number Density", "m⁻³"),
        ]
        for ax, (vals, pred, sigma, title, unit) in zip(axes, pairs):
            y = np.arange(k)
            ax.barh(y, vals, color=colours, edgecolor="grey", linewidth=0.3)
            ax.set_yticks(y)
            ax.set_yticklabels(
                [f"#{r+1:3d}  {lbl}  d={d:.3f}"
                 for r, (lbl, d) in enumerate(zip(labels, res["distances"]))],
                fontsize=8,
            )
            ax.invert_yaxis()
            ax.axvline(pred, color="crimson", lw=2.5, ls="--",
                       label=f"Prediction: {pred:.4g} {unit}\n"
                             f"±{sigma:.3g} {unit}  (1σ across analogs)")
            ax.set_xlabel(f"{title} [{unit}]", fontsize=11)
            ax.set_title(f"{title} at {alt_km} km | Top-{k} Analogs",
                         fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(axis="x", alpha=0.35)

        safe = self.model_name.replace(" ", "_").replace("-", "_")
        plt.suptitle(
            f"{self.model_name} Analog Forecast  |  alt={alt_km} km  |  "
            f"method={res['method']}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"analog_bars_{safe}_{int(alt_km)}km.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Saved: {path}")

    def plot_analog_timeseries(
        self,
        query_features: dict,
        alt_km:     float,
        k:          int = 20,
        output_dir: str = ".",
    ):
        """
        Full historical density and atomic O records with the top-k analog
        dates highlighted, coloured by rank (darkest = closest match).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        res  = self.predict(query_features, alt_km, k=k)
        alts = np.array(self.altitudes_km)

        if alt_km in alts:
            ai           = int(np.argwhere(alts == alt_km)[0])
            density_all  = self.density_db[:, ai]
            atomic_O_all = self.atomic_O_db[:, ai]
        else:
            density_all  = np.array([
                np.interp(alt_km, alts, self.density_db[i, :])
                for i in range(len(self.dates))
            ])
            atomic_O_all = np.array([
                np.interp(alt_km, alts, self.atomic_O_db[i, :])
                for i in range(len(self.dates))
            ])

        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        pairs = [
            (density_all,  res["density_analogs"],  res["density_pred"],
             "Mass Density",            "kg/m³"),
            (atomic_O_all, res["atomic_O_analogs"], res["atomic_O_pred"],
             "Atomic O Number Density", "m⁻³"),
        ]
        for ax, (full, analogs, pred, label, unit) in zip(axes, pairs):
            ax.plot(
                pd.DatetimeIndex(self.dates), full,
                color="steelblue", lw=0.6, alpha=0.75, label="Historical record",
            )
            sc = ax.scatter(
                pd.DatetimeIndex(res["analog_dates"]),
                analogs,
                c=np.arange(k), cmap="Reds_r", s=80, zorder=5,
                edgecolors="black", linewidths=0.5,
                label=f"Top-{k} analogs",
            )
            plt.colorbar(sc, ax=ax, label="Analog rank (1 = closest)")
            ax.axhline(pred, color="crimson", lw=2, ls="--",
                       label=f"Prediction: {pred:.4g} {unit}")
            ax.set_ylabel(f"{label} [{unit}]")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Date")
        safe = self.model_name.replace(" ", "_").replace("-", "_")
        plt.suptitle(
            f"{self.model_name} Analog Locations in Historical Record  |  "
            f"{alt_km} km",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        path = os.path.join(
            output_dir, f"analog_timeseries_{safe}_{int(alt_km)}km.png"
        )
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Saved: {path}")

    # ── SAVE / LOAD ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Serialise the fitted predictor to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[{self.model_name} AnalogPredictor] Saved → {path}")

    @staticmethod
    def load(path: str) -> "AtmosphericAnalogPredictor":
        """Deserialise a previously saved predictor."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[AnalogPredictor] Loaded ← {path}")
        return obj


# =============================================================================
# SECTION 3 – INTERACTIVE TERMINAL QUERY
# =============================================================================

_FEATURE_DESCRIPTIONS = {
    "F107"    : "F10.7 solar radio flux – current day             [sfu]",
    "F107A"   : "F10.7A – 81-day centred mean (MSIS notation)     [sfu]",
    "F107m"   : "F10.7m – 81-day centred mean (DTM notation)      [sfu]",
    "Ap_daily": "Ap – daily planetary geomagnetic amplitude index  [2nT]",
    "kp1"     : "Kp1 – instantaneous Kp (e.g. 09-12 UTC slot)",
    "kp2"     : "Kp2 – daily mean Kp",
    "doy_sin" : "sin(2π × DOY / 365.25)   e.g. 1 Jan ≈  0.0171",
    "doy_cos" : "cos(2π × DOY / 365.25)   e.g. 1 Jan ≈  0.9999",
}


def _doy_sincos(date_str: str):
    """Return (sin, cos) of day-of-year for a 'YYYY-MM-DD' string."""
    ts  = pd.Timestamp(date_str)
    doy = float(ts.day_of_year)
    return np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)


def terminal_query(
    predictor:  AtmosphericAnalogPredictor,
    output_dir: str = ".",
):
    """
    Interactive command-line interface for querying the analog predictor.

    For doy_sin / doy_cos, enter a date string (YYYY-MM-DD) to have the
    values computed automatically.  Results can be saved and/or plotted.
    """
    print("\n" + "=" * 72)
    print(f"  {predictor.model_name} ANALOG FORECASTER"
          "  –  Interactive Terminal Query")
    print("=" * 72)
    print(f"  Database  : {len(predictor.dates)} days  "
          f"({str(predictor.dates[0])[:10]} → "
          f"{str(predictor.dates[-1])[:10]})")
    print(f"  Features  : {predictor.feature_cols}")
    print(f"  Altitudes : {predictor.altitudes_km} km  "
          "(other values will be interpolated)")
    print("  Type 'q' at any prompt to exit.\n")

    while True:
        print("\n--- New Query ---")
        query = {}

        for col in predictor.feature_cols:
            desc = _FEATURE_DESCRIPTIONS.get(col, col)

            # Shortcut: accept a date string to auto-compute doy_sin and doy_cos
            if col == "doy_sin":
                raw = input(
                    "  Target date  "
                    "[YYYY-MM-DD  (auto-fills doy_sin & doy_cos)"
                    " OR press Enter to enter manually]: "
                ).strip()
                if raw.lower() == "q":
                    print("Exiting."); return
                if raw:
                    try:
                        s, c = _doy_sincos(raw)
                        query["doy_sin"] = s
                        query["doy_cos"] = c
                        print(f"  doy_sin = {s:.6f},  doy_cos = {c:.6f}")
                        continue
                    except Exception:
                        print("  Could not parse date – entering values manually.")

            if col == "doy_cos" and "doy_cos" in query:
                continue   # already filled by the date shortcut

            while True:
                raw = input(f"  {col:12s}  [{desc}]: ").strip()
                if raw.lower() == "q":
                    print("Exiting."); return
                try:
                    query[col] = float(raw); break
                except ValueError:
                    print("  ⚠  Please enter a numeric value.")

        # Altitude
        while True:
            raw = input(
                f"\n  alt_km   "
                f"[trained: {predictor.altitudes_km} km, "
                "or any value for interpolation]: "
            ).strip()
            if raw.lower() == "q":
                print("Exiting."); return
            try:
                alt_km = float(raw); break
            except ValueError:
                print("  ⚠  Please enter a number.")

        # k
        k_raw = input("  k        [analogs to use, default=20]: ").strip()
        k     = int(k_raw) if k_raw.isdigit() else 20

        # method
        m_raw = input(
            "  method   [inverse_distance / mean / best, "
            "default=inverse_distance]: "
        ).strip()
        method = (m_raw if m_raw in ("inverse_distance", "mean", "best")
                  else "inverse_distance")

        # ── Run prediction ─────────────────────────────────────────────────
        result = predictor.predict(query, alt_km, k=k, method=method)

        print("\n" + "-" * 66)
        print(f"  RESULT  |  alt={alt_km} km  |  k={k}  |  method={method}")
        print("-" * 66)
        print(f"  Mass Density       ρ  = "
              f"{result['density_pred']:.6e} kg/m³"
              f"   ±{result['density_sigma']:.3e}  (1σ)")
        print(f"  Atomic O Density  n_O = "
              f"{result['atomic_O_pred']:.6e} m⁻³"
              f"   ±{result['atomic_O_sigma']:.3e}  (1σ)")
        print(f"\n  Top-{k} Analog Dates:")
        print(f"  {'#':>4}  {'Date':>12}  {'Dist':>8}  "
              f"{'ρ [kg/m³]':>14}  {'n_O [m⁻³]':>14}  {'Weight':>8}")
        print("  " + "-" * 70)
        for i, (d, dist, rho, nO, w) in enumerate(zip(
            result["analog_dates"],
            result["distances"],
            result["density_analogs"],
            result["atomic_O_analogs"],
            result["weights"],
        ), start=1):
            print(
                f"  {i:>4d}  {str(pd.Timestamp(d))[:10]:>12}  "
                f"{dist:>8.4f}  {rho:>14.4e}  {nO:>14.4e}  {w:>8.4f}"
            )

        # Optional outputs
        if input("\n  Save ranked table to CSV? [y/n]: ").strip().lower() == "y":
            table    = predictor.rank_analogs(query, top_n=k)
            safe     = predictor.model_name.replace(" ", "_").replace("-", "_")
            csv_path = os.path.join(
                output_dir, f"analog_ranked_{safe}_{int(alt_km)}km.csv"
            )
            table.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

        if input("  Plot analog bar chart? [y/n]: ").strip().lower() == "y":
            predictor.plot_analog_bars(
                query, alt_km, k=k, output_dir=output_dir
            )

        if input(
            "  Plot analog positions on historical record? [y/n]: "
        ).strip().lower() == "y":
            predictor.plot_analog_timeseries(
                query, alt_km, k=k, output_dir=output_dir
            )

        if input("\n  Another query? [y/n]: ").strip().lower() != "y":
            print("Done."); break