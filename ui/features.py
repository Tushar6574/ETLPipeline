"""Shared feature engineering + series sanitisation for the forecasting layer.

Pure pandas/numpy - no I/O, no Streamlit - so it can be imported both by the
interactive app (``ui.forecast``) and by the offline training script
(``scripts/train_xgboost.py``) without pulling in any UI dependency.

Design notes
------------
* Forecasts run on a **regular hourly grid**: raw telemetry is resampled,
  outliers clipped via a rolling-median / MAD fence (protects against the known
  NWDP junk spikes, e.g. Musiri ~677 m or Mandla ~1394 m), and remaining gaps
  are linearly interpolated.
* ``XGBRegressor`` uses a **global** model trained on all stations at once, so
  categorical encodings (station / basin / kind) must be identical at training
  and inference time. :func:`category_maps` derives them deterministically from
  ``DEFAULT_STATIONS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_STATIONS

LEVEL_COL = "water_level"
TIMESTAMP_COL = "timestamp"
STATION_CODE_COL = "station_code"

# Features used by the XGBoost model (kept in a stable order).
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLL_WINDOWS = [(24, "mean"), (24, "std"), (72, "mean"), (168, "mean")]


def clip_values(
    series: pd.Series, k: float = 8.0, window: int = 48
) -> pd.Series:
    """Clip outliers to a rolling-median / MAD fence (values stay in metres).

    Uses the robust median absolute deviation (MAD) around a trailing rolling
    median so genuine flood surges are preserved while isolated telemetry
    spikes are pulled back to the fence.
    """
    med = series.rolling(window, min_periods=1).median()
    dev = (series - med).abs().rolling(window, min_periods=1).median()
    spread = 1.4826 * dev + 1e-9
    lo = med - k * spread
    hi = med + k * spread
    return series.clip(lo, hi)


def level_band(warning: float, danger: float, hfl: float):
    """Physically plausible level band from a station's thresholds.

    Known NWDP telemetry carries sustained junk plateaus (e.g. Mandla ~1394 m
    against a ~437 m danger level) that evade rolling-median clipping. A hard
    band around the official thresholds removes those regimes before both
    training and inference. Returns ``None`` when no threshold is usable.
    """
    vals = [float(v) for v in (warning, danger, hfl) if np.isfinite(v) and v > 0]
    if not vals:
        return None
    return (0.5 * min(vals), 1.6 * max(vals))


def _junk_regime_mask(
    s: pd.Series, band, window: int = 24, min_hours: int = 48, buffer: int = 36
) -> pd.Series:
    """Mask sustained junk regimes plus a safety buffer before each run.

    NWDP telemetry carries both flat multi-day locks and chaotic junk tails of
    wild spikes (e.g. Mandla ~1205 m against a ~437 m danger level). A rolling
    mean over a short window bridges noisy dips inside a regime, so any run
    lasting >= ``min_hours`` above the band midpoint is junk. The run is
    extended backwards by ``buffer`` hours so the transition warm-up (when the
    rolling mean has not caught up yet) is removed too, and individual values
    pinned exactly at the clip cap are always dropped.
    """
    if band is None or len(s) < 100:
        return pd.Series(False, index=s.index)
    lo, hi = band
    midpoint = lo + 0.5 * (hi - lo)
    smooth = s.rolling(window, min_periods=12).mean()
    cand = smooth > midpoint
    groups = (cand != cand.shift()).cumsum()
    run_len = cand.groupby(groups).transform("size")
    runs = cand & (run_len >= min_hours)
    extended = runs.rolling(buffer, min_periods=1).max().shift(-buffer).fillna(False)
    pinned = s >= hi - 1e-6
    return pd.Series((runs | extended | pinned).values, index=s.index)


def prepare_series(
    df: pd.DataFrame,
    station_code: str,
    level_col: str = LEVEL_COL,
    band=None,
) -> pd.Series:
    """Build a clean, regular hourly series for one station.

    1. Filter to the station and drop missing readings.
    2. Collapse duplicate timestamps and resample to a strict hourly grid.
    3. Clip outliers (rolling-median / MAD fence), then to a threshold-aware
       ``band`` (see :func:`level_band`).
    4. Interpolate normal gaps, then re-apply the junk-regime mask so multi-day
       sensor locks / chaotic junk tails are removed, not bridged.
    5. Drop any remaining NaN so forecasts anchor on real readings only.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    sub = df[df[STATION_CODE_COL] == station_code]
    if level_col not in sub.columns or TIMESTAMP_COL not in sub.columns:
        return pd.Series(dtype="float64")
    ts = pd.to_datetime(sub[TIMESTAMP_COL], errors="coerce", utc=True)
    levels = pd.to_numeric(sub[level_col], errors="coerce")
    s = (
        pd.DataFrame({"ts": ts, "v": levels})
        .dropna(subset=["ts", "v"])
        .set_index("ts")["v"]
        .sort_index()
    )
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample("h").mean()
    s = clip_values(s)
    junk = pd.Series(False, index=s.index)
    if band is not None:
        s = s.clip(lower=band[0], upper=band[1])
        junk = _junk_regime_mask(s, band)
    if s.isna().any():
        s = s.interpolate(method="time")
    s = s.mask(junk)
    s = s.dropna()
    return s.astype("float64")


def series_to_frame(
    series: pd.Series,
    station_code: str,
    basin_name: str,
    kind: str,
) -> pd.DataFrame:
    """Expand an hourly series into the supervised feature frame.

    Each row ``t`` holds lag / rolling / seasonal / categorical features for
    that hour; the target is the level at ``t + 1h`` (recursive multi-step).
    """
    df = pd.DataFrame({"y": series.values}, index=series.index)
    df.index.name = "ts"
    df["hour"] = df.index.hour
    df["dayofyear"] = df.index.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    for lag in LAG_HOURS:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window, agg in ROLL_WINDOWS:
        df[f"roll{window}_{agg}"] = df["y"].rolling(window).agg(agg)
    df["diff_1"] = df["y"].diff()
    df["diff_24"] = df["y"].diff(24)
    df[STATION_CODE_COL] = station_code
    df["basin_name"] = basin_name
    df["kind"] = kind
    return df


def category_maps() -> dict:
    """Deterministic categorical encodings derived from the station catalogue.

    Keys are ``station_code`` / ``basin_name`` / ``kind``; values map the raw
    string to a stable integer code used by the XGBoost model.
    """
    stations = sorted(DEFAULT_STATIONS, key=lambda s: s.station_code)
    return {
        STATION_CODE_COL: {s.station_code: i for i, s in enumerate(stations)},
        "basin_name": {b: i for i, b in enumerate(sorted({s.basin_name for s in stations}))},
        "kind": {k: i for i, k in enumerate(sorted({"telemetry", "manual"}))},
    }


def encode_categories(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """Apply the category maps in place (unknown values -> -1)."""
    for col, mapping in maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype("int32")
    return df


def feature_columns() -> list:
    """Ordered feature column list shared by training and inference."""
    cols = [
        STATION_CODE_COL,
        "basin_name",
        "kind",
        "sin_hour",
        "cos_hour",
        "sin_doy",
        "cos_doy",
    ]
    cols += [f"lag_{lag}" for lag in LAG_HOURS]
    cols += [f"roll{w}_{a}" for w, a in ROLL_WINDOWS]
    cols += ["diff_1", "diff_24"]
    return cols


__all__ = [
    "LEVEL_COL",
    "TIMESTAMP_COL",
    "STATION_CODE_COL",
    "LAG_HOURS",
    "ROLL_WINDOWS",
    "clip_values",
    "level_band",
    "prepare_series",
    "series_to_frame",
    "category_maps",
    "encode_categories",
    "feature_columns",
]
