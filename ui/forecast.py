"""Forecasting layer: TimesFM 2.5 (primary) with an XGBoost fallback.

* **TimesFM 2.5** - zero-shot foundation time-series model (``timesfm`` PyPI
  package). The model is heavy (~800 MB download on first use) and only loads
  when the user actually asks for a TimesFM forecast, through a
  ``st.cache_resource`` lazy wrapper.
* **XGBoost fallback** - a small global quantile model trained offline by
  ``scripts/train_xgboost.py`` and committed to ``models/``. Runs anywhere in
  milliseconds and needs no network.

Both engines share the cleaned hourly series from :mod:`ui.features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from config import DEFAULT_STATIONS
from ui.features import (
    STATION_CODE_COL,
    category_maps,
    encode_categories,
    feature_columns,
    level_band,
    prepare_series,
    series_to_frame,
)

# TimesFM 2.5 quantile head columns: col0 = point forecast,
# col1..col9 = q0.1..q0.9, col5 = median.
_QF_MEAN = 0
_QF_LOW = 1  # q0.10
_QF_MED = 5  # q0.50
_QF_HIGH = 9  # q0.90

MAX_CONTEXT = 2048
MAX_HORIZON = 256


@dataclass
class ForecastResult:
    """Container for one station's forecast fan-chart data."""

    station_code: str
    engine: str
    horizon: int
    timestamps: pd.DatetimeIndex
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    last_level: float
    note: str = ""

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "mean": self.mean,
                "low": self.low,
                "high": self.high,
            }
        ).set_index("timestamp")


def station_catalogue() -> dict:
    return {s.station_code: s for s in DEFAULT_STATIONS}


def timesfm_available() -> bool:
    """Best-effort availability probe (no model download)."""
    try:
        import timesfm  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# TimesFM engine
# ---------------------------------------------------------------------------
@dataclass
class _TimesFMWrapper:
    model: object
    note: str


def _load_timesfm() -> Optional[_TimesFMWrapper]:
    """Download/load the TimesFM checkpoint once per app session."""
    try:
        import torch  # noqa: F401
        import timesfm

        torch.set_float32_matmul_precision("high")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=MAX_CONTEXT,
                max_horizon=MAX_HORIZON,
                normalize_inputs=True,
                per_core_batch_size=8,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                fix_quantile_crossing=True,
            )
        )
        return _TimesFMWrapper(model=model, note="TimesFM 2.5 · 200M (torch)")
    except Exception:  # noqa: BLE001
        return None


def _forecast_timesfm(
    series: pd.Series, horizon: int, model: object
) -> ForecastResult:
    context = series.iloc[-MAX_CONTEXT:]
    try:
        point, quantiles = model.forecast(
            horizon=horizon, inputs=[context.values.astype("float32")]
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"TimesFM forecast failed: {exc}") from exc
    point = np.asarray(point, dtype="float64").ravel()
    q = np.asarray(quantiles, dtype="float64")[0]
    mean = point[:horizon]
    low = q[:horizon, _QF_LOW]
    med = q[:horizon, _QF_MED]
    high = q[:horizon, _QF_HIGH]
    # De-noise the band: ensure low <= median <= high at every step.
    low = np.minimum(low, med)
    high = np.maximum(high, med)
    # Re-anchor to the last observed level so the fan starts at reality.
    if len(series) and np.isfinite(series.iloc[-1]):
        offset = series.iloc[-1] - med[0]
        mean = mean + offset
        low = low + offset
        high = high + offset
    return ForecastResult(
        station_code=series.name if isinstance(series.name, str) else "",
        engine="timesfm",
        horizon=horizon,
        timestamps=pd.date_range(series.index[-1] + timedelta(hours=1), periods=horizon, freq="h"),
        mean=mean,
        low=low,
        high=high,
        last_level=float(series.iloc[-1]) if len(series) else float("nan"),
        note="Zero-shot · no retraining needed",
    )


# ---------------------------------------------------------------------------
# XGBoost engine
# ---------------------------------------------------------------------------
def _load_xgb_artifacts(models_dir) -> Dict[str, object]:
    """Load the center booster + per-station residual band offsets."""
    import json

    import xgboost as xgb

    center_path = models_dir / "xgb_q50.json"
    booster = None
    try:
        booster = xgb.Booster()
        booster.load_model(str(center_path))
    except Exception:  # noqa: BLE001
        booster = None
    bands: Dict[str, list] = {}
    bands_path = models_dir / "xgb_bands.json"
    if bands_path.exists():
        try:
            bands = json.loads(bands_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            bands = {}
    return {"q50": booster, "bands": bands}


def _forecast_xgb(
    df: pd.DataFrame,
    station_code: str,
    horizon: int,
    artifacts: Dict[str, object],
    basin_name: str,
    kind: str,
    maps: dict,
    band=None,
) -> ForecastResult:
    booster = artifacts.get("q50")
    if booster is None:
        raise RuntimeError("XGBoost model is missing - run scripts/train_xgboost.py")
    series = prepare_series(df, station_code, band=band)
    min_len = max(168, 2 * 24) + 2
    if len(series) < min_len:
        raise RuntimeError(
            f"Only {len(series)} hours available for {station_code} (need >= {min_len})."
        )
    code_int = int(maps[STATION_CODE_COL][station_code])
    try:
        lo_off, hi_off = (float(v) for v in artifacts["bands"][str(code_int)])
    except Exception:  # noqa: BLE001 - default band if offsets are missing
        lo_off, hi_off = -0.5, 0.5
    work = series
    means, lows, highs = [], [], []
    for _ in range(horizon):
        frame = series_to_frame(work, station_code, basin_name, kind)
        encode_categories(frame, maps)
        X = frame[feature_columns()].iloc[[-1]].astype("float32")
        q50 = float(booster.predict(xgb.DMatrix(X))[0])
        means.append(q50)
        lows.append(min(q50 + lo_off, q50))
        highs.append(max(q50 + hi_off, q50))
        next_ts = work.index[-1] + timedelta(hours=1)
        work = pd.concat([work, pd.Series([q50], index=[next_ts])])
    return ForecastResult(
        station_code=station_code,
        engine="xgboost",
        horizon=horizon,
        timestamps=pd.date_range(series.index[-1] + timedelta(hours=1), periods=horizon, freq="h"),
        mean=np.asarray(means),
        low=np.asarray(lows),
        high=np.asarray(highs),
        last_level=float(series.iloc[-1]) if len(series) else float("nan"),
        note="Global GBM (center) + per-station residual band",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def forecast_station(
    df: pd.DataFrame,
    station_code: str,
    horizon: int = 72,
    engine: str = "timesfm",
    timesfm_model: Optional[object] = None,
    models_dir: Optional[str] = None,
) -> ForecastResult:
    """Forecast one station for ``horizon`` hours.

    ``engine``: ``"timesfm"`` | ``"xgboost"``. Pass the cached TimesFM model
    (or None) and the models directory to reuse session-wide resources.
    """
    station = station_catalogue().get(station_code)
    horizon = max(1, min(int(horizon), MAX_HORIZON))
    band = (
        level_band(station.warning_level, station.danger_level, station.high_flood_level)
        if station
        else None
    )
    if engine == "timesfm":
        if timesfm_model is None:
            timesfm_model = _load_timesfm()
        if timesfm_model is None:
            raise RuntimeError(
                "TimesFM is not available (dependency or checkpoint download failed). "
                "Switch the engine to XGBoost or install `requirements-ui.txt`."
            )
        series = prepare_series(df, station_code, band=band)
        result = _forecast_timesfm(series, horizon, timesfm_model.model)
    else:
        from pathlib import Path

        models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parents[1] / "models"
        artifacts = _load_xgb_artifacts(models_dir)
        basin = station.basin_name if station else ""
        kind = station.kind if station else "telemetry"
        result = _forecast_xgb(
            df, station_code, horizon, artifacts, basin, kind, category_maps(), band=band
        )
    result.station_code = station_code
    return result


def hours_to_threshold(mean: np.ndarray, threshold: float) -> Optional[int]:
    """First hour index at which the forecast crosses ``threshold`` (or None)."""
    if not np.isfinite(threshold):
        return None
    idx = np.where(mean >= threshold)[0]
    return int(idx[0] + 1) if len(idx) else None


def risk_score(
    last_level: float,
    mean: np.ndarray,
    danger_level: float,
    hours_between: int = 1,
) -> Dict[str, float]:
    """Interpretable flood-risk heuristic in 0..100.

    Blend of proximity to the danger threshold, whether the forecast crosses
    it, and the steepest 24 h rise (surge pressure). Purely a dashboard
    heuristic - not a hydrological model.
    """
    if not np.isfinite(last_level) or not np.isfinite(danger_level):
        return {"score": float("nan"), "proximity": float("nan"), "crossing": 0.0, "surge": 0.0}
    warning_band = max(0.5, 0.4 * abs(danger_level - last_level))
    peak = float(np.max(mean)) if len(mean) else last_level
    min_dist = danger_level - max(last_level, peak)
    proximity = float(np.clip(70.0 * (1.0 - min_dist / max(warning_band, 1e-6)), 0.0, 70.0))
    if min_dist <= 0:
        first = hours_to_threshold(mean, danger_level)
        urgency = 1.0 if first is None else float(np.clip(1.0 - (first * hours_between) / 336.0, 0.0, 1.0))
        crossing = 30.0 * (0.5 + 0.5 * urgency)
    else:
        crossing = 0.0
    rises = np.diff(mean) if len(mean) > 1 else np.array([])
    if len(rises) >= 24:
        surge = float(np.clip(np.sum(rises[-24:]) / warning_band, 0.0, 1.0))
    else:
        surge = float(np.clip(np.max(rises, initial=0.0) * 12 / warning_band, 0.0, 1.0))
    surge_bonus = 15.0 * surge
    score = float(np.clip(proximity + crossing + surge_bonus, 0.0, 100.0))
    return {"score": score, "proximity": proximity, "crossing": crossing, "surge": surge_bonus}


__all__ = [
    "ForecastResult",
    "MAX_CONTEXT",
    "MAX_HORIZON",
    "station_catalogue",
    "timesfm_available",
    "forecast_station",
    "hours_to_threshold",
    "risk_score",
]
