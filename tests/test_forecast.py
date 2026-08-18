import numpy as np
import pytest

from ui.forecast import (
    _load_timesfm,
    forecast_station,
    hours_to_threshold,
    risk_score,
    timesfm_available,
)


def test_hours_to_threshold_crossing():
    mean = np.array([10.0, 11.0, 12.0, 13.0])
    assert hours_to_threshold(mean, 11.5) == 3
    assert hours_to_threshold(mean, 99.0) is None


def test_risk_score_bounds_and_components():
    res = risk_score(10.0, np.array([11.0, 12.0, 13.0, 14.0]), danger_level=12.5)
    assert 0.0 <= res["score"] <= 100.0
    assert res["crossing"] > 0.0  # mean crosses danger
    assert res["proximity"] > 0.0
    assert res["surge"] >= 0.0


def test_risk_score_nan_inputs():
    res = risk_score(np.nan, np.array([1.0, 2.0]), 3.0)
    assert np.isnan(res["score"])


def test_xgboost_forecast_end_to_end(synthetic_df):
    """The committed fallback model produces a sane, band-valid forecast."""
    from config import DEFAULT_STATIONS
    from pathlib import Path

    models_dir = Path(__file__).resolve().parents[1] / "models"
    if not (models_dir / "xgb_q50.json").exists():
        pytest.skip("committed XGBoost model not present")

    for code in ["CAVMSI", "YMNCKT"]:
        res = forecast_station(
            synthetic_df, code, horizon=24, engine="xgboost", models_dir=str(models_dir)
        )
        assert res.station_code == code
        assert len(res.mean) == 24
        assert np.all(res.low <= res.mean + 1e-9)
        assert np.all(res.mean <= res.high + 1e-9)
        assert np.all(np.isfinite(res.mean))
        station = {s.station_code: s for s in DEFAULT_STATIONS}[code]
        # Forecast should stay near the station's normal regime.
        assert abs(res.mean[0] - res.last_level) < 5.0


def test_timesfm_forecast_end_to_end(synthetic_df):
    """Optional: requires timesfm installed and the checkpoint downloadable."""
    if not timesfm_available():
        pytest.skip("timesfm not installed")
    model = _load_timesfm()
    if model is None:
        pytest.skip("timesfm checkpoint unavailable")
    res = forecast_station(
        synthetic_df, "CAVMSI", horizon=24, engine="timesfm", timesfm_model=model
    )
    assert len(res.mean) == 24
    assert np.all(np.isfinite(res.mean))
    assert np.all(res.low <= res.mean + 1e-9)
    assert np.all(res.mean <= res.high + 1e-9)