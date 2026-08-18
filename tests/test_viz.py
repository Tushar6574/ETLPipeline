import numpy as np
import pandas as pd

from ui.forecast import ForecastResult
from ui.viz import (
    build_risk_map,
    plot_forecast_fan,
    plot_level_history,
    plot_status_bars,
    status_color,
)


def _history_series():
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    return pd.Series(np.linspace(80.0, 81.0, 48), index=idx)


def _stations_df():
    return pd.DataFrame(
        {
            "station_code": ["CAVMSI", "YMNCKT"],
            "timestamp": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02")],
            "water_level": [81.0, 137.2],
            "flood_status": ["WARNING", "WARNING"],
            "latitude": [10.95, 25.2],
            "longitude": [78.8, 80.9],
        }
    )


def test_status_color_mapping():
    assert status_color("HIGH_FLOOD_EXCEEDED") == "#d62728"
    assert status_color("normal") == "#2ca02c"
    assert status_color("bogus") == "#7f7f7f"


def test_plot_level_history_returns_figure():
    df = pd.DataFrame(
        {
            "station_code": ["CAVMSI"] * 48,
            "timestamp": pd.date_range("2026-01-01", periods=48, freq="h"),
            "water_level": np.linspace(80.0, 81.0, 48),
        }
    )
    fig = plot_level_history(df, "CAVMSI", {"station_name": "Musiri"}, window_days=7)
    assert len(fig.data) >= 1


def test_plot_forecast_fan_returns_figure():
    result = ForecastResult(
        station_code="CAVMSI",
        engine="xgboost",
        horizon=24,
        timestamps=pd.date_range("2026-01-03", periods=24, freq="h"),
        mean=np.full(24, 81.0),
        low=np.full(24, 80.5),
        high=np.full(24, 81.5),
        last_level=81.0,
    )
    fig = plot_forecast_fan(result, _history_series(), {"station_name": "Musiri"})
    assert len(fig.data) >= 3


def test_plot_status_bars_returns_figure():
    fig = plot_status_bars(_stations_df())
    assert len(fig.data) == 1


def test_build_risk_map_returns_folium_map():
    m = build_risk_map(_stations_df(), forecasts={}, meta_map={})
    assert m.location is not None
    assert len(m._children) > 0