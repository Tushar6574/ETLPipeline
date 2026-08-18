import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_STATIONS
from ui.features import (
    category_maps,
    clip_values,
    encode_categories,
    feature_columns,
    level_band,
    prepare_series,
    series_to_frame,
)


def test_clip_values_tames_isolated_spike():
    idx = pd.date_range("2026-01-01", periods=100, freq="h")
    s = pd.Series(np.full(100, 10.0), index=idx)
    s.iloc[50] = 1000.0
    clipped = clip_values(s)
    assert clipped.max() < 100.0
    assert clipped.iloc[50] < 100.0


def test_prepare_series_unknown_station_is_empty():
    df = pd.DataFrame({"station_code": ["CAVMSI"], "timestamp": ["2026-01-01"], "water_level": [1.0]})
    out = prepare_series(df, "NOPE", band=(0.0, 10.0))
    assert len(out) == 0


def test_prepare_series_removes_sustained_junk_plateau():
    """A multi-day near-constant level above the band is dropped, not bridged."""
    idx = pd.date_range("2026-01-01", periods=168 + 96, freq="h")
    values = np.full(len(idx), 90.0)  # normal regime
    values[168:] = 703.0  # junk plateau pinned at the band cap
    df = pd.DataFrame(
        {"station_code": "NRMMDL", "timestamp": idx, "water_level": values}
    )
    band = level_band(437.2, 437.8, 439.405)
    out = prepare_series(df, "NRMMDL", band=band)
    assert len(out) <= 168 + 1
    assert out.max() < 500.0


def test_prepare_series_band_clips_out_of_range():
    idx = pd.date_range("2026-01-01", periods=50, freq="h")
    df = pd.DataFrame(
        {"station_code": "CAVMSI", "timestamp": idx, "water_level": np.full(50, 999.0)}
    )
    band = level_band(82.115, 83.115, 86.175)
    out = prepare_series(df, "CAVMSI", band=band)
    assert out.max() <= band[1] + 1e-9


def test_series_to_frame_schema():
    idx = pd.date_range("2026-01-01", periods=300, freq="h")
    s = pd.Series(np.linspace(80.0, 82.0, 300), index=idx)
    frame = series_to_frame(s, "CAVMSI", "Cauvery", "telemetry")
    assert "target" not in frame
    for col in feature_columns():
        assert col in frame.columns, col
    assert frame["lag_1"].notna().sum() > 100


def test_category_maps_cover_all_stations():
    maps = category_maps()
    assert set(maps["station_code"]) == {s.station_code for s in DEFAULT_STATIONS}
    # deterministic across calls
    assert maps == category_maps()


def test_encode_categories_unknown_to_minus_one():
    df = pd.DataFrame({"station_code": ["CAVMSI", "NOPE"]})
    encode_categories(df, category_maps())
    assert df.loc[0, "station_code"] >= 0
    assert df.loc[1, "station_code"] == -1