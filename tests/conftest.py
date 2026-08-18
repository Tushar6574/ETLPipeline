import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def synthetic_df():
    """Small deterministic hourly dataset for two configured stations."""
    import numpy as np
    import pandas as pd

    from config import DEFAULT_STATIONS

    codes = ["CAVMSI", "YMNCKT"]
    stations = {s.station_code: s for s in DEFAULT_STATIONS}
    rng = np.random.default_rng(7)
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for code in codes:
        station = stations[code]
        base = station.warning_level - 0.4
        hours = pd.date_range(start, periods=300, freq="h")
        levels = base + 0.05 * rng.normal(size=len(hours))
        for i, ts in enumerate(hours):
            rows.append(
                {
                    "station_code": code,
                    "timestamp": ts,
                    "water_level": float(levels[i]),
                }
            )
    return pd.DataFrame(rows)