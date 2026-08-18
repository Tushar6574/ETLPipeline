import numpy as np
import pandas as pd

from transform import derive_flood_status


def test_flood_status_buckets():
    df = pd.DataFrame(
        {
            "water_level": [10.0, 15.0, 20.0, 30.0],
            "warning_level": [12.0, 12.0, 12.0, 12.0],
            "danger_level": [18.0, 18.0, 18.0, 18.0],
            "high_flood_level": [25.0, 25.0, 25.0, 25.0],
        }
    )
    out = derive_flood_status(df)
    assert list(out) == ["NORMAL", "WARNING", "DANGER", "HIGH_FLOOD_EXCEEDED"]


def test_flood_status_nan_when_thresholds_missing():
    df = pd.DataFrame(
        {"water_level": [10.0, 15.0], "warning_level": [np.nan, np.nan]}
    )
    out = derive_flood_status(df)
    assert out.isna().all()


def test_flood_status_empty_frame():
    out = derive_flood_status(pd.DataFrame())
    assert len(out) == 0