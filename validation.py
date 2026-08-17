"""Post-run Validation & Quality Reporting.

Runs after the load step and produces:

* duplicate-timestamp checks per station (data-integrity gate),
* null-percentage per column (sensor-dropout visibility),
* an executive summary table of the published dataset.

The checks are informational by default; a duplicate-timestamp violation is
warned loudly but does not abort the pipeline, so a partially bad upstream
feed cannot take the whole refresh down.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from transform import (
    FLOOD_STATUS_COL,
    LATITUDE_COL,
    LONGITUDE_COL,
    STATION_CODE_COL,
    TIMESTAMP_COL,
    WATER_LEVEL_COL,
)

logger = logging.getLogger(__name__)


def check_duplicate_timestamps(df: pd.DataFrame) -> Dict[str, int]:
    """Count timestamp collisions per station.

    Exactly one reading per station per hour is expected, so any station with
    repeated ``(station_code, timestamp)`` pairs indicates upstream duplicates
    that the composite-key dedup did not collapse (e.g. genuinely different
    levels at the same instant).
    """
    if df.empty or STATION_CODE_COL not in df.columns or TIMESTAMP_COL not in df.columns:
        return {}
    dupes = df.duplicated(subset=[STATION_CODE_COL, TIMESTAMP_COL], keep=False)
    if dupes.any():
        logger.warning(
            "Found %d duplicate timestamp(s) across %d row(s) "
            "(station, timestamp) - inspect upstream feed.",
            int(df[dupes].groupby(STATION_CODE_COL).size().sum()),
            int(dupes.sum()),
        )
    counts = (
        df[dupes]
        .groupby(STATION_CODE_COL)
        .size()
        .reindex(df[STATION_CODE_COL].unique())
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    return counts


def null_percentage_per_column(df: pd.DataFrame) -> Dict[str, float]:
    """Share (0-100%) of missing values in each column."""
    if df.empty:
        return {col: 100.0 for col in df.columns}
    return {
        col: round(float(df[col].isna().mean() * 100.0), 2) for col in df.columns
    }


def _executive_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station summary: record count, bounds, extremes and alert mix."""
    if df.empty:
        return pd.DataFrame()
    summary = []
    for code, group in df.groupby(STATION_CODE_COL, sort=True):
        ts = pd.to_datetime(group[TIMESTAMP_COL], errors="coerce", utc=True)
        wl = pd.to_numeric(group.get(WATER_LEVEL_COL), errors="coerce")
        status_counts = group.get(FLOOD_STATUS_COL, pd.Series(dtype=object)).value_counts()
        summary.append(
            {
                "station_code": code,
                "records": len(group),
                "start": ts.min().strftime("%Y-%m-%d %H:%M") if ts.notna().any() else "n/a",
                "end": ts.max().strftime("%Y-%m-%d %H:%M") if ts.notna().any() else "n/a",
                "min_level_m": round(float(wl.min()), 3) if wl.notna().any() else None,
                "max_level_m": round(float(wl.max()), 3) if wl.notna().any() else None,
                "missing_gauge_pct": round(float(wl.isna().mean() * 100.0), 2),
                "status_dist": ", ".join(
                    f"{k}:{int(v)}" for k, v in status_counts.items()
                )
                or "n/a",
            }
        )
    return pd.DataFrame(summary)


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Run all sanity checks and print the executive summary report."""
    dupe_counts = check_duplicate_timestamps(df)
    null_pct = null_percentage_per_column(df)
    summary = _executive_summary_table(df)

    header = "=" * 78
    print(f"\n{header}")
    print("EXECUTIVE SUMMARY - CWC RIVER WATER LEVEL (TELEMETRY + MANUAL)")
    print(header)
    print(f"Total records : {len(df)}")
    print(f"Stations      : {df[STATION_CODE_COL].nunique() if not df.empty else 0}")
    print(f"Duplicate (station,timestamp) rows : {int(df.duplicated(subset=[STATION_CODE_COL, TIMESTAMP_COL]).sum()) if not df.empty else 0}")
    print(header)
    if not summary.empty:
        print(
            summary.to_string(index=False, justify="left")
        )
    print(header)
    print("NULL PERCENTAGE PER COLUMN")
    print(header)
    if null_pct:
        width = max(len(k) for k in null_pct)
        for col, pct in null_pct.items():
            print(f"  {col:<{width}} : {pct:6.2f}%")
    print(header)

    report = {
        "total_records": int(len(df)),
        "stations": int(df[STATION_CODE_COL].nunique()) if not df.empty else 0,
        "duplicate_timestamps_per_station": dupe_counts,
        "null_percentage_per_column": null_pct,
        "per_station_summary": summary.to_dict(orient="records"),
    }
    return report


__all__ = [
    "check_duplicate_timestamps",
    "null_percentage_per_column",
    "validate_dataset",
]