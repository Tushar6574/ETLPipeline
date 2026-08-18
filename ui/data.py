"""Data access layer for the dashboard.

Loads the committed pipeline artefacts (``output/cwc_river_water_levels.csv``
and ``output/dataset-metadata.json``) with Streamlit caching, and provides the
on-demand **Refresh now** action that re-runs the ETL pipeline incrementally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_paths() -> Tuple[Path, Path]:
    """Canonical output CSV and metadata paths inside the repository."""
    out = REPO_ROOT / "output"
    return out / "cwc_river_water_levels.csv", out / "dataset-metadata.json"


@st.cache_data(show_spinner=False, ttl=3600)
def load_dataset(csv_path: str = "", metadata_path: str = "") -> Tuple[pd.DataFrame, dict]:
    """Load the published dataset + metadata (cached for an hour)."""
    csv_path = csv_path or str(default_paths()[0])
    metadata_path = metadata_path or str(default_paths()[1])
    if not Path(csv_path).exists():
        return pd.DataFrame(), {}
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    meta = {}
    if Path(metadata_path).exists():
        with open(metadata_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    return df, meta


def latest_reading(df: pd.DataFrame) -> pd.DataFrame:
    """Last observation per station (empty frame when no data)."""
    if df is None or df.empty:
        return pd.DataFrame()
    ordered = df.sort_values("timestamp")
    return ordered.groupby("station_code", sort=True).tail(1).reset_index(drop=True)


def station_meta_map(meta: dict) -> dict:
    """Map station_code -> {name, basin, lat, lon, thresholds, records}."""
    out = {}
    for s in meta.get("monitored_stations", []):
        out[s["station_code"]] = s
    return out


def refresh_dataset() -> Tuple[pd.DataFrame, dict]:
    """Run the incremental ETL against the live NWDP API and reload.

    Uses the pipeline's watermark logic, so only records newer than the last
    published reading are fetched. On Streamlit Cloud the container filesystem
    is ephemeral, so refreshed data lasts for the current session only.
    """
    import main  # deferred: keeps this module importable without the ETL stack

    st.spinner("Refreshing from NWDP (incremental)...")
    log_lines: list = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
            log_lines.append(self.format(record))

    handler = _Capture()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        main.run_etl(use_mock=False)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
        raise RuntimeError(f"Refresh failed: {exc}\n\n" + "\n".join(log_lines[-40:])) from exc
    finally:
        root.removeHandler(handler)

    st.cache_data.clear()
    csv_path, metadata_path = default_paths()
    df, meta = load_dataset(str(csv_path), str(metadata_path))
    return df, meta


__all__ = [
    "REPO_ROOT",
    "default_paths",
    "load_dataset",
    "latest_reading",
    "station_meta_map",
    "refresh_dataset",
]
