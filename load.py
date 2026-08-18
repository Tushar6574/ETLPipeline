"""Loading & Metadata Generation layer.

Responsibilities
----------------
* **Atomic file writes** - data is written to a ``*.tmp`` sibling and then
  atomically renamed via ``os.replace``, so a crash mid-write can never leave
  a truncated CSV / JSON as the published artefact.
* **Idempotent merge** - new records are concatenated with the existing CSV,
  deduplicated on the composite key and trimmed to the retention window before
  the whole dataset is rewritten.
* **Open-data metadata** - emits ``dataset-metadata.json`` documenting
  temporal bounds, monitored basins / stations, record counts and the
  distribution of hydrological alert statuses.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import APIConfig, PipelineConfig, StationConfig
from transform import (
    BASIN_NAME_COL,
    COLUMN_ORDER,
    DANGER_LEVEL_COL,
    DEDUP_KEY_COL,
    FLOOD_STATUS_COL,
    HFL_COL,
    LATITUDE_COL,
    LONGITUDE_COL,
    STATION_CODE_COL,
    STATION_NAME_COL,
    TIMESTAMP_COL,
    WARNING_LEVEL_COL,
    WATER_LEVEL_COL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic I/O helpers
# ---------------------------------------------------------------------------


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` via an atomic temp-file + rename.

    Guarantees the destination either contains the complete new dataset or
    the previous complete dataset - never a partially written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    logger.info("Atomically wrote %d rows to %s", len(df), path)


def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    """Serialise ``obj`` to ``path`` atomically with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    logger.info("Atomically wrote metadata to %s", path)


# ---------------------------------------------------------------------------
# Dataset merge / retention
# ---------------------------------------------------------------------------


def merge_with_existing(
    new_df: pd.DataFrame,
    csv_path: Path,
    retention_days: int,
    dedup_col: str = DEDUP_KEY_COL,
) -> pd.DataFrame:
    """Merge fresh records into the persisted dataset idempotently.

    * Reads the existing CSV (empty DataFrame when absent).
    * Concatenates with new records and drops duplicate composite keys.
    * Applies the rolling retention window relative to the newest reading.
    """
    existing = pd.DataFrame()
    if csv_path.exists():
        existing = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d existing rows from %s", len(existing), csv_path)

    if existing.empty and new_df.empty:
        return new_df.copy()

    merged = pd.concat([existing, new_df], ignore_index=True)

    # Persist only the canonical schema. Legacy raw telemetry extras (SlNo,
    # _id, State, District, ...) carried by the existing CSV are dropped so the
    # published schema stays stable and mixed-dtype columns cannot resurface.
    merged = merged[[c for c in COLUMN_ORDER if c in merged.columns]]

    # Normalise dtypes after concat: existing CSV strings + fresh datetime64
    # objects otherwise produce an object column that breaks serialisation.
    if TIMESTAMP_COL in merged.columns:
        merged[TIMESTAMP_COL] = pd.to_datetime(
            merged[TIMESTAMP_COL], errors="coerce", utc=True
        )
    for col in (WATER_LEVEL_COL, LATITUDE_COL, LONGITUDE_COL):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    if dedup_col in merged.columns:
        merged = merged.drop_duplicates(subset=[dedup_col], keep="last")
        logger.info("After composite-key dedup: %d rows", len(merged))

    if TIMESTAMP_COL in merged.columns and retention_days > 0:
        ts = pd.to_datetime(merged[TIMESTAMP_COL], errors="coerce", utc=True)
        if ts.notna().any():
            cutoff = ts.max() - pd.Timedelta(days=retention_days)
            before = len(merged)
            merged = merged[ts >= cutoff].copy()
            logger.info(
                "Retention window [%s, %s]: dropped %d old row(s)",
                cutoff.isoformat(),
                ts.max().isoformat(),
                before - len(merged),
            )

    merged = merged.sort_values([STATION_CODE_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Dataset publishing
# ---------------------------------------------------------------------------


def write_partitions(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Write one partitioned CSV per station into ``partition_dir``."""
    if df.empty or STATION_CODE_COL not in df.columns:
        return
    cfg.partition_dir.mkdir(parents=True, exist_ok=True)
    for code, group in df.groupby(STATION_CODE_COL, sort=True):
        atomic_write_csv(group, cfg.partition_dir / f"{code}.csv")


def write_dataset(
    df: pd.DataFrame, cfg: PipelineConfig, partition: bool = True
) -> None:
    """Publish the canonical dataset (CSV + optional Parquet + partitions)."""
    atomic_write_csv(df, cfg.csv_path)
    if cfg.write_parquet:
        cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cfg.parquet_path, index=False)
        logger.info("Wrote %d rows to %s", len(df), cfg.parquet_path)
    if partition:
        write_partitions(df, cfg)


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------


def _temporal_bounds(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty or TIMESTAMP_COL not in df.columns:
        return {"start_date": "unknown", "end_date": "unknown"}
    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
    if ts.notna().any():
        return {
            "start_date": ts.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date": ts.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    return {"start_date": "unknown", "end_date": "unknown"}


def _alert_distribution(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or FLOOD_STATUS_COL not in df.columns:
        return {}
    counts = df[FLOOD_STATUS_COL].dropna().value_counts().to_dict()
    return {k: int(v) for k, v in counts.items()}


def _station_list(
    df: pd.DataFrame, stations: StationConfig
) -> List[Dict[str, Any]]:
    """One entry per monitored station, with catalogue thresholds."""
    out = []
    for code in stations.station_codes:
        s = stations.get(code)
        out.append(
            {
                "station_code": s.station_code,
                "station_name": s.station_name,
                "basin_name": s.basin_name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "warning_level_m": s.warning_level,
                "danger_level_m": s.danger_level,
                "high_flood_level_m": s.high_flood_level,
                "records": int(
                    (df[STATION_CODE_COL] == s.station_code).sum()
                    if not df.empty and STATION_CODE_COL in df.columns
                    else 0
                ),
            }
        )
    return out


def _source_list(api: Optional[APIConfig]) -> List[Dict[str, str]]:
    """Describe the upstream NWDP resources this run draws from."""
    if api is None:
        return []
    return [
        {
            "provider": "CWC / National Water Data Portal (NWDP)",
            "basin": res.basin,
            "kind": res.kind,
            "time_range": res.time_range,
            "resource_id": res.resource_id,
        }
        for res in api.resolve_resources()
    ]


def generate_metadata(
    df: pd.DataFrame,
    stations: StationConfig,
    cfg: PipelineConfig,
    api: Optional[APIConfig] = None,
) -> Dict[str, Any]:
    """Build the ``dataset-metadata.json`` document following Open Data norms."""
    return {
        "dataset": "CWC River Water Level (Telemetry + Manual, Hourly)",
        "data_provider": "Central Water Commission (CWC) via National Water Data "
        "Portal (NWDP, nwdp.nwic.gov.in)",
        "description": (
            "Hourly river water level readings for key CWC gauge stations across "
            "major Indian river basins, sourced from the CWC Telemetry and Manual "
            "datasets on NWDP and categorised into hydrological alert statuses "
            "(NORMAL / WARNING / DANGER / HIGH_FLOOD_EXCEEDED)."
        ),
        "temporalCoverage": _temporal_bounds(df),
        "monitored_basins": stations.basins,
        "monitored_stations": _station_list(df, stations),
        "total_records": int(len(df)),
        "alert_status_distribution": _alert_distribution(df),
        "schema": {"columns": [c for c in COLUMN_ORDER if c in df.columns]},
        "sources": _source_list(api),
        "retention_days": cfg.retention_days,
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def write_metadata(
    df: pd.DataFrame,
    stations: StationConfig,
    cfg: PipelineConfig,
    api: Optional[APIConfig] = None,
) -> Dict[str, Any]:
    """Generate and persist metadata; returns the metadata dict."""
    metadata = generate_metadata(df, stations, cfg, api=api)
    atomic_write_json(metadata, cfg.metadata_path)
    return metadata


__all__ = [
    "atomic_write_csv",
    "atomic_write_json",
    "merge_with_existing",
    "write_dataset",
    "write_partitions",
    "generate_metadata",
    "write_metadata",
]