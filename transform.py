"""Transformation & Enrichment Engine.

Pure functions only - no I/O, no random state, no side effects. Every
function takes a DataFrame (and optional station catalogue) and returns a new
DataFrame, which makes the layer trivially unit-testable and re-orderable.

Pipeline order
--------------
1. :func:`normalize_columns`  - map provider field aliases to standard names.
2. :func:`parse_timestamps`   - ISO / custom datetime strings -> UTC timestamps.
3. :func:`coerce_float64`     - water level and coordinates to ``np.float64``.
4. :func:`join_station_metadata` - attach WL / DL / HFL thresholds & basin info.
5. :func:`derive_flood_status` - hydrological alert categorisation.
6. :func:`create_dedup_key`   - idempotency hash ``{station}_{ts}_{level}``.
7. :func:`order_columns`      - standardised schema / column ordering.

Edge cases
----------
* Missing gauge readings (``NaN`` water level) -> ``flood_status`` stays
  ``NaN`` rather than being mis-labelled ``NORMAL``; the validation layer
  reports the null percentage so dropouts are visible.
* Unparseable timestamps -> rows are dropped and a warning is logged.
* Unknown station codes -> thresholds become ``NaN`` (no categorisation).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import StationConfig

logger = logging.getLogger(__name__)

STATION_CODE_COL = "station_code"
STATION_NAME_COL = "station_name"
BASIN_NAME_COL = "basin_name"
TIMESTAMP_COL = "timestamp"
WATER_LEVEL_COL = "water_level"
FLOOD_STATUS_COL = "flood_status"
WARNING_LEVEL_COL = "warning_level"
DANGER_LEVEL_COL = "danger_level"
HFL_COL = "high_flood_level"
LATITUDE_COL = "latitude"
LONGITUDE_COL = "longitude"
DEDUP_KEY_COL = "dedup_key"

# Standardised output schema / column ordering.
COLUMN_ORDER: List[str] = [
    STATION_CODE_COL,
    STATION_NAME_COL,
    BASIN_NAME_COL,
    TIMESTAMP_COL,
    WATER_LEVEL_COL,
    FLOOD_STATUS_COL,
    WARNING_LEVEL_COL,
    DANGER_LEVEL_COL,
    HFL_COL,
    LATITUDE_COL,
    LONGITUDE_COL,
    DEDUP_KEY_COL,
]

# Hydrological alert categories, in ascending severity order.
FLOOD_STATUS_NORMAL = "NORMAL"
FLOOD_STATUS_WARNING = "WARNING"
FLOOD_STATUS_DANGER = "DANGER"
FLOOD_STATUS_EXCEEDED = "HIGH_FLOOD_EXCEEDED"
FLOOD_STATUS_CHOICES: List[str] = [
    FLOOD_STATUS_NORMAL,
    FLOOD_STATUS_WARNING,
    FLOOD_STATUS_DANGER,
    FLOOD_STATUS_EXCEEDED,
]

# Provider field aliases -> standard names (lower-cased key match).
_FIELD_ALIASES: Dict[str, List[str]] = {
    STATION_CODE_COL: [
        "station_code",
        "code_station",
        "stationcode",
        "station code",
        "stationid",
        "site code",
    ],
    STATION_NAME_COL: [
        "station",
        "station_name",
        "station name",
        "name",
    ],
    TIMESTAMP_COL: [
        "timestamp",
        "datetime",
        "obs_time",
        "observed at",
        "observedat",
        "date_time",
        "time",
        "date",
        "data acquisition time",
        "dataacquisitiontime",
    ],
    WATER_LEVEL_COL: [
        "water_level",
        "waterlevel",
        "water level",
        "wl",
        "level",
        "resultat_obs_elab",
        "river water level telemetry hourly (meter)",
        "river water level manual hourly (meter)",
    ],
    LATITUDE_COL: ["latitude", "lat"],
    LONGITUDE_COL: ["longitude", "lon", "long"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename provider-specific field spellings to the standard schema.

    Multiple source columns mapping to the same standard name (e.g. the CWC
    Telemetry and Manual water level columns) are coalesced into one standard
    column, first non-null wins.
    """
    if df.empty:
        return df.copy()
    rename: Dict[str, str] = {}
    for standard, aliases in _FIELD_ALIASES.items():
        alias_set = {a.lower().strip() for a in aliases}
        matches = [
            c for c in df.columns
            if c not in rename and c.lower().strip() in alias_set
        ]
        if not matches:
            continue
        rename[matches[0]] = standard
        for extra in matches[1:]:
            rename[extra] = f"{standard}__merge"
    if not rename:
        return df.copy()

    out = df.rename(columns=rename)
    merge_cols = [c for c in out.columns if c.endswith("__merge")]
    for col in merge_cols:
        base = col[: -len("__merge")]
        out[base] = out[base].combine_first(out[col])
        out = out.drop(columns=[col])
    return out


def parse_timestamps(
    df: pd.DataFrame, source_timezone: str = "UTC"
) -> pd.Series:
    """Parse ISO / custom datetime strings into tz-aware UTC timestamps.

    Naive (timezone-less) values are localised to ``source_timezone`` (CWC
    publishes ``DD-MM-YYYY HH:MM`` IST, so this should be ``Asia/Kolkata``)
    before converting to UTC. Timestamps that cannot be parsed become ``NaT``
    and the caller is expected to drop them (see :func:`transform`).
    """
    if df.empty or TIMESTAMP_COL not in df.columns:
        return pd.Series(index=df.index, dtype="datetime64[ns, UTC]")
    raw = df[TIMESTAMP_COL]
    parsed = pd.to_datetime(raw, errors="coerce", format="%d-%m-%Y %H:%M")
    if parsed.isna().any():
        fallback = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        parsed = parsed.fillna(fallback)
    tz = ZoneInfo(source_timezone)
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    else:
        parsed = parsed.dt.tz_convert(tz)
    return parsed.dt.tz_convert("UTC")


def coerce_float64(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce the given columns to ``np.float64``, turning bad values to NaN."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            out[col] = np.nan
    return out


def _clean(raw: object) -> str:
    """Case- and whitespace-insensitive name normalisation for matching."""
    return " ".join(str(raw).strip().lower().split())


def join_station_metadata(
    df: pd.DataFrame, stations: StationConfig
) -> pd.DataFrame:
    """Attach static station metadata (code, basin, thresholds, coordinates).

    Incoming CWC records carry a station *name* (``station_name``), so each
    name is resolved against the catalogue via :meth:`StationConfig
    .resolve_station`. Resolved rows get the station code, display name,
    basin and thresholds; unresolvable names keep their raw name as the code
    (so grouping / partitions stay stable) but get ``NaN`` thresholds and can
    never be mis-categorised.
    """
    if df.empty or STATION_NAME_COL not in df.columns:
        return df.copy()
    out = df.copy()

    cleaned = out[STATION_NAME_COL].map(_clean)
    profile = cleaned.map(stations.resolve_station)

    code = profile.map(lambda s: s.station_code if s is not None else np.nan)
    code = code.fillna(out[STATION_NAME_COL]).fillna("UNKNOWN")
    out[STATION_CODE_COL] = code.astype(str)

    resolved_name = profile.map(lambda s: s.station_name if s is not None else np.nan)
    out[STATION_NAME_COL] = resolved_name.fillna(out[STATION_NAME_COL])

    out[BASIN_NAME_COL] = profile.map(
        lambda s: s.basin_name if s is not None else np.nan
    )
    for col, attr in (
        (WARNING_LEVEL_COL, "warning_level"),
        (DANGER_LEVEL_COL, "danger_level"),
        (HFL_COL, "high_flood_level"),
    ):
        out[col] = profile.map(lambda s, a=attr: getattr(s, a) if s is not None else np.nan)

    for col, attr in ((LATITUDE_COL, "latitude"), (LONGITUDE_COL, "longitude")):
        catalog = profile.map(
            lambda s, a=attr: getattr(s, a) if s is not None else np.nan
        )
        # Prefer raw source coordinates when present, else catalogue values.
        out[col] = out[col].fillna(catalog)

    unresolved = code[profile.isna()].unique()
    if len(unresolved):
        logger.warning(
            "Station(s) without a configured profile (thresholds=NaN): %s",
            ", ".join(sorted(str(u) for u in unresolved)),
        )
    return out


def derive_flood_status(df: pd.DataFrame) -> pd.Series:
    """Derive the hydrological alert category from water level vs thresholds.

    ``NORMAL``:             level < Warning Level
    ``WARNING``:            WL <= level < Danger Level
    ``DANGER``:             DL <= level < High Flood Level
    ``HIGH_FLOOD_EXCEEDED``: level >= High Flood Level

    Missing levels or missing thresholds yield ``NaN`` (never ``NORMAL``).
    """
    if df.empty or WATER_LEVEL_COL not in df.columns:
        return pd.Series(index=df.index, dtype="object")

    wl = df[WATER_LEVEL_COL].astype("float64")
    warning = df.get(WARNING_LEVEL_COL, pd.Series(np.nan, index=df.index)).astype("float64")
    danger = df.get(DANGER_LEVEL_COL, pd.Series(np.nan, index=df.index)).astype("float64")
    hfl = df.get(HFL_COL, pd.Series(np.nan, index=df.index)).astype("float64")

    conditions = [
        wl < warning,
        (wl >= warning) & (wl < danger),
        (wl >= danger) & (wl < hfl),
        wl >= hfl,
    ]
    return pd.Series(
        np.select(conditions, FLOOD_STATUS_CHOICES, default=np.nan),
        index=df.index,
        name=FLOOD_STATUS_COL,
    )


def create_dedup_key(df: pd.DataFrame) -> pd.Series:
    """Build idempotency keys ``{station_code}_{timestamp}_{water_level}``.

    The key is lossless for water-level readings, so a repeated run that
    re-fetches identical observations collapses them into a single row.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype="object")
    station = df[STATION_CODE_COL].astype(str)
    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
    ts_str = ts.dt.strftime("%Y-%m-%dT%H:%M:%S%z").fillna("")
    level = df.get(WATER_LEVEL_COL, pd.Series(np.nan, index=df.index)).astype("float64")
    level_str = level.map(
        lambda v: "" if pd.isna(v) else format(v, ".4f")
    ).fillna("")
    return (station + "_" + ts_str + "_" + level_str).rename(DEDUP_KEY_COL)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the standardised column ordering; keep unknown extras last."""
    if df.empty:
        return df.copy()
    present = [c for c in COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in present]
    return df[present + extras]


def transform(
    raw_df: pd.DataFrame,
    stations: Optional[StationConfig] = None,
    source_timezone: str = "UTC",
) -> pd.DataFrame:
    """Full transformation & enrichment pipeline (pure function).

    Accepts either standardised or raw provider columns and returns a clean,
    typed, deduplicated DataFrame ordered by :data:`COLUMN_ORDER`. Naive
    provider timestamps are localised to ``source_timezone`` (e.g.
    ``Asia/Kolkata``) then converted to UTC.
    """
    if raw_df is None or raw_df.empty:
        logger.info("No records to transform; returning empty DataFrame.")
        return pd.DataFrame(columns=COLUMN_ORDER)

    df = normalize_columns(raw_df)
    df[TIMESTAMP_COL] = parse_timestamps(df, source_timezone=source_timezone)

    n_before = len(df)
    df = df.dropna(subset=[TIMESTAMP_COL]).copy()
    if len(df) < n_before:
        logger.warning(
            "Dropped %d row(s) with unparseable timestamps.", n_before - len(df)
        )

    df = coerce_float64(
        df, [WATER_LEVEL_COL, LATITUDE_COL, LONGITUDE_COL]
    )

    if stations is not None:
        df = join_station_metadata(df, stations)

    df[FLOOD_STATUS_COL] = derive_flood_status(df)
    df[DEDUP_KEY_COL] = create_dedup_key(df)
    df = order_columns(df)
    df = df.sort_values([STATION_CODE_COL, TIMESTAMP_COL]).reset_index(drop=True)
    logger.info("Transform produced %d clean records.", len(df))
    return df


__all__ = [
    "COLUMN_ORDER",
    "FLOOD_STATUS_CHOICES",
    "normalize_columns",
    "parse_timestamps",
    "coerce_float64",
    "join_station_metadata",
    "derive_flood_status",
    "create_dedup_key",
    "order_columns",
    "transform",
]