"""Extraction & Ingestion Engine for the CWC / India-WRIS pipeline.

Responsibilities
----------------
* **Incremental watermarking** - compute the latest ``timestamp`` already
  stored in the target CSV and fetch only records strictly newer than it, so
  re-runs never re-fetch history (idempotency at the source).
* **Resilient live ingestion** - a :class:`requests.Session` wrapped in
  ``urllib3.util.Retry`` with exponential backoff, plus offset-based
  pagination against the Data.gov.in API.
* **Offline mock stream** - a deterministic synthetic generator producing an
  hourly telemetry stream (diurnal sinusoidal flow + stochastic flood surges +
  occasional sensor dropouts) when ``use_mock=True``.

Edge cases handled
------------------
* Empty / missing target CSV -> falls back to ``default_start_date``.
* Unparseable or nested API payloads -> tolerated via key aliasing and a
  one-level flatten; malformed records are surfaced to the transform layer.
* Sensor dropouts -> mock generator emits ``NaN`` water levels which the
  transform layer preserves (never fabricates values).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import APIConfig, SourceResource, StationConfig

logger = logging.getLogger(__name__)

# Standardised field names used across the whole pipeline.
STATION_CODE_COL = "station_code"
STATION_NAME_COL = "station_name"
TIMESTAMP_COL = "timestamp"
WATER_LEVEL_COL = "water_level"
LATITUDE_COL = "latitude"
LONGITUDE_COL = "longitude"

# Raw NWDP / CWC field spellings used by the extractor and mock generator.
STATION_RAW_COL = "Station"
TIME_RAW_COL = "Data Acquisition Time"
TELEMETRY_LEVEL_COL = "River Water Level Telemetry Hourly (meter)"
MANUAL_LEVEL_COL = "River Water Level Manual Hourly (meter)"

# Aliases for raw CWC / India-WRIS field spellings -> standard names.
# Keys are matched case-insensitively after stripping spaces/underscores.
_FIELD_ALIASES: Dict[str, List[str]] = {
    STATION_CODE_COL: [
        "station_code",
        "code_station",
        "stationcode",
        "station code",
        "stationid",
        "station id",
        "sitecode",
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
        "time",
        "date",
        "date_time",
        TIME_RAW_COL,
        "dataacquisitiontime",
    ],
    WATER_LEVEL_COL: [
        "water_level",
        "waterlevel",
        "water level",
        "wl",
        "level",
        "resultat_obs_elab",
        "gauge reading",
        "gaugereading",
        TELEMETRY_LEVEL_COL,
        MANUAL_LEVEL_COL,
    ],
    LATITUDE_COL: ["latitude", "lat"],
    LONGITUDE_COL: ["longitude", "lon", "long"],
}


# ---------------------------------------------------------------------------
# HTTP session with retry / backoff
# ---------------------------------------------------------------------------


def build_http_session(cfg: APIConfig) -> requests.Session:
    """Build a ``requests.Session`` with exponential-backoff retry policy.

    ``urllib3.util.Retry`` retries connection errors plus the statuses in
    ``cfg.retry_status_forcelist``. ``backoff_factor=1`` yields sleeps of
    1s, 2s, 4s, 8s... between attempts (jitter not applied, per urllib3).
    """
    retry = Retry(
        total=cfg.retry_total,
        connect=cfg.retry_total,
        read=cfg.retry_total,
        status=cfg.retry_total,
        backoff_factor=cfg.retry_backoff_factor,
        status_forcelist=tuple(cfg.retry_status_forcelist),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Live API ingestion
# ---------------------------------------------------------------------------


def _normalise_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one API record to standard field names.

    Nested payloads (e.g. India-WRIS wrapping fields under a ``station`` or
    ``observation`` key) are merged one level deep before alias matching.
    """
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            flat.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            flat[key] = value

    out: Dict[str, Any] = {}
    for standard, aliases in _FIELD_ALIASES.items():
        for raw_key, value in flat.items():
            if raw_key.lower().strip() in {a.lower() for a in aliases}:
                out[standard] = value
                break
    # Preserve any non-standard columns for downstream enrichment.
    for key, value in flat.items():
        out.setdefault(key, value)
    return out


def _payload_records(payload: Any) -> List[Dict[str, Any]]:
    """Extract the list of records from a Data.gov.in style response."""
    if isinstance(payload, dict):
        for key in ("records", "data", "results", "observations"):
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _to_utc_ts(value: Any, source_timezone: str = "UTC") -> Optional[pd.Timestamp]:
    """Coerce a single raw timestamp value to a tz-aware UTC Timestamp.

    Naive values (CWC publishes ``DD-MM-YYYY HH:MM`` IST) are localised to
    ``source_timezone`` before converting to UTC.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parsed = pd.to_datetime(value, errors="coerce", format="%d-%m-%Y %H:%M")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    tz = ZoneInfo(source_timezone)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(tz, nonexistent="shift_forward")
    else:
        parsed = parsed.tz_convert(tz)
    return parsed.tz_convert("UTC")


def _resource_in_window(
    resource: SourceResource, since: pd.Timestamp, end: pd.Timestamp
) -> bool:
    """True when a resource's year range can contain records in (since, end]."""
    if since > pd.Timestamp(f"{resource.end_year}-12-31T23:59:59", tz="UTC"):
        return False
    if end < pd.Timestamp(f"{resource.start_year}-01-01", tz="UTC"):
        return False
    return True


def _resource_station_names(
    stations: StationConfig, resource: SourceResource
) -> List[str]:
    """Configured source station names belonging to this resource's basin."""
    wanted_basin = resource.basin.lower()
    names = {
        s.source_station_name
        for s in stations.stations.values()
        if s.basin_name.lower() == wanted_basin
    }
    return sorted(names)


def fetch_live_data(
    cfg: APIConfig,
    stations: StationConfig,
    since: pd.Timestamp,
    end: Optional[pd.Timestamp] = None,
    basins: Optional[List[str]] = None,
    time_ranges: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Ingest records from the NWDP CKAN ``datastore_search`` API.

    Iterates the configured basin x time-range resources (:meth:`APIConfig
    .resolve_resources`). Each resource is queried **once per configured
    station** with an exact-match ``filters={"Station": name}`` so the server
    only returns the monitored stations (a Godavari 2021-25 backfill drops from
    ~1.06M rows to ~30K per station), and paginated with ``limit``/``offset``
    until ``total`` is reached. Resources whose year range cannot contain
    records newer than ``since`` are skipped.

    Records are additionally filtered client-side to the configured station
    set and the ``(since, end]`` timestamp window (idempotency safety net).

    Raises
    ------
    RuntimeError
        If the response cannot be fetched or decoded after retries.
    """
    end = end or pd.Timestamp.now(tz="UTC")
    session = build_http_session(cfg)
    resources = cfg.resolve_resources(basins=basins, time_ranges=time_ranges)
    if not resources:
        raise RuntimeError(
            f"no source resources matched basins={basins} time_ranges={time_ranges}"
        )
    wanted = stations.raw_name_keys()
    records: List[Dict[str, Any]] = []
    base_params: Dict[str, Any] = {"limit": cfg.max_per_page}
    if cfg.api_key:
        base_params["api-key"] = cfg.api_key

    for resource in resources:
        if not _resource_in_window(resource, since, end):
            logger.info(
                "Skipping resource %s %s (outside extraction window %s)",
                resource.basin, resource.time_range, since.isoformat(),
            )
            continue
        query_names = _resource_station_names(stations, resource)
        if not query_names:
            query_names = [None]  # fallback: unfiltered, client-side filtering only

        for station_name in query_names:
            logger.info(
                "Fetching resource %s (%s %s) since %s station=%s",
                resource.resource_id, resource.basin, resource.time_range,
                since.isoformat(), station_name or "ALL",
            )
            offset = 0
            while True:
                params = dict(
                    base_params, resource_id=resource.resource_id, offset=offset
                )
                if station_name is not None:
                    params["filters"] = json.dumps({"Station": station_name})
                try:
                    resp = session.get(
                        cfg.base_url, params=params, timeout=cfg.timeout_seconds
                    )
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    raise RuntimeError(
                        f"live fetch failed at offset={offset} for "
                        f"{resource.resource_id} station={station_name}: {exc}"
                    ) from exc

                payload = resp.json()
                result = (
                    payload.get("result", payload)
                    if isinstance(payload, dict)
                    else payload
                )
                batch = _payload_records(result)
                if not batch:
                    logger.info(
                        "Resource %s exhausted at offset=%d; total rows=%s.",
                        resource.resource_id, offset,
                        result.get("total") if isinstance(result, dict) else "?",
                    )
                    break

                for raw in batch:
                    norm = _normalise_record(raw)
                    raw_name = str(norm.get(STATION_NAME_COL, "")).strip()
                    if raw_name and wanted and raw_name.lower().strip() not in wanted:
                        continue
                    ts = _to_utc_ts(norm.get(TIMESTAMP_COL), cfg.source_timezone)
                    if ts is None:
                        records.append(norm)  # keep; transform layer flags the NaT
                        continue
                    if since < ts <= end:
                        records.append(norm)

                total = result.get("total") if isinstance(result, dict) else None
                offset += len(batch)
                if len(batch) < cfg.max_per_page or (
                    total is not None and offset >= int(total)
                ):
                    break

    if not records:
        logger.info("No new live records after watermark %s.", since.isoformat())
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Mock synthetic telemetry stream
# ---------------------------------------------------------------------------

_MOCK_DAY_SURGE_PROBABILITY = 0.25   # fraction of days that carry a surge event
_MOCK_SURGE_HOURS_RANGE = (2, 8)     # surge duration in hours
_MOCK_DROPOUT_PROBABILITY = 0.02     # per-hour chance of a missing gauge reading


def _seed_for(*parts: object) -> int:
    """Stable, process-independent RNG seed derived from arbitrary parts.

    Uses SHA-256 rather than the builtin ``hash()`` so values are identical
    across processes (``hash()`` is salted per interpreter), guaranteeing the
    mock stream is reproducible no matter which window is being generated.
    """
    digest = hashlib.sha256(
        "|".join(str(p) for p in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def generate_mock_stream(
    stations: StationConfig,
    since: pd.Timestamp,
    end: Optional[pd.Timestamp] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic hourly synthetic telemetry stream.

    Model per station, per hour::

        level = base + diurnal_sinusoid + seasonal_drift + surge + noise

    * ``base`` sits a few metres below the warning level (typical monsoon
      pre-flood conditions), so flood surges occasionally push readings into
      ``WARNING`` / ``DANGER`` territory.
    * Surge events are drawn per calendar day (deterministic per day) and
      decay exponentially over 2-8 hours; diurnal, seasonal and noise terms
      are deterministic functions of the absolute timestamp.
    * Sensor dropouts emit ``NaN`` water levels to exercise the missing-gauge
      edge case in the transform layer.

    The emitted records mirror the **raw NWDP schema** - ``Station`` (source
    name), ``Data Acquisition Time`` (naive IST ``DD-MM-YYYY HH:MM``) and the
    telemetry / manual level column - so mock and live runs share the exact
    same transform path.

    Determinism across incremental runs
    -----------------------------------
    Every per-hour quantity is derived from ``SHA256(seed, station, ts)`` and
    per-day surge events from ``SHA256(seed, station, day)``, so re-fetching a
    single hour later reproduces byte-identical readings and the composite-key
    dedup collapses repeats - guaranteed idempotency for incremental loads.
    """
    end = end or pd.Timestamp.now(tz="UTC")
    ist = ZoneInfo("Asia/Kolkata")
    rows: List[Dict[str, Any]] = []
    hour_index = pd.date_range(
        since.ceil("h"), end.ceil("h"), freq="h", tz="UTC"
    )

    for station in stations.stations.values():
        profile_rng = np.random.default_rng(
            _seed_for(seed, station.station_code, "profile")
        )
        base = station.warning_level - 2.2
        phase = float(profile_rng.uniform(0.0, 2 * np.pi))
        amplitude = float(profile_rng.uniform(0.10, 0.30))
        seasonal_amp = float(profile_rng.uniform(0.20, 0.45))
        lat_jitter = float(profile_rng.normal(0.0, 0.0005))
        lon_jitter = float(profile_rng.normal(0.0, 0.0005))
        level_col = (
            MANUAL_LEVEL_COL if station.kind == "manual" else TELEMETRY_LEVEL_COL
        )

        # Per-day surge events keep the stream continuous within a day while
        # remaining reproducible regardless of the extraction window.
        for day_midnight in pd.unique(hour_index.normalize()):
            day_hours = hour_index[hour_index.normalize() == day_midnight]
            day = day_midnight.date()
            day_rng = np.random.default_rng(
                _seed_for(seed, station.station_code, "day", day)
            )
            surge_magnitude = 0.0
            surge_start = 0
            surge_duration = 0
            if day_rng.random() < _MOCK_DAY_SURGE_PROBABILITY:
                surge_start = int(day_rng.integers(0, 24))
                surge_duration = int(day_rng.integers(*_MOCK_SURGE_HOURS_RANGE))
                surge_magnitude = float(day_rng.uniform(0.8, 4.0))

            for ts in day_hours:
                hour_rng = np.random.default_rng(
                    _seed_for(seed, station.station_code, "hour", ts)
                )
                diurnal = amplitude * np.sin(
                    2 * np.pi * (ts.hour + ts.minute / 60.0) / 24.0 + phase
                )
                seasonal = seasonal_amp * np.sin(
                    2 * np.pi * ts.dayofyear / 365.0 - 1.5
                )
                elapsed = ts.hour - surge_start
                surge = (
                    surge_magnitude * (0.9 ** elapsed)
                    if 0 <= elapsed < surge_duration
                    else 0.0
                )
                noise = float(hour_rng.uniform(-0.05, 0.05))
                water_level = base + diurnal + seasonal + surge + noise
                if hour_rng.random() < _MOCK_DROPOUT_PROBABILITY:
                    water_level = np.nan

                rows.append(
                    {
                        STATION_RAW_COL: station.source_station_name,
                        TIME_RAW_COL: ts.tz_convert(ist).strftime("%d-%m-%Y %H:%M"),
                        level_col: float(water_level),
                        "Latitude": round(station.latitude + lat_jitter, 5),
                        "Longitude": round(station.longitude + lon_jitter, 5),
                    }
                )

    df = pd.DataFrame(rows)
    logger.info(
        "Generated %d mock rows for %d station(s) over %s -> %s",
        len(df),
        len(stations.stations),
        hour_index.min(),
        hour_index.max(),
    )
    return df


# ---------------------------------------------------------------------------
# Incremental watermark
# ---------------------------------------------------------------------------


def determine_watermark(
    csv_path: Optional[Path], default_start: str = "2024-01-01"
) -> pd.Timestamp:
    """Return the newest ``timestamp`` already persisted in the target CSV.

    Used as the incremental extraction watermark: only records strictly newer
    than this value are fetched. Missing or empty files fall back to
    ``default_start`` (full backfill).
    """
    fallback = pd.Timestamp(default_start, tz="UTC")
    if csv_path is None or not Path(csv_path).exists():
        logger.info("No existing CSV; watermark defaults to %s", fallback.isoformat())
        return fallback

    try:
        existing = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        logger.warning("Could not read %s; defaulting watermark", csv_path)
        return fallback

    if existing.empty or TIMESTAMP_COL not in existing.columns:
        logger.info("CSV empty / missing timestamp column; using default watermark")
        return fallback

    parsed = pd.to_datetime(existing[TIMESTAMP_COL], errors="coerce", utc=True)
    max_ts = parsed.max()
    if pd.isna(max_ts):
        return fallback
    logger.info("Extraction watermark from %s: %s", csv_path, max_ts.isoformat())
    return max_ts


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def extract(
    cfg: APIConfig,
    stations: StationConfig,
    since: pd.Timestamp,
    end: Optional[pd.Timestamp] = None,
    seed: int = 42,
    basins: Optional[List[str]] = None,
    time_ranges: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Top-level extraction entry point.

    Selects the mock stream or live API based on ``cfg.use_mock`` and always
    returns raw, un-transformed records newer than ``since``. ``basins`` /
    ``time_ranges`` restrict which NWDP resources are fetched in live mode.
    """
    if cfg.use_mock:
        return generate_mock_stream(stations, since=since, end=end, seed=seed)
    return fetch_live_data(
        cfg, stations, since=since, end=end, basins=basins, time_ranges=time_ranges
    )


__all__ = [
    "build_http_session",
    "fetch_live_data",
    "generate_mock_stream",
    "determine_watermark",
    "extract",
]