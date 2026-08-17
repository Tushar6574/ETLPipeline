"""Configuration Layer for the CWC / India-WRIS river water level ETL pipeline.

Typed ``dataclasses`` that centralise every tunable knob of the pipeline:

* :class:`APIConfig`        - connection, pagination, retry and mock-toggle settings.
* :class:`Station`          - a single telemetry station profile (thresholds + geo).
* :class:`StationConfig`    - the pre-configured station catalogue keyed by station code.
* :class:`PipelineConfig`   - output paths, retention and logging settings.

Design notes
------------
* Credentials are never hardcoded; the Data.gov.in API key is read from the
  ``CWC_API_KEY`` environment variable at runtime.
* ``use_mock`` toggles the whole pipeline into an offline, deterministic
  synthetic-data mode so local development / CI does not require an API key.
* All validation lives in ``__post_init__`` so misconfiguration fails fast,
  before any network or file activity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceResource:
    """One NWDP ``datastore_search`` resource: a (basin, kind, time range) slice.

    The CWC publishes river water level data on the National Water Data Portal
    (nwdp.nwic.gov.in) partitioned by basin x time range. Each ``resource_id``
    below is a CKAN datastore resource; ``start_year``/``end_year`` allow the
    extractor to skip resources whose range cannot contain new records.
    """

    basin: str
    kind: str  # "telemetry" | "manual"
    time_range: str  # human-readable label, e.g. "(2026 - 2030)"
    resource_id: str
    start_year: int
    end_year: int


# Registered NWDP resources for the monitored basins (verified 2026-08-17).
# Telemetry package: 68600163-c5a0-4327-aa1a-fa7157b86cce
# Manual package:    d951a09c-6cf8-470e-be77-e80116f13d34
SOURCE_RESOURCES: List[SourceResource] = [
    SourceResource("Godavari", "telemetry", "(2021 - 2025)",
                   "4d139325-8371-422f-acef-d34ccec4300e", 2021, 2025),
    SourceResource("Godavari", "telemetry", "(2026 - 2030)",
                   "c6f31452-b416-4599-a6ae-07ad4217cdf4", 2026, 2030),
    SourceResource("Krishna", "telemetry", "(2021 - 2025)",
                   "4ddede56-5e49-4956-bb6f-df3786bf601f", 2021, 2025),
    SourceResource("Krishna", "telemetry", "(2026 - 2030)",
                   "d80798b9-4b11-4626-8b63-964202ba7216", 2026, 2030),
    SourceResource("Cauvery", "telemetry", "(2021 - 2025)",
                   "b5776a31-0583-4c27-a2d9-ee81e3b3a03a", 2021, 2025),
    SourceResource("Cauvery", "telemetry", "(2026 - 2030)",
                   "d027c5ac-379d-4ac2-8ced-97b02b6edbc0", 2026, 2030),
    SourceResource("Narmada", "telemetry", "(2021 - 2025)",
                   "0b01ff69-83de-4274-8233-6e564a6eeb9a", 2021, 2025),
    SourceResource("Narmada", "telemetry", "(2026 - 2030)",
                   "a2e056f6-ed8a-45bf-9142-01dc1904405a", 2026, 2030),
    SourceResource("Yamuna Basin", "manual", "(2021 - 2025)",
                   "dd3b9bba-6ba2-4954-93cf-e3ee84b5d9af", 2021, 2025),
    SourceResource("Yamuna Basin", "manual", "(2026 - 2030)",
                   "8663d648-5057-4eaf-86fc-72d5e755f053", 2026, 2030),
]


@dataclass(frozen=True)
class APIConfig:
    """Settings for the CWC river water level API on the National Water Data
    Portal (NWDP, nwdp.nwic.gov.in - a CKAN portal).

    Attributes
    ----------
    base_url:
        CKAN ``datastore_search`` endpoint. ``resource_id`` is passed as a
        query parameter per request.
    api_key_env:
        Optional environment variable holding an API key. NWDP data is public,
        so this is only sent when the variable is set.
    time_ranges:
        Time ranges to ingest (labels must match ``SOURCE_RESOURCES``).
    source_timezone:
        IANA timezone of the raw ``Data Acquisition Time`` values. CWC timestamps
        are naive IST; the transform layer localises to this zone before
        converting to UTC.
    max_per_page:
        Pagination ``limit`` per CKAN request.
    timeout_seconds:
        Per-request socket timeout.
    retry_total:
        Maximum retries per request (exponential backoff).
    retry_backoff_factor:
        Backoff base factor; sleeps ``backoff * (2 ** attempt)`` seconds.
    retry_status_forcelist:
        HTTP statuses that trigger a retry.
    use_mock:
        If ``True`` the extractor uses the synthetic stream instead of the API.
    default_start_date:
        ISO date used when no existing data / watermark is available. Defaults
        to the 2026-2030 epoch so routine hourly runs stay fast; pass
        ``--backfill --start-date 2021-01-01`` to ingest history.
    """

    base_url: str = "https://nwdp.nwic.gov.in/api/3/action/datastore_search"
    api_key_env: str = "CWC_API_KEY"
    time_ranges: Tuple[str, ...] = ("(2021 - 2025)", "(2026 - 2030)")
    source_timezone: str = "Asia/Kolkata"
    max_per_page: int = 1000
    timeout_seconds: int = 30
    retry_total: int = 5
    retry_backoff_factor: float = 1.0
    retry_status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504)
    use_mock: bool = True
    default_start_date: str = "2026-01-01"

    def __post_init__(self) -> None:
        if not self.time_ranges:
            raise ValueError("time_ranges must not be empty")
        if not 1 <= self.max_per_page <= 50000:
            raise ValueError("max_per_page must be within [1, 50000]")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retry_total < 0:
            raise ValueError("retry_total must be non-negative")
        if self.retry_backoff_factor < 0:
            raise ValueError("retry_backoff_factor must be non-negative")

    @property
    def api_key(self) -> str:
        """API key read from the environment; empty string when unset."""
        return os.getenv(self.api_key_env, "")

    def resolve_resources(
        self,
        basins: Optional[List[str]] = None,
        time_ranges: Optional[List[str]] = None,
    ) -> List[SourceResource]:
        """Return the registered resources matching the basin / range filters.

        Filtering is case-insensitive and accepts substrings, so ``--basins
        godavari,krishna`` selects every Godavari / Krishna resource.
        """
        wanted_basins = {b.strip().lower() for b in basins} if basins else None
        wanted_ranges = {r.strip().lower() for r in time_ranges} if time_ranges else None
        out = []
        for res in SOURCE_RESOURCES:
            if wanted_basins and not any(
                w in res.basin.lower() for w in wanted_basins
            ):
                continue
            if wanted_ranges and res.time_range.lower() not in wanted_ranges:
                continue
            out.append(res)
        return out


# ---------------------------------------------------------------------------
# Station profiles
# ---------------------------------------------------------------------------


def _clean_name(value: object) -> str:
    """Normalise a raw station name for case/whitespace-insensitive matching."""
    return " ".join(str(value).strip().lower().split())


@dataclass(frozen=True)
class Station:
    """Profile for a single CWC river gauge station.

    Attributes
    ----------
    station_code:
        Internal stable identifier used as the output partition / dedup key
        (e.g. ``YMNDLH`` for Old Delhi Railway Bridge).
    station_name:
        Human readable station / river gauge name (display).
    source_station_name:
        Exact ``Station`` value as published by CWC / NWDP - the join key used
        to match incoming records (e.g. ``Jaikwadi Dam``, ``MUSIRI``).
    match_aliases:
        Optional alternate spellings accepted when matching source records.
    basin_name:
        Hydrological basin the station belongs to.
    kind:
        ``telemetry`` or ``manual``; selects the source level column name.
    warning_level:
        Warning level (WL) in metres above Mean Sea Level.
    danger_level:
        Danger level (DL) in metres. Must be ``>= warning_level``.
    high_flood_level:
        High flood level (HFL) in metres. Must be ``>= danger_level``.
    latitude:
        WGS84 latitude in decimal degrees.
    longitude:
        WGS84 longitude in decimal degrees.
    """

    station_code: str
    station_name: str
    source_station_name: str
    basin_name: str
    kind: str = "telemetry"
    warning_level: float = 0.0
    danger_level: float = 0.0
    high_flood_level: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    match_aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.station_code.strip():
            raise ValueError("station_code must not be empty")
        if not self.source_station_name.strip():
            raise ValueError("source_station_name must not be empty")
        if self.kind not in {"telemetry", "manual"}:
            raise ValueError(f"kind must be 'telemetry' or 'manual', got {self.kind!r}")
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"latitude out of range for {self.station_code}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"longitude out of range for {self.station_code}")
        # Monotonic flood thresholds guard against nonsense configurations
        # that would otherwise silently mis-categorise flood statuses.
        if not self.warning_level <= self.danger_level <= self.high_flood_level:
            raise ValueError(
                f"thresholds must satisfy WL <= DL <= HFL for {self.station_code}: "
                f"{self.warning_level} <= {self.danger_level} <= {self.high_flood_level}"
            )

    def name_matches(self, raw_name: object) -> bool:
        """True when ``raw_name`` matches any configured name for this station."""
        candidates = {self.source_station_name, self.station_name, *self.match_aliases}
        return _clean_name(raw_name) in {_clean_name(c) for c in candidates}


@dataclass
class StationConfig:
    """Catalogue of monitored stations, keyed by station code.

    Wraps a ``dict`` so lookups stay O(1) and the catalogue can be validated
    in one pass at construction time. Incoming CWC records carry a station
    *name* rather than a code, so :meth:`resolve_station` maps names onto the
    catalogue (case- and whitespace-insensitive).
    """

    stations: Dict[str, Station] = field(
        default_factory=lambda: {s.station_code: s for s in DEFAULT_STATIONS}
    )

    def __post_init__(self) -> None:
        if not self.stations:
            raise ValueError("station catalogue must not be empty")
        for code, station in self.stations.items():
            if code != station.station_code:
                raise ValueError(
                    f"dict key {code!r} does not match station_code "
                    f"{station.station_code!r}"
                )

    def get(self, station_code: str) -> Station:
        """Return the station profile, raising ``KeyError`` if unknown."""
        try:
            return self.stations[station_code]
        except KeyError:
            raise KeyError(
                f"no station profile configured for code {station_code!r}"
            ) from None

    def resolve_station(self, raw_name: object) -> Optional[Station]:
        """Map a raw CWC station name onto the catalogue (or ``None``)."""
        target = _clean_name(raw_name)
        for station in self.stations.values():
            if station.name_matches(target):
                return station
        return None

    def name_in_catalogue(self, raw_name: object) -> bool:
        return self.resolve_station(raw_name) is not None

    def raw_name_keys(self) -> set:
        """Every accepted spelling across the catalogue (cleaned, lowercased)."""
        keys: set = set()
        for station in self.stations.values():
            for candidate in (
                station.source_station_name,
                station.station_name,
                *station.match_aliases,
            ):
                keys.add(_clean_name(candidate))
        return keys

    @property
    def station_codes(self) -> List[str]:
        return sorted(self.stations.keys())

    @property
    def basins(self) -> List[str]:
        return sorted({s.basin_name for s in self.stations.values()})


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """File-system and housekeeping settings for the pipeline run.

    Attributes
    ----------
    output_dir:
        Root directory for generated artefacts.
    partition_dir:
        Directory holding per-station partition files.
    csv_filename / parquet_filename / metadata_filename:
        Artefact names inside ``output_dir``.
    logging_level:
        Standard Python logging level name.
    retention_days:
        Records older than this many days relative to the newest reading are
        dropped from the published dataset (rolling-window retention).
    write_parquet:
        Whether to additionally emit a Parquet copy of the dataset.
    """

    output_dir: Path = field(default_factory=lambda: Path("output"))
    csv_filename: str = "cwc_river_water_levels.csv"
    parquet_filename: str = "cwc_river_water_levels.parquet"
    metadata_filename: str = "dataset-metadata.json"
    logging_level: str = "INFO"
    retention_days: int = 365
    write_parquet: bool = True

    def __post_init__(self) -> None:
        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive")
        if self.logging_level.upper() not in logging._nameToLevel:
            raise ValueError(
                f"invalid logging level {self.logging_level!r}; "
                f"expected one of {sorted(logging._nameToLevel)}"
            )
        object.__setattr__(
            self, "partition_dir", self.output_dir / "partitions"
        )

    @property
    def csv_path(self) -> Path:
        return self.output_dir / self.csv_filename

    @property
    def parquet_path(self) -> Path:
        return self.output_dir / self.parquet_filename

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / self.metadata_filename

    def for_mode(self, use_mock: bool) -> "PipelineConfig":
        """Return a copy rooted at ``mock_output/`` when running offline.

        Keeps synthetic artefacts strictly separated from production data so
        a mock run can never overwrite the real published dataset.
        """
        if not use_mock:
            return self
        return _replace(self, output_dir=Path("mock_output"))


def _replace(cfg: PipelineConfig, **kwargs: object) -> PipelineConfig:
    """Reconstruct a frozen dataclass with overridden fields."""
    fields = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
    fields.update(kwargs)
    return PipelineConfig(**fields)


# ---------------------------------------------------------------------------
# Reference stations (source names confirmed against NWDP datastores 2026-08-17)
# ---------------------------------------------------------------------------
# ``source_station_name`` is the exact ``Station`` value published by CWC/NWDP
# and is the join key for incoming records.
#
# VERIFY: WL / DL / HFL thresholds are curated approximations of published CWC
# flood bulletin values; confirm against official CWC bulletins before relying
# on ``flood_status`` for operational decisions. Stations without live 2026
# readings yet (e.g. Jaikwadi Dam, Old Delhi Railway Bridge) are retained so a
# ``--backfill`` of the 2021-2025 resources captures their history; the pipeline
# logs a warning when a configured station yields no records in a run.

DEFAULT_STATIONS: List[Station] = [
    Station(
        station_code="YMNDLH",
        station_name="Yamuna at Old Delhi Railway Bridge",
        source_station_name="Old Delhi Railway Bridge",
        basin_name="Yamuna Basin",
        kind="manual",
        warning_level=203.00,
        danger_level=205.32,
        high_flood_level=207.49,
        latitude=28.6585,
        longitude=77.2347,
    ),
    Station(
        station_code="YMNCKT",
        station_name="Yamuna at Chitrakoot",
        source_station_name="Chitrakoot",
        basin_name="Yamuna Basin",
        kind="manual",
        warning_level=136.00,
        danger_level=137.00,
        high_flood_level=138.00,
        latitude=25.14222222,
        longitude=80.85472222,
    ),
    Station(
        station_code="PTHNGD",
        station_name="Godavari at Paithan (Jaikwadi Dam)",
        source_station_name="Jaikwadi Dam",
        basin_name="Godavari",
        kind="telemetry",
        warning_level=458.31,
        danger_level=458.96,
        high_flood_level=460.56,
        latitude=19.4720,
        longitude=75.3500,
    ),
    Station(
        station_code="GDBDCL",
        station_name="Godavari at Bhadrachalam",
        source_station_name="Bhadrachalam",
        basin_name="Godavari",
        kind="telemetry",
        warning_level=47.50,
        danger_level=48.50,
        high_flood_level=53.00,
        latitude=17.66944444,
        longitude=80.87388889,
    ),
    Station(
        station_code="KRSNWDP",
        station_name="Krishna at Wadenepally",
        source_station_name="Wadenepally",
        basin_name="Krishna",
        kind="telemetry",
        warning_level=18.45,
        danger_level=19.84,
        high_flood_level=23.07,
        latitude=16.79416667,
        longitude=80.07305556,
    ),
    Station(
        station_code="CAVMSI",
        station_name="Cauvery at Musiri",
        source_station_name="MUSIRI",
        basin_name="Cauvery",
        kind="telemetry",
        warning_level=84.51,
        danger_level=85.40,
        high_flood_level=89.25,
        latitude=10.93805556,
        longitude=78.44027778,
    ),
    Station(
        station_code="NRMMDL",
        station_name="Narmada at Mandla",
        source_station_name="Mandla",
        basin_name="Narmada",
        kind="telemetry",
        warning_level=439.00,
        danger_level=441.00,
        high_flood_level=444.00,
        latitude=22.59833333,
        longitude=80.36527778,
    ),
]


def build_configs(
    use_mock: bool | None = None,
) -> Tuple[APIConfig, StationConfig, PipelineConfig]:
    """Convenience factory returning the three top-level config objects.

    ``use_mock`` is resolved from ``APIConfig.use_mock`` when left ``None``,
    allowing environment-driven toggling via ``CWC_USE_MOCK``.
    """
    api = APIConfig(use_mock=_resolve_use_mock(use_mock))
    stations = StationConfig()
    pipeline = PipelineConfig().for_mode(api.use_mock)
    return api, stations, pipeline


def _resolve_use_mock(use_mock: bool | None) -> bool:
    if use_mock is not None:
        return use_mock
    env = os.getenv("CWC_USE_MOCK")
    if env is None:
        return True
    return env.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "APIConfig",
    "SourceResource",
    "Station",
    "StationConfig",
    "PipelineConfig",
    "DEFAULT_STATIONS",
    "SOURCE_RESOURCES",
    "build_configs",
]