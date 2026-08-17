"""Orchestration Entry Point for the CWC / NWDP river water level ETL pipeline.

Ties the layers together in a strict, idempotent sequence::

    extract -> transform -> load -> validate

Run modes
---------
* ``python main.py``                - default (mock if ``CWC_USE_MOCK`` unset).
* ``python main.py --live``         - hit the real NWDP CKAN API (public;
  ``CWC_API_KEY`` is sent only when set).
* ``python main.py --backfill --start-date 2021-01-01``
                                    - ignore the existing watermark and
  backfill history across the configured basin resources.
* ``python main.py --live --basins godavari,yamuna --time-ranges "(2026 - 2030)"``
                                    - restrict which NWDP resources are fetched.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import APIConfig, PipelineConfig, StationConfig, build_configs
from extract import determine_watermark, extract
from load import merge_with_existing, write_dataset, write_metadata
from transform import STATION_CODE_COL, transform
from validation import validate_dataset

logger = logging.getLogger("etl")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_etl(
    use_mock: Optional[bool] = None,
    start_date: Optional[str] = None,
    seed: int = 42,
    backfill: bool = False,
    basins: Optional[List[str]] = None,
    time_ranges: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the full Extract -> Transform -> Load -> Validate pipeline.

    Parameters
    ----------
    use_mock:
        Force mock or live mode; ``None`` defers to env/config.
    start_date:
        ISO date used when no existing watermark is available.
    seed:
        Determinism seed for the mock generator.
    backfill:
        If ``True``, ignore the existing CSV watermark and start from
        ``start_date`` (or ``APIConfig.default_start_date``).
    basins:
        Optional basin filter (substrings, e.g. ``["godavari", "yamuna"]``).
    time_ranges:
        Optional resource time-range filter (e.g. ``["(2026 - 2030)"]``).

    Returns the final published DataFrame.
    """
    api: APIConfig
    stations: StationConfig
    pipeline: PipelineConfig
    api, stations, pipeline = build_configs(use_mock=use_mock)

    _configure_logging(pipeline.logging_level)
    mode = "MOCK" if api.use_mock else "LIVE"
    logger.info(
        "=== Starting ETL run (mode=%s, stations=%d, basins=%s, ranges=%s) ===",
        mode,
        len(stations.stations),
        basins or "all",
        time_ranges or "all",
    )

    default_start = start_date or api.default_start_date
    watermark = (
        pd.Timestamp(default_start, tz="UTC")
        if backfill
        else determine_watermark(pipeline.csv_path, default_start=default_start)
    )
    logger.info("Extraction watermark: %s", watermark.isoformat())

    raw = extract(
        api,
        stations,
        since=watermark,
        seed=seed,
        basins=basins,
        time_ranges=time_ranges,
    )
    logger.info("Extract stage returned %d raw record(s).", len(raw))

    clean = transform(raw, stations, source_timezone=api.source_timezone)

    merged = merge_with_existing(
        clean,
        csv_path=pipeline.csv_path,
        retention_days=pipeline.retention_days,
    )

    _warn_absent_stations(merged, stations)

    write_dataset(merged, pipeline)
    write_metadata(merged, stations, pipeline, api=api)
    validate_dataset(merged)

    logger.info("=== ETL run complete: %d records published ===", len(merged))
    return merged


def _warn_absent_stations(
    df: pd.DataFrame, stations: StationConfig
) -> None:
    """Log configured stations that produced no records in this run."""
    if df.empty or STATION_CODE_COL not in df.columns:
        missing = stations.station_codes
    else:
        present = set(df[STATION_CODE_COL].astype(str))
        missing = [c for c in stations.station_codes if c not in present]
    if missing:
        logger.warning(
            "Configured station(s) with no records in this run (may be newly "
            "added upstream or outside the fetched window): %s",
            ", ".join(missing),
        )


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CWC / NWDP hourly river water level ETL pipeline."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--use-mock",
        dest="use_mock",
        action="store_true",
        default=None,
        help="Generate synthetic data offline (no API key needed).",
    )
    mode.add_argument(
        "--live",
        dest="use_mock",
        action="store_false",
        help="Fetch from the NWDP CKAN API (public; CWC_API_KEY optional).",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Ignore the existing watermark and extract from the start date.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="ISO start date used when no watermark exists (default from config).",
    )
    parser.add_argument(
        "--basins",
        type=str,
        default=None,
        help="Comma-separated basin filter, e.g. godavari,krishna,yamuna.",
    )
    parser.add_argument(
        "--time-ranges",
        type=str,
        default=None,
        help='Comma-separated resource time-range filter, e.g. "(2026 - 2030)".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Determinism seed for the mock generator.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level or "INFO")
    basins = (
        [b.strip() for b in args.basins.split(",") if b.strip()]
        if args.basins
        else None
    )
    time_ranges = (
        [r.strip() for r in args.time_ranges.split(",") if r.strip()]
        if args.time_ranges
        else None
    )
    try:
        run_etl(
            use_mock=args.use_mock,
            start_date=args.start_date,
            seed=args.seed,
            backfill=args.backfill,
            basins=basins,
            time_ranges=time_ranges,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logger.error("ETL pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())