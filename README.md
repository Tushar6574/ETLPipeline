# CWC River Water Level ETL Pipeline (Telemetry + Manual)

An idempotent, open-data ETL pipeline that ingests **hourly river water level
readings** published by the Central Water Commission (CWC) on India's
**National Water Data Portal (NWDP)**, transforms them into a clean, typed
dataset enriched with hydrological flood alerts (`NORMAL` / `WARNING` /
`DANGER` / `HIGH_FLOOD_EXCEEDED`), and publishes the result as CSV + Parquet +
per-station partitions every hour via GitHub Actions.

```
  NWDP (CWC)                        extract            transform          load                validate
  datastore_search  ──────────►  paginated fetch   normalise / parse   merge + retention   quality report
  (basin × range)                  + watermark      thresholds / status  CSV/Parquet/partitions
```

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Data source](#data-source)
- [Monitored stations](#monitored-stations)
- [Dataset schema](#dataset-schema)
- [Flood alert categorisation](#flood-alert-categorisation)
- [Quickstart](#quickstart)
- [Dashboard UI](#dashboard-ui)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [GitHub Actions](#github-actions)
- [Operational history](#operational-history)
- [Data quality notes](#data-quality-notes)
- [Project layout](#project-layout)
- [Disclaimer](#disclaimer)

---

## Features

- **Real CWC/NWDP data** - live hourly telemetry + manual gauge readings for
  five Indian basins, ingested from the public NWDP CKAN `datastore_search` API
  (no API key required; `CWC_API_KEY` is sent only when set).
- **Deterministic mock mode** - fully offline, reproducible synthetic stream
  that exercises the *same* transform path as live data, so CI and local
  development never require network access.
- **Incremental & idempotent** - watermark-based extraction (`determine_watermark`),
  per-hour determinism in the mock generator, and composite-key dedup guarantee
  re-runs never duplicate or re-fetch history.
- **Server-side station filters** - every NWDP resource is queried once per
  station with `filters={"Station": name}`, cutting a full-basin backfill from
  ~1.06M rows to ~30K per station.
- **Official flood thresholds** - WL / DL / HFL verified against CWC flood
  bulletins and the CWC site lists (see [Monitored stations](#monitored-stations)).
- **Rolling retention** - records older than 365 days relative to the newest
  reading are trimmed automatically.
- **Atomic publishing** - CSV / JSON are written via temp-file + rename, so a
  crash can never leave a truncated dataset.
- **Automated hourly refresh** - GitHub Actions runs the pipeline on a cron
  schedule and auto-commits the refreshed dataset.

---

## Architecture

The pipeline is split into four layers with a strict execution order
(`main.py` → `extract` → `transform` → `load` → `validate`):

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Extract** | `extract.py` | Watermark computation; resilient (retry/backoff) paginated fetch from NWDP CKAN with per-station `filters`; deterministic mock stream generation. |
| **Transform** | `transform.py` | Provider-field aliasing, timestamp parsing (naive IST `DD-MM-YYYY HH:MM` → UTC), numeric coercion, station metadata join, flood-status derivation, dedup keys, canonical schema enforcement. |
| **Load** | `load.py` | Idempotent merge with the persisted CSV, 365-day retention, atomic CSV/JSON writes, Parquet output, per-station partitions, open-data metadata. |
| **Validate** | `validation.py` | Duplicate-timestamp checks, null-percentage report, per-station executive summary. |
| **Config** | `config.py` | Typed dataclasses for every tunable knob: API endpoints/resources, station catalogue, pipeline paths. |

### Design principles

- **Pure transform layer** - `transform.py` contains no I/O or random state,
  making it trivially unit-testable.
- **Fail-fast configuration** - invalid thresholds, coordinates, or settings
  are rejected in `__post_init__` before any network/file activity.
- **Missing data is preserved, never fabricated** - `NaN` gauge readings are
  kept (and reported as a null-percentage), and `flood_status` stays `NaN`
  when the level or thresholds are missing.
- **CI-safe** - a misbehaving upstream feed cannot take a refresh down;
  validation is informational by default.

---

## Data source

- **Portal**: [National Water Data Portal (NWDP)](https://nwdp.nwic.gov.in/) - a CKAN portal
- **Endpoint**: `https://nwdp.nwic.gov.in/api/3/action/datastore_search`
- **Authentication**: none (public data). An optional API key is read from the
  `CWC_API_KEY` environment variable and passed as `api-key` when set.
- **Packages**:
  - *River Water Level (Telemetry - Hourly), CWC* - `68600163-c5a0-4327-aa1a-fa7157b86cce`
  - *River Water Level (Manual - Hourly), CWC* - `d951a09c-6cf8-470e-be77-e80116f13d34`
- **Partitioning**: resources are sliced by **basin × time-range**
  (`(2021 - 2025)` / `(2026 - 2030)`). The 10 registered resources are listed
  in `SOURCE_RESOURCES` in `config.py`:

  | Basin | Kind | Time range |
  |-------|------|------------|
  | Godavari | telemetry | (2021 - 2025) / (2026 - 2030) |
  | Krishna | telemetry | (2021 - 2025) / (2026 - 2030) |
  | Cauvery | telemetry | (2021 - 2025) / (2026 - 2030) |
  | Narmada | telemetry | (2021 - 2025) / (2026 - 2030) |
  | Yamuna Basin | manual | (2021 - 2025) / (2026 - 2030) |

  Resources whose year range cannot contain records newer than the extraction
  watermark are skipped entirely (`_resource_in_window`).

> **Note on the Ganga basin:** despite the original design targeting Ganga /
> Varanasi, no Ganga river-level package exists on NWDP. The pipeline therefore
> monitors the peninsular telemetry basins plus the Yamuna Basin manual
> package (which contains only Chitrakoot and Juddo Dam stations).

---

## Monitored stations

Stations are matched to incoming records by their exact NWDP `Station` value
(case/whitespace-insensitive), configured in `DEFAULT_STATIONS` in `config.py`.

| Code | Station | Basin | Kind | WL (m) | DL (m) | HFL (m) | Status |
|------|---------|-------|------|--------|--------|---------|--------|
| `YMNDLH` | Yamuna at Old Delhi Railway Bridge | Yamuna Basin | manual | 204.50 | 205.33 | 207.49 | verified |
| `YMNCKT` | Yamuna at Chitrakoot | Yamuna Basin | manual | 137.00 | 137.50 | 138.50 | VERIFY* |
| `PTHNGD` | Godavari at Paithan (Jaikwadi Dam) | Godavari | telemetry | 463.91 | 465.50 | 465.50 | VERIFY* |
| `GDBDCL` | Godavari at Bhadrachalam | Godavari | telemetry | 45.72 | 48.77 | 55.66 | verified |
| `KRSNWDP` | Krishna at Wadenepally | Krishna | telemetry | 23.00 | 23.50 | 24.00 | VERIFY* |
| `CAVMSI` | Cauvery at Musiri | Cauvery | telemetry | 82.115 | 83.115 | 86.175 | verified |
| `NRMMDL` | Narmada at Mandla | Narmada | telemetry | 437.2 | 437.8 | 439.405 | verified |

**Threshold provenance (verified 2026-08-18):**

- **YMNDLH** - CWC flood bulletins 2022-2025 (HFL 207.49 m on 06-09-1978; the
  207.55 m record of 2023 exceeded it).
- **GDBDCL / CAVMSI / NRMMDL** - CWC level-forecast site lists (HFL dated
  16-08-1986, 13-11-1977 and 15-07-1974 respectively).
- **PTHNGD** - CWC inflow-forecast table: FRL 463.91 m / MWL 465.5 m, mapped as
  WL=FRL and DL=HFL=MWL.

**\*VERIFY** - stations where the NWDP gauge datum does **not** match the
published CWC datum:

- **YMNCKT** - NWDP reads ~136-138 m (Mandakini gauge at Chitrakoot town) vs a
  published 93.5 m danger level for the Yamuna/Rajapur gauge. Thresholds are
  bracketed around observed NWDP data.
- **KRSNWDP** - NWDP reads ~22-24 m vs a published CWC HFL of 42.494 m (MSL
  datum). Bracketed around observed data; the telemetry also contains junk
  spikes (e.g. a 790 m reading).
- **PTHNGD** - NWDP telemetry is low quality (many `NaN`s, negative readings,
  max ~432 m vs FRL 463.91 m), so official FRL/MWL are used as-is.

---

## Dataset schema

The published dataset is restricted to a **canonical 12-column schema** (raw
provider extras such as `SlNo`, `_id`, `State`, `District`, `River`, ... are
dropped so the schema stays stable):

| Column | Type | Description |
|--------|------|-------------|
| `station_code` | string | Stable internal station identifier (`YMNDLH`, `GDBDCL`, ...). |
| `station_name` | string | Human-readable station name. |
| `basin_name` | string | Hydrological basin (`Godavari`, `Krishna`, `Cauvery`, `Narmada`, `Yamuna Basin`). |
| `timestamp` | datetime (UTC) | Observation time, timezone-aware. |
| `water_level` | float (m) | River water level in metres above the gauge datum; `NaN` for dropouts. |
| `flood_status` | string | `NORMAL` / `WARNING` / `DANGER` / `HIGH_FLOOD_EXCEEDED` / `NaN`. |
| `warning_level` | float (m) | Warning level (WL) from the station catalogue. |
| `danger_level` | float (m) | Danger level (DL). |
| `high_flood_level` | float (m) | High flood level (HFL). |
| `latitude` | float | WGS84 latitude. |
| `longitude` | float | WGS84 longitude. |
| `dedup_key` | string | Idempotency key `{station_code}_{timestamp}_{water_level}`. |

### Output artefacts

| Path | Description |
|------|-------------|
| `output/cwc_river_water_levels.csv` | Canonical dataset (all stations). |
| `output/cwc_river_water_levels.parquet` | Same data in Parquet (columnar). |
| `output/partitions/<STATION_CODE>.csv` | One CSV per station. |
| `output/dataset-metadata.json` | Open-data metadata: temporal bounds, per-station records & thresholds, alert-status distribution, source resources. |

In mock mode everything is written under `mock_output/` instead, so synthetic
data can never overwrite the real published dataset.

---

## Flood alert categorisation

`derive_flood_status` implements CWC's four-category categorisation relative to
the station's WL / DL / HFL:

| Condition | `flood_status` |
|-----------|----------------|
| level < WL | `NORMAL` |
| WL ≤ level < DL | `WARNING` |
| DL ≤ level < HFL | `DANGER` |
| level ≥ HFL | `HIGH_FLOOD_EXCEEDED` |
| level or thresholds missing | `NaN` (never `NORMAL`) |

---

## Quickstart

Requirements: **Python 3.10+**, `pip`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run offline (deterministic mock data, no network / API key needed)
python main.py

# 3. Run against the real NWDP API (public data)
python main.py --live

# 4. Backfill full history from 2021-01-01
python main.py --live --backfill --start-date 2021-01-01

# 5. Restrict to specific basins / resources
python main.py --live --basins godavari,yamuna --time-ranges "(2026 - 2030)"
```

After each run, inspect the artefacts under `output/` (or `mock_output/`) and
the printed **executive summary** (record counts, min/max levels, missing-gauge
percentages and per-station `flood_status` distributions).

---

## Dashboard UI

**Live app: [https://ETLPipeline.streamlit.app](https://ETLPipeline.streamlit.app)**

A Streamlit dashboard (`app.py`) visualises the committed dataset and forecasts
flood-prone zones. The light UI stack (streamlit, plotly, folium, xgboost) is
bundled into `requirements.txt`, so one install covers both the pipeline and the
dashboard. TimesFM + torch stay in a separate optional file.

```bash
pip install -r requirements.txt      # ETL + light UI stack (XGBoost engine ready)
pip install -r requirements-timesfm.txt  # OPTIONAL: TimesFM 2.5 + torch (~800 MB)
streamlit run app.py
```

Without the TimesFM extras the dashboard boots with the **XGBoost fallback**
engine (models are committed in `models/`); install them to enable the
TimesFM 2.5 primary engine.

| Tab | What it shows |
|-----|---------------|
| **Overview** | KPIs, folium map with alert-coloured markers + risk-scaled buffers, and a risk-ranked station table (XGBoost snapshot, 72 h default). |
| **Water Level Explorer** | Historical level chart with WL / DL / HFL threshold lines and the latest readings. |
| **Flood Forecast** | Forecast fan chart (observed tail + mean + 10-90% band), last/peak level, risk score and hours-to-danger. |
| **Data & Quality** | Per-station record counts, schema, and documented NWDP data-quality caveats. |

Sidebar controls: **forecast engine**, **horizon** (24-168 h), and a
**Refresh now** button that re-runs the incremental ETL against the live NWDP
API (on Streamlit Cloud the container is ephemeral, so refreshed data lasts
only for the current session).

### Forecasting engines

- **TimesFM 2.5** (Google, `timesfm==2.0.2`, torch) - zero-shot foundation
  model used as the primary engine. The ~800 MB checkpoint is downloaded on
  first use and cached for the session (`st.cache_resource`). Forecasts run on
  a cleaned hourly series (resampled, MAD-clipped, threshold-banded, junk
  plateaus removed).
- **XGBoost fallback** (global GBM, committed in `models/`) - a single
  `reg:squarederror` center model trained offline on all stations plus
  per-station residual band offsets (`xgb_bands.json`), recursive multi-step.
  Runs in milliseconds with no network. Retrain after significant data changes:

```bash
python scripts/train_xgboost.py          # writes models/xgb_q50.json + xgb_bands.json
```

The band on the fan chart is the model's 10-90% interval (TimesFM quantile
head columns; XGBoost per-station residual quantiles). The **risk score** (0-100)
is an interpretable heuristic blending threshold proximity, forecast crossing
and the steepest 24 h rise - not a hydrological model.

### Deploying to Streamlit Community Cloud

1. Push this repo (public, or private with the Cloud app connected to GitHub).
2. New app → select the repo, set **Main file path** = `app.py`.
3. Under **Advanced settings** set the requirements file to
   **`requirements-cloud.txt`** (ETL + UI stack + TimesFM 2.5) and **Python
   version to 3.13**. Do not use Python 3.14: `pyarrow` has no 3.14 wheel yet,
   so the build falls back to compiling from source and fails on `cmake`.
4. The app URL is generated from the repo name, so creating the app activates
   **https://ETLPipeline.streamlit.app**.
5. Need a faster, lighter build (or the torch install exceeds the free-tier
   budget)? Point Advanced settings at **`requirements-ui.txt`** instead to run
   on the XGBoost engine only — or leave the default **`requirements.txt`**,
   which now also installs the full UI stack as a fallback (still XGBoost-only,
   no TimesFM). Keep Python 3.13 either way.

---

## CLI reference

```
usage: main.py [-h] [--use-mock | --live] [--backfill] [--start-date START_DATE]
               [--basins BASINS] [--time-ranges TIME_RANGES] [--seed SEED]
               [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

| Flag | Description |
|------|-------------|
| `--use-mock` | Force offline synthetic mode (no API access). |
| `--live` | Force live NWDP fetch (public; `CWC_API_KEY` optional). |
| `--backfill` | Ignore the existing watermark and extract from the start date. |
| `--start-date ISO` | Start date used when no watermark exists (default `2026-01-01`). |
| `--basins a,b` | Comma-separated basin filter, e.g. `godavari,krishna,yamuna`. |
| `--time-ranges "a,b"` | Resource time-range filter, e.g. `"(2026 - 2030)"`. |
| `--seed N` | Determinism seed for the mock generator (default `42`). |
| `--log-level L` | Override logging verbosity. |

---

## Configuration

All knobs live in `config.py` (typed dataclasses; validated at construction):

- **`APIConfig`** - NWDP endpoint, time ranges, `source_timezone="Asia/Kolkata"`,
  pagination `max_per_page=1000`, retry policy, mock toggle,
  `default_start_date="2026-01-01"`.
- **`SOURCE_RESOURCES`** - the 10 registered NWDP basin × time-range resources.
- **`Station` / `StationConfig` / `DEFAULT_STATIONS`** - station catalogue
  (source name, kind, WL/DL/HFL, coordinates, aliases).
- **`PipelineConfig`** - output paths, `retention_days=365`, Parquet toggle.

### Environment / GitHub variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CWC_USE_MOCK` | `"true"` forces mock mode, `"false"` forces live. Unset ⇒ mock. | `true` |
| `CWC_API_KEY` | Optional NWDP API key, sent as `api-key` when set. | *(unset)* |

On GitHub the repository variable `CWC_USE_MOCK=false` switches the workflow to
live data; `CWC_API_KEY` is a (optional) repository secret.

---

## GitHub Actions

The workflow (`.github/workflows/etl_schedule.yml`) runs the pipeline and
**auto-commits** the refreshed dataset back to `main`:

- **Schedule**: every hour at minute 15 (UTC), via `schedule: cron`.
- **Manual dispatch** (Actions → *ETL Hourly Schedule* → *Run workflow*):
  - `backfill` - full backfill ignoring the watermark.
  - `basins` - basin filter, e.g. `godavari,yamuna`.
  - `start_date` - ISO start date for backfills.
- **Concurrency**: a single `etl-hourly` group prevents overlapping runs.
- **Timeout**: 45 minutes per run.
- **Environment**: `CWC_USE_MOCK` from repo variables, `CWC_API_KEY` from secrets.
- **Auto-commit**: `git add output/ mock_output/` → commit
  `chore(etl): hourly river water level dataset refresh [skip ci]` → push.

> When pushing code changes locally, `git pull --rebase origin main` first —
> the workflow auto-commits can advance `main` between your local commits.

### Triggering a run with `gh`

```bash
gh workflow run "ETL Hourly Schedule" --ref main
gh run watch <run-id> --exit-status
```

---

## Operational history

The live dataset was brought up in three approved steps:

1. **Go live** - repo created and published, `CWC_USE_MOCK=false`, first live
   run: **7,971 records / 5 stations / 0 duplicates**.
2. **Historical backfill** - added per-station server-side `filters`, ran a
   2021-2025 backfill: **12,291 records / 5 stations / 0 duplicates**
   (Jaikwadi Dam's pre-2026 history was trimmed by the 365-day retention).
3. **Threshold verification** - WL / DL / HFL updated to official CWC figures
   (see [Monitored stations](#monitored-stations)).

Notable fixes along the way:

- Timestamp parsing for days > 12 (explicit `%d-%m-%Y %H:%M` format + `dayfirst`
  fallback).
- Duplicate `water_level__merge` columns from multi-column coalescing (now
  uniquely numbered merge targets).
- `pyarrow` dependency added for Parquet output on the CI runner.
- Mixed-dtype raw extras (`SlNo` string vs numeric) crashing Parquet serialisation
  - raw provider extras are now dropped, keeping the published schema canonical.

---

## Data quality notes

- **Telemetry noise**: some stations contain erroneous spikes (e.g. Musiri
  ~677 m, Mandla ~1394 m, Wadenepally ~790 m) that will be classified as
  `HIGH_FLOOD_EXCEEDED`. These are upstream sensor artifacts, not floods.
- **Gauge datum mismatches**: for `YMNCKT`, `KRSNWDP`, and `PTHNGD` the NWDP
  gauge datum differs from the published CWC datum; thresholds are bracketed
  around observed data and flagged `VERIFY` (see
  [Monitored stations](#monitored-stations)).
- **Retention**: the published dataset keeps only the most recent 365 days.
  Run `--backfill --start-date 2021-01-01` to rebuild full history.
- **Validation is informational** - it never aborts a run, so a partially bad
  upstream feed cannot take the hourly refresh down.

---

## Project layout

```
.
├── main.py                 # Orchestration: extract -> transform -> load -> validate
├── config.py               # Typed config: API, resources, station catalogue, paths
├── extract.py              # NWDP fetch + watermark + deterministic mock stream
├── transform.py            # Aliasing, timestamps, thresholds, flood status, schema
├── load.py                 # Idempotent merge, retention, atomic writes, metadata
├── validation.py           # Duplicate / null / summary checks
├── app.py                  # Streamlit dashboard (UI, see Dashboard UI)
├── ui/
│   ├── features.py         # Shared feature engineering / series sanitisation
│   ├── data.py             # Cached dataset loaders + on-demand refresh
│   ├── forecast.py         # TimesFM 2.5 + XGBoost engines, risk heuristic
│   └── viz.py              # Plotly charts + folium risk map
├── scripts/
│   └── train_xgboost.py    # Trains + commits the XGBoost fallback models
├── tests/                  # pytest suite (features, forecast, viz, transform)
├── models/                 # Committed XGBoost artifacts (xgb_q50.json, xgb_bands.json)
├── requirements.txt        # ETL + light UI stack (requests, pandas, numpy, pyarrow, streamlit, plotly, folium, xgboost)
├── requirements-ui.txt     # Light UI stack (included by requirements.txt)
├── requirements-timesfm.txt# timesfm 2.5 + torch (optional TimesFM engine)
├── requirements-cloud.txt  # full stack for Streamlit Cloud (ETL + UI + TimesFM)
├── requirements-dev.txt    # pytest
├── .github/workflows/
│   ├── etl_schedule.yml    # Hourly cron + manual dispatch + auto-commit
│   └── model_retrain.yml   # Weekly XGBoost fallback retrain + auto-commit
├── output/                 # Published live dataset (CSV / Parquet / partitions / metadata)
├── mock_output/            # Mock-mode artefacts (kept strictly separate)
└── etl_pipeline.py         # Legacy monolith (superseded; kept for reference)
```

---

## Disclaimer

Flood levels, thresholds, and categorisations are provided for informational
purposes only. The WL / DL / HFL figures come from CWC publications and press
reports of CWC bulletins; stations flagged `VERIFY` use locally bracketed
values. Always refer to official Central Water Commission flood bulletins for
operational decisions.
