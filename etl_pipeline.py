"""
Production-Grade Paris Flood Monitoring ETL Pipeline
Extracts hydrological data from the Hub'Eau API, transforms/cleans it,
and loads it into CSV and metadata formats for dataset publishing.
"""

import json
import os
import random
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ==========================================
# 1. CONFIGURATION LAYER
# ==========================================

@dataclass
class APIConfig:
    """API configuration for Hub'Eau data fetching."""
    use_mock: bool = True
    base_url: str = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
    metric: str = "HIXnJ"  # Daily max water level
    max_per_page: int = 20000
    timeout_seconds: int = 60

    def __post_init__(self):
        if self.max_per_page <= 0:
            raise ValueError("max_per_page must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class StationConfig:
    """Station monitoring configuration."""
    station_codes: List[str] = field(
        default_factory=lambda: [
            "F700000109",
            "F700000110",
            "F700000111",
            "F700000102",
            "F700000103",
        ]
    )
    flood_threshold_mm: int = 6000
    earliest_date: str = "1900-01-01"


@dataclass
class IndianAPIConfig:
    """API configuration for Indian Water Commission data fetching."""
    use_mock: bool = True
    base_url: str = "https://cwc.gov.in/api/hydrology/obs_elab"
    metric: str = "water_level"  # Daily water level
    max_per_page: int = 5000
    timeout_seconds: int = 120

    def __post_init__(self):
        if self.max_per_page <= 0:
            raise ValueError("max_per_page must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class IndiaStationConfig:
    """Indian station monitoring configuration."""
    station_codes: List[str] = field(
        default_factory=lambda: [
            "CTR/UP/01",     # Uttar Pradesh
            "CTR/MH/01",     # Maharashtra
            "CTR/RJ/01",     # Rajasthan
            "CTR/AP/01",     # Andhra Pradesh
            "CTR/KT/01",     # Karnataka
        ]
    )
    flood_threshold_mm: int = 8000
    earliest_date: str = "1950-01-01"


@dataclass
class KaggleConfig:
    """Kaggle dataset publishing configuration."""
    dataset_slug: str = "grimespoint/paris-flood-dataset"
    input_csv: str = "kaggle/input/datasets/{slug}/paris_flood_dataset.csv"
    output_dir: Path = field(default_factory=lambda: Path("kaggle/working/kaggle_dataset"))
    mock_output_dir: Path = field(default_factory=lambda: Path("mock_output"))
    output_filename: str = "paris_flood_dataset.csv"
    mock_output_filename: str = "mock_flood_dataset.csv"
    metadata_filename: str = "dataset-metadata.json"
    title: str = "Paris flood dataset"
    keywords: List[str] = field(
        default_factory=lambda: [
            "tabular",
            "weather and climate",
            "environment",
            "europe",
            "time series analysis",
        ]
    )
    geospatial_coverage: str = "Paris, France"
    update_frequency: str = "Weekly"
    license_name: str = "CC0-1.0"
    output_csv_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    mock_output_path: Path = field(init=False)

    def __post_init__(self):
        self.input_csv = self.input_csv.format(slug=self.dataset_slug)
        self.output_csv_path = self.output_dir / self.output_filename
        self.metadata_path = self.output_dir / self.metadata_filename
        self.mock_output_path = self.mock_output_dir / self.mock_output_filename


# Regional configurations
REGION_CONFIGS = {
    "france": {
        "api": APIConfig(),
        "stations": StationConfig(),  # Use local config copy
        "kaggle": KaggleConfig(dataset_slug="grimespoint/paris-flood-dataset",
                               geospatial_coverage="Paris, France"),
    },
    "india": {
        "api": IndianAPIConfig(),
        "stations": IndiaStationConfig(),
        "kaggle": KaggleConfig(dataset_slug="grimespoint/india-flood-dataset",
                               geospatial_coverage="India"),
    },
}


# Global Singletons - default to France
API_CONFIG = APIConfig()
STATION_CONFIG = StationConfig()
KAGGLE_CONFIG = KaggleConfig()

# Mappings & Schema Order
API_TO_EN = {
    "code_site": "location_code",
    "code_station": "station_code",
    "date_obs_elab": "record_date",
    "resultat_obs_elab": "water_level_mm",
    "date_prod": "data_production_date",
    "code_statut": "validation_status_code",
    "libelle_statut": "validation_status",
    "code_methode": "production_method_code",
    "libelle_methode": "production_method",
    "code_qualification": "quality_code",
    "libelle_qualification": "quality_assessment",
    "longitude": "longitude",
    "latitude": "latitude",
    "grandeur_hydro_elab": "hubeau_elab_code",
}

EN_TO_API = {v: k for k, v in API_TO_EN.items()}

CATEGORICAL_MAPPINGS = {
    "validation_status": {
        "Donnée validée": "validated",
        "Donnée brute": "raw",
        "Donnée pré-validée": "pre-validated",
    },
    "quality_assessment": {
        "Bonne": "good",
        "Non qualifiée": "unqualified",
        "Douteuse": "dubious",
    },
    "production_method": {
        "Calculée": "calculated",
        "Mesurée": "measured",
        "Expertisée": "expert-reviewed",
    },
}

# Indian categorical mappings (Hindi/regional status terms)
INDIA_CATEGORICAL_MAPPINGS = {
    "validation_status": {
        "मान्य": "validated",
        "अमान्य": "unvalidated",
        "प्रारंभिक": "pre-validated",
    },
    "quality_assessment": {
        "अच्छी": "good",
        "खराब": "poor",
        "संतोषजनक": "satisfactory",
    },
    "flood_severity": {
        "अत्यधिक": "severe",
        "मध्यम": "moderate",
        "कम": "low",
    },
}

COLUMN_ORDER = [
    "station_code",
    "record_date",
    "water_level_mm",
    "flood_alert",
    "hubeau_elab_code",
    "data_production_date",
    "validation_status_code",
    "validation_status",
    "production_method_code",
    "production_method",
    "quality_code",
    "quality_assessment",
    "location_code",
    "longitude",
    "latitude",
]


# ==========================================
# 2. EXTRACT STEP
# ==========================================

# Mappings & Schema Order
API_TO_EN = {
    "code_site": "location_code",
    "code_station": "station_code",
    "date_obs_elab": "record_date",
    "resultat_obs_elab": "water_level_mm",
    "date_prod": "data_production_date",
    "code_statut": "validation_status_code",
    "libelle_statut": "validation_status",
    "code_methode": "production_method_code",
    "libelle_methode": "production_method",
    "code_qualification": "quality_code",
    "libelle_qualification": "quality_assessment",
    "longitude": "longitude",
    "latitude": "latitude",
    "grandeur_hydro_elab": "hubeau_elab_code",
}

EN_TO_API = {v: k for k, v in API_TO_EN.items()}

CATEGORICAL_MAPPINGS = {
    "validation_status": {
        "Donnée validée": "validated",
        "Donnée brute": "raw",
        "Donnée pré-validée": "pre-validated",
    },
    "quality_assessment": {
        "Bonne": "good",
        "Non qualifiée": "unqualified",
        "Douteuse": "dubious",
    },
    "production_method": {
        "Calculée": "calculated",
        "Mesurée": "measured",
        "Expertisée": "expert-reviewed",
    },
}

COLUMN_ORDER = [
    "station_code",
    "record_date",
    "water_level_mm",
    "flood_alert",
    "hubeau_elab_code",
    "data_production_date",
    "validation_status_code",
    "validation_status",
    "production_method_code",
    "production_method",
    "quality_code",
    "quality_assessment",
    "location_code",
    "longitude",
    "latitude",
]


# ==========================================
# 2. EXTRACT STEP
# ==========================================

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file using the Null Object pattern if missing."""
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False, parse_dates=True, delimiter=",")
    return pd.DataFrame()


def determine_update_range(existing: pd.DataFrame, region: str = "france") -> Tuple[bool, Optional[str]]:
    """Determine whether incremental update is needed and starting date."""
    config = REGION_CONFIGS[region]["stations"]

    if existing.empty:
        print("No existing data found. Will fetch all data from earliest date.")
        return True, config.earliest_date

    if "date_obs_elab" in existing.columns:
        record_colname = "date_obs_elab"
    elif "record_date" in existing.columns:
        record_colname = "record_date"
    else:
        raise KeyError("Missing date column: expected 'date_obs_elab' or 'record_date'")

    s = pd.to_datetime(existing[record_colname], errors="coerce")
    last_day = s.max().date()
    yesterday = date.today() - timedelta(days=1)

    if last_day >= yesterday:
        print("\nDataset already covers yesterday or later. No update needed.")
        return False, None

    next_day = last_day + pd.Timedelta(days=1)
    print(f"\nWill retrieve data starting from: {next_day}")
    return True, next_day.isoformat()


def generate_mock_api_data(station_code: str, start_date: str, num_days: int = 10) -> List[Dict]:
    """Generate realistic mock API observations."""
    start = pd.to_datetime(start_date).date()
    records = []
    validation_statuses = ["Donnée validée", "Donnée brute", "Donnée pré-validée"]
    qualities = ["Bonne", "Non qualifiée", "Douteuse"]
    methods = ["Mesurée", "Calculée", "Expertisée"]

    for i in range(num_days):
        obs_date = start + timedelta(days=i)
        base_level = 5500 + int(station_code[-2:])
        noise = random.randint(-200, 200)
        water_level = base_level + noise

        record = {
            "code_site": "mock_" + station_code[1:],
            "code_station": "mock_" + station_code,
            "date_obs_elab": obs_date.isoformat(),
            "resultat_obs_elab": water_level,
            "date_prod": (obs_date + timedelta(days=1)).isoformat(),
            "code_statut": "1",
            "libelle_statut": random.choice(validation_statuses),
            "code_methode": "1",
            "libelle_methode": random.choice(methods),
            "code_qualification": "1",
            "libelle_qualification": random.choice(qualities),
            "longitude": 2.3522 + random.uniform(-0.01, 0.01),
            "latitude": 48.8566 + random.uniform(-0.01, 0.01),
            "grandeur_hydro_elab": "mock_HIXnJ",
        }
        records.append(record)
    return records


def fetch_single_station_data(station_code: str, start_date: str, use_mock: bool = True) -> pd.DataFrame:
    """Fetch station observations via mock generator or live HTTP session with pagination."""
    if use_mock:
        data = generate_mock_api_data(station_code, start_date, num_days=7)
        page_df = pd.DataFrame(data)
        page_df["date_obs_elab"] = pd.to_datetime(page_df["date_obs_elab"], errors="coerce").dt.normalize()
        return page_df

    session = requests.Session()
    frames = []
    cursor = start_date

    while True:
        params = {
            "code_entite": station_code,
            "grandeur_hydro_elab": API_CONFIG.metric,
            "date_debut_obs_elab": cursor,
            "size": API_CONFIG.max_per_page,
        }
        try:
            response = session.get(API_CONFIG.base_url, params=params, timeout=API_CONFIG.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching data for station {station_code}: {e}")
            break

        data = response.json().get("data", [])
        if not data:
            break

        page_df = pd.DataFrame(data)
        page_df["date_obs_elab"] = pd.to_datetime(page_df["date_obs_elab"], errors="coerce").dt.normalize()
        frames.append(page_df)

        last_page_date = page_df["date_obs_elab"].max()
        yesterday = date.today() - timedelta(days=1)

        if pd.isna(last_page_date) or last_page_date.date() >= yesterday:
            break

        cursor = (last_page_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if len(data) < API_CONFIG.max_per_page:
            break

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def fetch_all_data(start_date: str, use_mock: bool = True, region: str = "france") -> pd.DataFrame:
    """Orchestrate data extraction across all monitored stations."""
    config = REGION_CONFIGS[region]
    frames = []
    for station_code in config["stations"].station_codes:
        print(f"Fetching data for station {station_code}...")
        df_station = fetch_single_station_data(station_code, start_date, use_mock=use_mock, region=region)
        if not df_station.empty:
            print(f"  Got {len(df_station)} records")
            frames.append(df_station)
        else:
            print("  (no data)")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def generate_indian_mock_api_data(station_code: str, start_date: str, num_days: int = 10) -> List[Dict]:
    """Generate realistic mock Indian API observations."""
    start = pd.to_datetime(start_date).date()
    records = []
    validation_statuses = ["मान्य", "अमान्य", "प्रारंभिक"]
    qualities = ["अच्छी", "खराब", "संतोषजनक"]
    methods = ["स्वचालित", "हाथ से", "विशेषज्ञ"]

    for i in range(num_days):
        obs_date = start + timedelta(days=i)
        base_level = 300 + int(station_code.split("/")[1]) * 100
        noise = random.randint(-50, 50)
        water_level = base_level + noise

        record = {
            "code_site": "indian_mock_" + station_code.split("/")[0],
            "code_station": "indian_mock_" + station_code,
            "date_obs_elab": obs_date.isoformat(),
            "resultat_obs_elab": water_level,
            "date_prod": (obs_date + timedelta(days=1)).isoformat(),
            "code_statut": "1",
            "libelle_statut": random.choice(validation_statuses),
            "code_methode": "1",
            "libelle_methode": random.choice(methods),
            "code_qualification": "1",
            "libelle_qualification": random.choice(qualities),
            "longitude": 77.2168 + random.uniform(-0.5, 0.5),
            "latitude": 28.6139 + random.uniform(-0.5, 0.5),
            "grandeur_hydro_elab": "IN_WL",
        }
        records.append(record)
    return records


def fetch_single_station_data(station_code: str, start_date: str, use_mock: bool = True, region: str = "france") -> pd.DataFrame:
    """Fetch station observations via mock generator or live HTTP session with pagination."""
    if use_mock:
        if region == "india":
            data = generate_indian_mock_api_data(station_code, start_date, num_days=7)
        else:
            data = generate_mock_api_data(station_code, start_date, num_days=7)
        page_df = pd.DataFrame(data)
        page_df["date_obs_elab"] = pd.to_datetime(page_df["date_obs_elab"], errors="coerce").dt.normalize()
        return page_df

    session = requests.Session()
    frames = []
    cursor = start_date

    api_config = REGION_CONFIGS[region]["api"]

    while True:
        params = {
            "code_entite": station_code,
            "grandeur_hydro_elab": api_config.metric,
            "date_debut_obs_elab": cursor,
            "size": api_config.max_per_page,
        }
        try:
            response = session.get(api_config.base_url, params=params, timeout=api_config.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching data for station {station_code}: {e}")
            break

        data = response.json().get("data", [])
        if not data:
            break

        page_df = pd.DataFrame(data)
        page_df["date_obs_elab"] = pd.to_datetime(page_df["date_obs_elab"], errors="coerce").dt.normalize()
        frames.append(page_df)

        last_page_date = page_df["date_obs_elab"].max()
        yesterday = date.today() - timedelta(days=1)

        if pd.isna(last_page_date) or last_page_date.date() >= yesterday:
            break

        cursor = (last_page_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if len(data) < api_config.max_per_page:
            break

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ==========================================
# 3. TRANSFORM STEP
# ==========================================

def convert_to_date(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def auto_convert_columns(df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = []
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if len(sample) == 0:
                continue
            try:
                pd.to_datetime(sample, errors="raise", format="mixed")
                datetime_cols.append(col)
                continue
            except (ValueError, TypeError):
                pass
            try:
                pd.to_numeric(sample, errors="raise")
                numeric_cols.append(col)
                continue
            except (ValueError, TypeError):
                pass

    df = convert_to_date(df, datetime_cols)
    df = convert_to_numeric(df, numeric_cols)
    return df


def rename_to_english(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    columns_to_rename = {
        src: dst for src, dst in API_TO_EN.items() if src in df.columns and dst not in df.columns
    }
    return df.rename(columns=columns_to_rename)


def rename_to_api_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    columns_to_rename = {k: v for k, v in EN_TO_API.items() if k in df.columns}
    return df.rename(columns=columns_to_rename)


def apply_categorical_mappings(df: pd.DataFrame, region: str = "france") -> pd.DataFrame:
    df = df.copy()
    mappings = CATEGORICAL_MAPPINGS if region == "france" else INDIA_CATEGORICAL_MAPPINGS
    for col_name, mapping in mappings.items():
        if col_name in df.columns:
            df[col_name] = df[col_name].map(mapping).fillna(df[col_name])
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "water_level_mm" in df.columns:
        df["flood_alert"] = df["water_level_mm"] > STATION_CONFIG.flood_threshold_mm
    return df


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    present_cols = [c for c in COLUMN_ORDER if c in df.columns]
    other_cols = [c for c in df.columns if c not in present_cols]
    return df[present_cols + other_cols]


def create_dedup_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    if "code_station" in df.columns:
        parts.append(df["code_station"].astype(str))
    elif "station_code" in df.columns:
        parts.append(df["station_code"].astype(str))

    if "date_obs_elab" in df.columns:
        parts.append(pd.to_datetime(df["date_obs_elab"]).dt.strftime("%Y-%m-%d"))
    elif "record_date" in df.columns:
        parts.append(pd.to_datetime(df["record_date"]).dt.strftime("%Y-%m-%d"))

    if "resultat_obs_elab" in df.columns:
        parts.append(df["resultat_obs_elab"].astype(str))
    elif "water_level_mm" in df.columns:
        parts.append(df["water_level_mm"].astype(str))

    if not parts:
        return pd.Series(index=df.index, dtype="object")
    return pd.Series(["_".join(row) for row in zip(*parts)], index=df.index)


def remove_duplicates(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Enforce idempotency by filtering out existing keys using O(1) set lookups."""
    if existing.empty or new.empty:
        return new.copy()

    existing_std = auto_convert_columns(existing)
    new_std = auto_convert_columns(new)

    existing_keys = set(create_dedup_key(existing_std).dropna())
    new_keys = create_dedup_key(new_std)

    mask = ~new_keys.isin(existing_keys)
    return new.iloc[new_keys[mask].index].copy()


def postprocess(df: pd.DataFrame, region: str = "france") -> pd.DataFrame:
    """Linear pure-function sequencing pipeline."""
    if df.empty:
        return df

    df = df.copy()
    print("  1. Converting types...")
    df = auto_convert_columns(df)
    print("  2. Renaming columns (French -> English)...")
    df = rename_to_english(df)
    print("  3. Mapping categorical values...")
    df = apply_categorical_mappings(df, region=region)
    print("  4. Adding derived columns...")
    df = add_derived_columns(df)
    print("  5. Reordering columns...")
    df = order_columns(df)
    print("  6. Sorting and resetting index...")
    df = df.sort_values(["record_date", "station_code"]).reset_index(drop=True)
    return df


# ==========================================
# 4. LOAD STEP
# ==========================================

def create_output_dir(use_mock: bool = False) -> None:
    output_dir = KAGGLE_CONFIG.mock_output_dir if use_mock else KAGGLE_CONFIG.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)


def create_metadata(df: pd.DataFrame, config: KaggleConfig) -> Dict:
    if df.empty or "record_date" not in df.columns:
        first_date = "unknown"
        last_date = "unknown"
    else:
        first_date = pd.to_datetime(df["record_date"]).min().strftime("%Y-%m-%d")
        last_date = pd.to_datetime(df["record_date"]).max().strftime("%Y-%m-%d")

    return {
        "title": config.title,
        "id": config.dataset_slug,
        "licenses": [{"name": config.license_name}],
        "keywords": config.keywords,
        "temporalCoverage": {"startDate": first_date, "endDate": last_date},
        "geospatialCoverage": config.geospatial_coverage,
        "updateFrequency": config.update_frequency,
    }


def write_metadata(df: pd.DataFrame, use_mock: bool = False) -> None:
    metadata = create_metadata(df, KAGGLE_CONFIG)
    output_dir = KAGGLE_CONFIG.mock_output_dir if use_mock else KAGGLE_CONFIG.output_dir
    metadata_file = output_dir / KAGGLE_CONFIG.metadata_filename
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to {metadata_file}")


def publish_to_kaggle() -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Weekly update: {timestamp}"
    cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(KAGGLE_CONFIG.output_dir),
        "-m",
        message,
        "--dir-mode",
        "zip",
    ]
    print("Publishing to Kaggle...")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("Successfully published to Kaggle.")
    except subprocess.CalledProcessError as e:
        print(f"Error publishing to Kaggle: {e}")
        raise


# ==========================================
# 5. VALIDATION & ORCHESTRATION
# ==========================================

def validate_and_analyze(df: pd.DataFrame) -> None:
    """Post-run sanity checks and summary reporting."""
    print("\n" + "=" * 40)
    print("POST-RUN VALIDATION REPORT")
    print("=" * 40)
    print(f"Total Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date Range: {df['record_date'].min()} to {df['record_date'].max()}")
    print(f"Flood Alert Count: {df['flood_alert'].sum()}")
    print("\nSummary Statistics for Water Level (mm):")
    print(df["water_level_mm"].describe())
    print("\nNull Counts per Column:")
    print(df.isnull().sum())
    print("=" * 40)


def run_etl_pipeline(use_mock: bool = True, region: str = "france") -> pd.DataFrame:
    """Run full idempotent incremental ETL pipeline."""
    config = REGION_CONFIGS[region]

    # 1. Extract Existing Data
    if use_mock:
        if region == "india":
            mock_seed = generate_indian_mock_api_data("CTR/UP/01", "2026-07-01", num_days=5)
        else:
            mock_seed = generate_mock_api_data("F700000109", "2026-07-01", num_days=5)
        loaded_df = pd.DataFrame(mock_seed)
        loaded_df = rename_to_english(loaded_df)
    else:
        loaded_df = load_csv(config["kaggle"].input_csv)

    # 2. Check Range
    should_update, start_date = determine_update_range(loaded_df, region=region)
    if not should_update:
        return loaded_df

    # 3. Extract New Data
    fetched_data = fetch_all_data(start_date, use_mock=use_mock, region=region)

    # 4. Transform: Deduplicate
    deduped_fetched_data = remove_duplicates(loaded_df, fetched_data)

    # 5. Transform: Merge
    new_df_english = rename_to_english(deduped_fetched_data)
    merged = pd.concat([loaded_df, new_df_english], ignore_index=True)

    # 6. Transform: Postprocess
    processed = postprocess(merged, region=region)

    # 7. Load: CSV & Metadata
    create_output_dir(use_mock=use_mock)
    target_csv = config["kaggle"].mock_output_path if use_mock else config["kaggle"].output_csv_path
    processed.to_csv(target_csv, index=False, sep=",")
    write_metadata(processed, use_mock=use_mock)

    return processed


def main() -> None:
    # Set use_mock=True to test locally without hitting the live endpoint
    # Set region="india" for Indian flood data, "france" for Paris data
    final_dataset = run_etl_pipeline(use_mock=True, region="france")
    validate_and_analyze(final_dataset)


if __name__ == "__main__":
    main()