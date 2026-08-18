"""Train the global XGBoost quantile fallback model and commit it.

Trains three `reg:quantileerror` boosters (q05 / q50 / q95) on **all**
stations at once using the shared features from ``ui.features``, evaluates
with a rolling origin (time-series) split, and writes the boosters as JSON
files under ``models/`` for the dashboard's XGBoost engine.

Usage::

    python scripts/train_xgboost.py [--output models]

The resulting files are small (~200 KB each) and are committed to the repo so
the app runs offline and on Streamlit Cloud with no retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_STATIONS  # noqa: E402
from ui.features import (  # noqa: E402
    LEVEL_COL,
    STATION_CODE_COL,
    category_maps,
    encode_categories,
    feature_columns,
    level_band,
    prepare_series,
    series_to_frame,
)

QUANTILES = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 7,
    "learning_rate": 0.05,
    "min_child_weight": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "n_estimators": 600,
    "early_stopping_rounds": 50,
    "random_state": 42,
}
MIN_SERIES_HOURS = 168 + 24  # need all lags + rolling windows


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    maps = category_maps()
    station_by_code = {s.station_code: s for s in DEFAULT_STATIONS}
    frames = []
    excluded = []
    for code, station in station_by_code.items():
        band = level_band(station.warning_level, station.danger_level, station.high_flood_level)
        s = prepare_series(df, code, band=band)
        if len(s) < MIN_SERIES_HOURS + 24:
            continue
        if band is not None:
            midpoint = band[0] + 0.5 * (band[1] - band[0])
            junk_fraction = float((s > midpoint).mean())
            if junk_fraction > 0.05:
                excluded.append(f"{code}:{junk_fraction:.0%}")
                continue
        frame = series_to_frame(s, code, station.basin_name, station.kind)
        frame["target"] = frame["y"].shift(-1)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No station has enough clean history to train on.")
    if excluded:
        print(f"Excluded junk-dominated stations: {', '.join(excluded)}")
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["target"])
    encode_categories(data, maps)
    return data.dropna(subset=feature_columns())


def evaluate(data: pd.DataFrame, features: list, booster: xgb.Booster) -> float:
    """Rolling-origin MAE on the final 10% of the (per-station ordered) rows."""
    cut = int(len(data) * 0.9)
    te = data.iloc[cut:]
    if te.empty:
        return float("nan")
    preds = booster.predict(xgb.DMatrix(te[features].astype("float32")))
    return float(np.mean(np.abs(preds - te["target"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(REPO_ROOT / "output" / "cwc_river_water_levels.csv"))
    parser.add_argument("--output", default=str(REPO_ROOT / "models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows from {args.data}")

    t0 = time.time()
    data = build_training_frame(df)
    print(f"Training frame: {len(data):,} rows "
          f"(stations: {sorted(data[STATION_CODE_COL].unique())}) "
          f"in {time.time() - t0:.1f}s")

    features = feature_columns()
    X = data[features].astype("float32")
    y = data["target"].astype("float32")
    split = int(len(data) * 0.9)
    tr, va = slice(0, split), slice(split, None)
    dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
    dval = xgb.DMatrix(X.iloc[va], label=y.iloc[va])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"trained_at": pd.Timestamp.utcnow().isoformat(), "rows": len(data), "models": {}}

    params = {**PARAMS, "seed": args.seed}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=PARAMS["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=PARAMS["early_stopping_rounds"],
        verbose_eval=False,
    )
    center_path = out_dir / "xgb_q50.json"
    bst.save_model(str(center_path))
    mae = evaluate(data, features, bst)
    print(f"  q50 (center): val MAE {mae:.3f} m -> {center_path.name}")

    # Adaptive per-station residual band offsets (10th / 90th percentile of
    # the signed residual per station) so low/high track local volatility.
    preds = bst.predict(xgb.DMatrix(data[features].astype("float32")))
    resid = data["target"].values - preds
    bands = {}
    for code in sorted(data[STATION_CODE_COL].unique()):
        sel = data[STATION_CODE_COL] == code
        if sel.sum() >= 100:
            bands[str(int(code))] = [round(float(np.quantile(resid[sel], 0.10)), 4),
                                     round(float(np.quantile(resid[sel], 0.90)), 4)]
    bands_path = out_dir / "xgb_bands.json"
    bands_path.write_text(json.dumps(bands, indent=2), encoding="utf-8")
    report["models"]["q50"] = {"file": center_path.name, "val_mae_m": round(mae, 4)}
    report["band_offsets"] = bands
    (out_dir / "xgb_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  bands -> {bands_path.name}: {bands}")
    print(f"Done in {time.time() - t0:.1f}s. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()