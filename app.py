"""CWC River Water Level Dashboard.

Visualises the committed ETL output (``output/``), floods NWDP history, and
forecasts flood-prone zones with Google TimesFM 2.5 (XGBoost fallback).

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium
except Exception:  # noqa: BLE001 - keep the app usable if the component is missing
    import streamlit.components.v1 as _components

    def st_folium(m, width="100%", height=480, **kwargs):
        _components.html(m.get_root()._repr_html_(), height=height, scrolling=True)

from config import DEFAULT_STATIONS
from ui.data import default_paths, load_dataset, latest_reading, refresh_dataset, station_meta_map
from ui.features import prepare_series
from ui.forecast import (
    ForecastResult,
    forecast_station,
    hours_to_threshold,
    risk_score,
    station_catalogue,
    timesfm_available,
    xgb_trained_stations,
)
from ui.viz import build_risk_map, plot_forecast_fan, plot_level_history, plot_status_bars

st.set_page_config(page_title="CWC River Water Levels", page_icon="\U0001F30A", layout="wide")

HORIZON_MIN, HORIZON_MAX, HORIZON_DEFAULT = 24, 168, 72
HORIZON_STEP = 24


# ---------------------------------------------------------------------------
# Resources / caches
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _timesfm_model_resource():
    from ui.forecast import _load_timesfm

    return _load_timesfm()


@st.cache_data(show_spinner=False, ttl=900)
def _forecast_cached(
    df: pd.DataFrame, station_code: str, horizon: int, engine: str
) -> ForecastResult:
    model = _timesfm_model_resource() if engine == "timesfm" else None
    return forecast_station(
        df, station_code, horizon=horizon, engine=engine, timesfm_model=model
    )


@st.cache_data(show_spinner=False, ttl=900)
def _overview_risks(df: pd.DataFrame, horizon: int) -> dict:
    """Fast XGBoost risk snapshot for every trained station (Overview tab)."""
    out = {}
    catalogue = station_catalogue()
    trained = xgb_trained_stations()
    for code in sorted(df["station_code"].dropna().unique()):
        if code not in trained:
            continue
        try:
            res = forecast_station(df, code, horizon=horizon, engine="xgboost")
            station = catalogue.get(code)
            danger = station.danger_level if station else np.nan
            score = risk_score(res.last_level, res.mean, float(danger))
            hours = hours_to_threshold(res.mean, float(danger)) if np.isfinite(danger) else None
            out[code] = {
                "risk_score": score["score"],
                "proximity": score["proximity"],
                "crossing": score["crossing"],
                "surge": score["surge"],
                "hours_to_danger": hours,
            }
        except Exception:  # noqa: BLE001 - one bad station must not kill the tab
            continue
    return out


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("\U0001F30A CWC River Levels")
    st.caption("NWDP / CWC · ETL pipeline + TimesFM 2.5")
    engine = st.radio(
        "Forecast engine",
        ["xgboost", "timesfm"],
        index=0,
        help=(
            "XGBoost (default - fast, offline) or TimesFM 2.5 (Google, "
            "zero-shot, ~800 MB checkpoint downloaded on first use)."
        ),
    )
    horizon = st.slider(
        "Forecast horizon", HORIZON_MIN, HORIZON_MAX, HORIZON_DEFAULT, step=HORIZON_STEP
    )
    if not timesfm_available():
        st.warning("`timesfm` is not installed - using XGBoost.")
    if st.button("\U0001F504 Refresh now", width="stretch"):
        try:
            with st.spinner("Running incremental ETL against NWDP..."):
                df_ref, meta_ref = refresh_dataset()
            st.session_state["df"] = df_ref
            st.session_state["meta"] = meta_ref
            st.success("Dataset refreshed from NWDP.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Refresh failed: {exc}")
    st.divider()
    st.caption("Deployed on Streamlit Cloud the container is ephemeral - "
               "Refresh now re-runs the ETL for this session only.")


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state["df"], st.session_state["meta"] = load_dataset(*map(str, default_paths()))
df: pd.DataFrame = st.session_state["df"]
meta: dict = st.session_state["meta"]
meta_map = station_meta_map(meta)

if df.empty:
    st.error("No data found in output/. Run the ETL first: `python main.py`.")
    st.stop()

station_codes = sorted(df["station_code"].dropna().unique())
catalogue = station_catalogue()
named_stations = [
    code for code in station_codes if code in catalogue
] or station_codes
latest = latest_reading(df)


def station_label(code: str) -> str:
    s = catalogue.get(code)
    return f"{s.station_name} ({code})" if s else code


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_explorer, tab_forecast, tab_quality = st.tabs(
    ["Overview", "Water Level Explorer", "Flood Forecast", "Data & Quality"]
)

# ---- Overview -------------------------------------------------------------
with tab_overview:
    with st.spinner("Computing risk snapshot (XGBoost)..."):
        risks = _overview_risks(df, horizon)

    latest = latest_reading(df)
    non_normal = latest[latest["flood_status"].ne("NORMAL")] if "flood_status" in latest else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations", len(latest))
    c2.metric("Records", f"{len(df):,}")
    c3.metric("Last update", pd.to_datetime(df["timestamp"]).max().strftime("%d %b %H:%M"))
    c4.metric("Not normal", len(non_normal))

    map_col, risk_col = st.columns([3, 2], gap="large")
    with map_col:
        m = build_risk_map(latest, risks, meta_map)
        st_folium(m, width="100%", height=480)
    with risk_col:
        st.subheader("Risk ranking (next 72 h)")
        rows = [
            {
                "Station": station_label(code),
                "Risk": int(r["risk_score"]),
                "Hours to danger": r["hours_to_danger"] if r["hours_to_danger"] is not None else "\u221E",
            }
            for code, r in sorted(risks.items(), key=lambda kv: -kv[1]["risk_score"])
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True
            )
            max_risk = max(r["risk_score"] for r in risks.values())
            st.progress(int(min(max_risk, 100)))
            st.caption("Risk 0 = safely below danger with no forecasted rise. "
                       "Blends threshold proximity, forecast crossing and surge "
                       "slope (heuristic). Buffers scale with risk.")
        else:
            st.info("No risk scores computed - check the training models.")
        skipped = sorted(
            set(df["station_code"].dropna().unique()) - set(risks)
        )
        if skipped:
            st.caption(
                f"No XGBoost model for {', '.join(skipped)} - use the TimesFM "
                "engine in the Flood Forecast tab."
            )

    st.divider()
    st.subheader("Current status distribution")
    st.plotly_chart(plot_status_bars(latest), width="stretch")

# ---- Explorer -------------------------------------------------------------
with tab_explorer:
    st.subheader("Historical water levels")
    exp_col1, exp_col2 = st.columns([2, 3])
    with exp_col1:
        code = st.selectbox("Station", named_stations, format_func=station_label, key="explorer_station")
        window_days = st.slider("Look-back (days)", 7, 366, 90, key="explorer_window")
    with exp_col2:
        s = catalogue.get(code)
        if s:
            st.metric("Current level", f"{float(latest[latest['station_code'] == code]['water_level'].iloc[0]):.2f} m")
            st.caption(
                f"Warning {s.warning_level} m · Danger {s.danger_level} m · "
                f"High flood {s.high_flood_level} m · Basin: {s.basin_name}"
            )
    st.plotly_chart(
        plot_level_history(df, code, meta_map.get(code, {"station_name": code}), window_days),
        width="stretch",
    )
    recent = df[df["station_code"] == code].sort_values("timestamp").tail(10)
    st.dataframe(
        recent[["timestamp", "water_level", "flood_status"]].rename(
            columns={"water_level": "Level (m)"}
        ),
        width="stretch",
        hide_index=True,
    )

# ---- Forecast -------------------------------------------------------------
with tab_forecast:
    st.subheader("Flood forecast")
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1:
        code = st.selectbox("Station", named_stations, format_func=station_label, key="forecast_station")
    with f_col2:
        st.metric("Horizon", f"{horizon} h")
    with f_col3:
        st.metric("Engine", engine)

    try:
        with st.spinner(f"Running {engine} forecast..."):
            res = _forecast_cached(df, code, horizon, engine)
        station = catalogue.get(code)
        station_info = {
            "station_name": station.station_name if station else code,
            "warning_level": station.warning_level if station else np.nan,
            "danger_level": station.danger_level if station else np.nan,
            "high_flood_level": station.high_flood_level if station else np.nan,
        }
        danger = float(station_info["danger_level"])
        risk = risk_score(res.last_level, res.mean, danger)
        hours_danger = hours_to_threshold(res.mean, danger) if np.isfinite(danger) else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last observed", f"{res.last_level:.2f} m")
        m2.metric("Forecast peak", f"{res.mean.max():.2f} m")
        m3.metric("Risk score", f"{risk['score']:.0f} / 100")
        m4.metric(
            "Hours to danger",
            hours_danger if hours_danger is not None else "\u221E",
        )

        hist = prepare_series(df, code)
        st.plotly_chart(
            plot_forecast_fan(res, hist, station_info),
            width="stretch",
        )

        st.caption(
            f"{res.note}. Band = 10-90% quantiles (TimesFM col1/col9; "
            "XGBoost q05/q95 mirror-guarded). Time-to-threshold is the first "
            "hour the mean forecast crosses it. Risk 0 = safely below danger "
            "with no forecasted rise."
        )
        if np.isfinite(risk["proximity"]):
            st.write(
                f"Components - proximity: {risk['proximity']:.0f} · "
                f"crossing: {risk['crossing']:.0f} · surge: {risk['surge']:.0f}"
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Forecast failed for {code}: {exc}")

# ---- Data & Quality -------------------------------------------------------
with tab_quality:
    st.subheader("Dataset summary")
    st.write(f"Total records: **{len(df):,}** · Columns: {', '.join(df.columns)}")
    per_station = df.groupby("station_code").agg(
        records=("timestamp", "count"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
        min_level=("water_level", "min"),
        max_level=("water_level", "max"),
    ).round(2)
    st.dataframe(per_station, width="stretch")

    st.subheader("Known data-quality caveats")
    st.info(
        "NWDP telemetry contains occasional junk spikes (e.g. Musiri ~677 m, "
        "Mandla ~1394 m, Wadenepally ~790 m) that can look like "
        "HIGH_FLOOD_EXCEEDED. The dashboard clips these via a rolling-median "
        "MAD fence before forecasting. Thresholds are official CWC figures; "
        "two stations (PTHNGD, YMNCKT) use bracketed values pending "
        "verification."
    )
    if "dedup_key" in df.columns:
        dupes = int(df.duplicated(subset=["dedup_key"]).sum())
        st.write(f"Duplicate dedup_keys in this export: {dupes}.")
    st.caption(
        f"Source: NWDP CKAN (nwdp.nwic.gov.in). Pipeline: "
        f"`python main.py` hourly via GitHub Actions; UI: `streamlit run app.py`."
    )