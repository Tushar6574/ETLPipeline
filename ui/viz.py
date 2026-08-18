"""Plotting + map helpers for the dashboard (plotly / folium)."""

from __future__ import annotations

from typing import Optional

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go

STATUS_COLORS = {
    "HIGH_FLOOD_EXCEEDED": "#d62728",
    "DANGER": "#ff7f0e",
    "WARNING": "#f7c33d",
    "NORMAL": "#2ca02c",
}


def status_color(status: str) -> str:
    s = str(status).strip().upper()
    return STATUS_COLORS.get(s, "#7f7f7f")


def plot_level_history(
    df: pd.DataFrame,
    station_code: str,
    station: dict,
    window_days: int = 60,
    title: Optional[str] = None,
) -> go.Figure:
    """Past water level line with flood-threshold bands."""
    sub = df[df["station_code"] == station_code].sort_values("timestamp")
    if sub.empty:
        return go.Figure()
    if window_days:
        cut = sub["timestamp"].max() - pd.Timedelta(days=window_days)
        sub = sub[sub["timestamp"] >= cut]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["timestamp"],
            y=sub["water_level"],
            mode="lines+markers",
            name="Water level",
            line=dict(color="#1f77b4", width=1.5),
            marker=dict(size=4),
        )
    )
    for label, key, color in (
        ("Warning", "warning_level", "#f7c33d"),
        ("Danger", "danger_level", "#ff7f0e"),
        ("High flood", "high_flood_level", "#d62728"),
    ):
        value = station.get(key)
        if pd.notna(value):
            fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=label)
    fig.update_layout(
        title=title or f"{station.get('station_name', station_code)} - water level",
        xaxis_title="",
        yaxis_title="Level (m)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_forecast_fan(
    result,
    history: pd.Series,
    station: dict,
    window_hours: int = 24 * 14,
) -> go.Figure:
    """Observed tail + forecast mean with 10-90% band and threshold lines."""
    fig = go.Figure()
    if history is not None and len(history):
        tail = history.iloc[-window_hours:]
        fig.add_trace(
            go.Scatter(
                x=tail.index,
                y=tail.values,
                mode="lines",
                name="Observed",
                line=dict(color="#1f77b4", width=1.5),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.high,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.low,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.20)",
            name="10-90% band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.mean,
            mode="lines+markers",
            name=f"Forecast ({result.engine})",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=4),
        )
    )
    for label, key, color in (
        ("Warning", "warning_level", "#f7c33d"),
        ("Danger", "danger_level", "#ff7f0e"),
        ("High flood", "high_flood_level", "#d62728"),
    ):
        value = station.get(key)
        if pd.notna(value):
            fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=label)
    fig.update_layout(
        title=f"{station.get('station_name', '')} - forecast fan chart",
        xaxis_title="",
        yaxis_title="Level (m)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_risk_map(
    stations: pd.DataFrame,
    forecasts: Optional[dict] = None,
    meta_map: Optional[dict] = None,
) -> folium.Map:
    """Folium map: alert-coloured markers + risk-scaled buffers.

    ``stations`` is the per-station latest-reading frame. ``forecasts`` maps
    station_code -> ``ForecastResult`` to colour the buffer by risk score.
    """
    meta_map = meta_map or {}
    if stations.empty:
        return folium.Map(location=[21.0, 79.0], zoom_start=5)
    center = [float(stations["latitude"].mean()), float(stations["longitude"].mean())]
    m = folium.Map(location=center, zoom_start=6, control_scale=True)
    for _, row in stations.iterrows():
        code = row["station_code"]
        meta = meta_map.get(code, {})
        lat, lon = float(row["latitude"]), float(row["longitude"])
        status = str(row.get("flood_status", "NORMAL") or "NORMAL")
        color = status_color(status)
        risk = np.nan
        if forecasts and code in forecasts:
            risk = float(forecasts[code].get("risk_score", np.nan))
        popup = folium.Popup(
            _station_popup_html(code, row, meta, status, risk),
            max_width=320,
        )
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            tooltip=f"{meta.get('station_name', code)} ({status})",
            icon=folium.Icon(color="darkred" if color == "#d62728" else "orange", icon="tint"),
        ).add_to(m)
        if np.isfinite(risk):
            radius = 5_000 + risk * 55_000
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color=color,
                weight=1,
                fill=True,
                fill_opacity=0.12 + 0.25 * (risk / 100.0),
            ).add_to(m)
    return m


def plot_status_bars(latest: pd.DataFrame) -> go.Figure:
    """Per-status station count bar chart (current latest readings)."""
    if "flood_status" not in latest or latest.empty:
        return go.Figure()
    counts = latest["flood_status"].value_counts().sort_index()
    if counts.empty:
        return go.Figure()
    colors = [status_color(s) for s in counts.index]
    fig = go.Figure(
        go.Bar(x=list(counts.index), y=counts.values, marker_color=colors, text=counts.values)
    )
    fig.update_layout(
        title="Current status - stations",
        xaxis_title="",
        yaxis_title="Stations",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


def _station_popup_html(code: str, row: pd.Series, meta: dict, status: str, risk: float) -> str:
    risk_txt = f"{risk:.0f} / 100" if np.isfinite(risk) else "n/a"
    ts = row.get("timestamp")
    ts_txt = str(ts) if pd.notna(ts) else "n/a"
    return (
        "<b>"
        + meta.get("station_name", code)
        + f"</b><br>Basin: {meta.get('basin_name', 'n/a')}"
        + f"<br>Level: {row.get('water_level', float('nan')):.2f} m"
        + f"<br>Status: <b>{status}</b>"
        + f"<br>Risk (forecast): {risk_txt}"
        + f"<br>Updated: {ts_txt}"
    )


__all__ = [
    "STATUS_COLORS",
    "status_color",
    "plot_level_history",
    "plot_forecast_fan",
    "plot_status_bars",
    "build_risk_map",
]