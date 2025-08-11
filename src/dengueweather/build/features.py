# src/dengueweather/build/features.py
"""
Feature engineering for unit-week panels.

- Merge (H3, iso_year, iso_week) panel with weekly weather.
- Optionally compute absolute humidity (AH) from Temp + RH.
- Create lagged weather features per H3 using config.features.lags_weeks.
"""

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from dengueweather.config import load_config
from ..logging_setup import setup_logging


def _absolute_humidity(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """
    We calcualte absolute humidity using the Clausius-Clapeyron equation
    AH (g/m^3) from T(°C) and RH(%):
      e_s (hPa) = 6.112 * exp(17.67*T/(T+243.5))
      e = RH * e_s / 100
      AH = 2.1674 * e / (273.15 + T)
    """
    T = temp_c.astype(float)
    es = 6.112 * np.exp(17.67 * T / (T + 243.5))
    e = (rh_pct.astype(float) / 100.0) * es
    ah = 2.1674 * e / (273.15 + T)
    return ah


def _detect_weather_cols(df: pd.DataFrame) -> List[str]:
    """Return normal weather column names present in df."""
    candidates = [
        "rain_mm",
        "temp_c",
        "rh_pct",
        "wind_kmh",
        "abs_humidity_gm3",
    ]
    return [c for c in candidates if c in df.columns]


def _make_lags(
    df: pd.DataFrame,
    by: Iterable[str],
    cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Create lagged copies of `cols` within groups defined by `by`."""
    df = df.sort_values(["iso_year", "iso_week"]).copy()
    g = df.groupby(list(by), sort=False)
    for col in cols:
        for L in lags:
            df[f"{col}_lag_{L}"] = g[col].shift(L)
    return df


def build_features(
    cluster_week_path: str,
    weather_path: str,
    out: str,
    lags: Optional[List[int]] = None,
) -> None:
    """
    Construct unit-week feature set.

    Parameters
    ----------
    cluster_week_path : Parquet with (h3, iso_year, iso_week, y_cluster_present, ...)
    weather_path      : Parquet with weekly weather (iso_year, iso_week, rain/temp/RH/wind/AH)
    out               : Output Parquet path
    lags              : Optional list of lag weeks; if None, use config.features.lags_weeks
    """
    setup_logging()
    cfg = load_config()
    if lags is None:
        lags = list(cfg.features.lags_weeks)

    # 1) Load inputs
    panel = pd.read_parquet(cluster_week_path)
    weather = pd.read_parquet(weather_path)
    for k in ["iso_year", "iso_week"]:
        if k not in panel.columns or k not in weather.columns:
            raise ValueError(f"Missing key column '{k}' in panel or weather.")

    # 2) Compute absolute humidity if requested and missing
    if cfg.features.use_absolute_humidity:
        if "abs_humidity_gm3" not in weather.columns:
            if {"temp_c", "rh_pct"}.issubset(weather.columns):
                weather["abs_humidity_gm3"] = _absolute_humidity(weather["temp_c"], weather["rh_pct"])
            else:
                # It's fine to continue without AH if inputs aren't there
                pass
    merged = panel.merge(weather, on=["iso_year", "iso_week"], how="left")

    # Keep label from panel (do NOT overwrite)
    if "y_cluster_present" not in merged.columns:
        # If for some reason your panel lacks it, derive from original (shouldn't happen)
        merged["y_cluster_present"] = 0

    # 4) Add time helpers (optional but handy)
    merged["week_of_year"] = merged["iso_week"].astype(int)
    # cyclical encoding helps models capture seasonality
    merged["woy_sin"] = np.sin(2 * np.pi * merged["week_of_year"] / 52.0)
    merged["woy_cos"] = np.cos(2 * np.pi * merged["week_of_year"] / 52.0)

    # 5) Build lagged features per H3
    weather_cols = _detect_weather_cols(merged)
    if weather_cols:
        merged = _make_lags(merged, by=["h3"], cols=weather_cols, lags=lags)

    # 6) Save
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(
        f"features: wrote {out} "
        f"(rows={len(merged)}, weather_cols={weather_cols}, lags={lags})"
    )
