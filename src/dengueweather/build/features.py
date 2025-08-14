# src/dengueweather/build/features.py
"""Feature engineering for unit-week panels."""

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from dengueweather.config import load_config
from ..logging_setup import setup_logging


def _absolute_humidity(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    T = temp_c.astype(float)
    es = 6.112 * np.exp(17.67 * T / (T + 243.5))
    e = (rh_pct.astype(float) / 100.0) * es
    return 2.1674 * e / (273.15 + T)


def _detect_weather_cols(df: pd.DataFrame) -> List[str]:
    candidates = ["rain_mm", "temp_c", "rh_pct", "wind_kmh", "abs_humidity_gm3"]
    return [c for c in candidates if c in df.columns]


def _make_lags(df: pd.DataFrame, by: Iterable[str], cols: List[str], lags: List[int]) -> pd.DataFrame:
    df = df.sort_values(list(by) + ["iso_year", "iso_week"]).copy()
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
    setup_logging()
    cfg = load_config()
    if lags is None:
        # cfg.features is a dict in this project
        lags = list(cfg.features.get("lags_weeks", [0, 1, 2, 3]))

    cw = pd.read_parquet(cluster_week_path)  # expected: h3, iso_year, iso_week, y_cluster_present==1 per cluster-week rows
    weather = pd.read_parquet(weather_path)

    for k in ["iso_year", "iso_week"]:
        if k not in weather.columns:
            raise ValueError(f"Missing key column '{k}' in weather parquet.")

    h3_df = pd.DataFrame({"h3": sorted(cw["h3"].dropna().unique())})
    weeks = weather[["iso_year", "iso_week"]].drop_duplicates().sort_values(["iso_year", "iso_week"])
    h3_df["__key"] = 1
    weeks["__key"] = 1
    panel = h3_df.merge(weeks, on="__key").drop(columns="__key")

    if {"h3", "iso_year", "iso_week"}.issubset(cw.columns):
        pres = (
            cw.groupby(["h3", "iso_year", "iso_week"])
              .size()
              .rename("y_cluster_present")
              .reset_index()
        )
        pres["y_cluster_present"] = 1
        panel = panel.merge(pres, on=["h3", "iso_year", "iso_week"], how="left")
    else:
        panel["y_cluster_present"] = np.nan  # safety

    panel["y_cluster_present"] = panel["y_cluster_present"].fillna(0).astype(int)

    merged = panel.merge(weather, on=["iso_year", "iso_week"], how="left")
    if cfg.features.get("use_absolute_humidity", True) and "abs_humidity_gm3" not in merged.columns:
        if {"temp_c", "rh_pct"}.issubset(merged.columns):
            merged["abs_humidity_gm3"] = _absolute_humidity(merged["temp_c"], merged["rh_pct"])

    #Seasonality helpers
    merged["week_of_year"] = merged["iso_week"].astype(int)
    merged["woy_sin"] = np.sin(2 * np.pi * merged["week_of_year"] / 52.0)
    merged["woy_cos"] = np.cos(2 * np.pi * merged["week_of_year"] / 52.0)

    # Lagged features per H3
    weather_cols = _detect_weather_cols(merged)
    if weather_cols:
        merged = _make_lags(merged, by=["h3"], cols=weather_cols, lags=lags)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(
        f"features: wrote {out} "
        f"(rows={len(merged):,}, uniq_h3={merged['h3'].nunique()}, "
        f"weeks={weeks.shape[0]}, weather_cols={weather_cols}, lags={lags})"
    )
