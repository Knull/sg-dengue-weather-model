"""Feature engineering for unit-week panels (Updated for v2.0)."""

from pathlib import Path
from typing import Iterable, List, Optional

import h3
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def _compute_spatial_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 'spatial_case_lag_1': The number of active clusters in neighboring cells 
    during the previous week. This proxies 'infection pressure'.
    """
    print("Computing spatial lags (this might take a moment)...")
    
    # 1. Create a quick lookup for (iso_year, iso_week, h3) -> present (1/0)
    # Ensure we sort by time to handle temporal shifting correctly
    df = df.sort_values(["iso_year", "iso_week", "h3"]).reset_index(drop=True)
    
    # We need the previous week's cluster status. 
    # Group by H3 and shift 'y_cluster_present' by 1 week to get "self_lag_1"
    df["self_lag_1"] = df.groupby("h3")["y_cluster_present"].shift(1).fillna(0)
    
    # 2. Pivot to wide format: Index=Time, Columns=H3, Values=Self_Lag_1
    # This matrix tells us where clusters *were* last week.
    grid = df.pivot_table(
        index=["iso_year", "iso_week"], 
        columns="h3", 
        values="self_lag_1", 
        fill_value=0
    )
    
    # 3. For every H3 cell, find neighbors and sum their values from the grid
    # This is vectorizable but doing it column-wise is easier to read/maintain.
    # Pre-compute neighbor map
    all_h3 = list(grid.columns)
    neighbor_map = {
        cell: h3.k_ring(cell, 1) # Ring 1 includes self and immediate neighbors
        for cell in all_h3
    }
    
    # Create the result matrix (Infection Pressure)
    # We want: Sum of neighbors' lag_1
    spatial_pressure = pd.DataFrame(index=grid.index, columns=grid.columns)
    
    # This loop is faster than row-wise apply for large T
    for cell in tqdm(all_h3, desc="Spatial Convolution"):
        neighbors = [n for n in neighbor_map[cell] if n in grid.columns]
        if not neighbors:
            spatial_pressure[cell] = 0
            continue
        # Sum active clusters in neighborhood (excluding self if you prefer, but standard is inclusive)
        # Here we do inclusive (self + neighbors) as "local area pressure"
        spatial_pressure[cell] = grid[neighbors].sum(axis=1)
        
    # 4. Melt back to long format to merge
    spatial_long = spatial_pressure.reset_index().melt(
        id_vars=["iso_year", "iso_week"], 
        var_name="h3", 
        value_name="neighbor_pressure_lag_1"
    )
    
    return df.merge(spatial_long, on=["iso_year", "iso_week", "h3"], how="left")

def build_features(
    cluster_week_path: str,
    weather_path: str,
    out: str,
    lags: Optional[List[int]] = None,
) -> None:
    setup_logging()
    cfg = load_config()
    if lags is None:
        lags = list(cfg.features.get("lags_weeks", [0, 1, 2, 3]))

    cw = pd.read_parquet(cluster_week_path)
    weather = pd.read_parquet(weather_path)

    # ... [Merge logic same as before] ...
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
        panel["y_cluster_present"] = np.nan

    panel["y_cluster_present"] = panel["y_cluster_present"].fillna(0).astype(int)

    panel = _compute_spatial_lags(panel)
    # -------------------------------------

    merged = panel.merge(weather, on=["iso_year", "iso_week"], how="left")
    if cfg.features.get("use_absolute_humidity", True) and "abs_humidity_gm3" not in merged.columns:
        if {"temp_c", "rh_pct"}.issubset(merged.columns):
            merged["abs_humidity_gm3"] = _absolute_humidity(merged["temp_c"], merged["rh_pct"])

    merged["week_of_year"] = merged["iso_week"].astype(int)
    merged["woy_sin"] = np.sin(2 * np.pi * merged["week_of_year"] / 52.0)
    merged["woy_cos"] = np.cos(2 * np.pi * merged["week_of_year"] / 52.0)

    weather_cols = _detect_weather_cols(merged)
    if weather_cols:
        merged = _make_lags(merged, by=["h3"], cols=weather_cols, lags=lags)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"Features (v2) wrote to {out} with Spatial Lags.")