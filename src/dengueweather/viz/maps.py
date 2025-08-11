"""Generate risk maps aggregated to URA subzones.

This module defines a function to produce a choropleth map of predicted
dengue cluster probabilities for a given week and year. 
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import geopandas as gpd  # type: ignore[import]
import h3
import joblib  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from shapely.geometry import Polygon  # type: ignore[import]

from dengueweather.config import load_config
from dengueweather.logging_setup import setup_logging


def map_hindcast(
    model_path: str,
    out_dir: str,
    *,
    features_path: Optional[str] = None,
    ura_path: Optional[str] = None,
    iso_year: Optional[int] = None,
    iso_week: Optional[int] = None,
    agg_method: str = "mean",
) -> None:
    setup_logging()
    cfg = load_config()
    if features_path is None:
        processed_dir = cfg.data.get("processed_dir", "data/processed")
        features_path = str(Path(processed_dir) / "unit_week_features.parquet")
    if ura_path is None:
        ura_path = cfg.data.get("ura_subzones_path", "data/external/ura_subzones.geojson")
    # Load feature table
    try:
        feats = pd.read_parquet(features_path)
    except Exception as exc:
        print(f"Failed to load features from {features_path}: {exc}")
        return
    if feats.empty:
        print("Features table is empty; cannot create map.")
        return
    # Determine week to visualise
    if iso_year is None or iso_week is None:
        latest = feats[["iso_year", "iso_week"]].dropna().sort_values(
            ["iso_year", "iso_week"], ascending=[True, True]
        ).iloc[-1]
        iso_year = int(latest["iso_year"])
        iso_week = int(latest["iso_week"])
    mask = (feats["iso_year"] == iso_year) & (feats["iso_week"] == iso_week)
    week_df = feats.loc[mask].copy()
    if week_df.empty:
        print(f"No records found for year {iso_year}, week {iso_week}")
        return
    try:
        model_dict = joblib.load(model_path)
        model = model_dict.get("model")
        feature_cols = model_dict.get("feature_cols", [])
    except Exception as exc:
        print(f"Failed to load model from {model_path}: {exc}")
        return
    if model is None or not feature_cols:
        print(f"Model or feature columns missing in {model_path}")
        return
    # Construct feature matrix X for the week
    X = week_df[feature_cols].fillna(0.0)
    try:
        preds = model.predict_proba(X)[:, 1]
    except Exception as exc:
        print(f"Error computing predicted probabilities: {exc}")
        return
    week_df = week_df.assign(pred_proba=preds)
    # Convert H3 indices to polygons
    # Build GeoDataFrame with geometry
    def h3_to_polygon(h: str) -> Polygon:
        boundary = h3.h3_to_geo_boundary(h, geo_json=True)
        # boundary is list of [lat, lon]; shapely expects (lon, lat)
        return Polygon([(lon, lat) for lat, lon in boundary])
    week_df = week_df.copy()
    week_df["geometry"] = week_df["h3"].apply(h3_to_polygon)
    h3_gdf = gpd.GeoDataFrame(week_df, geometry="geometry", crs="EPSG:4326")
    try:
        ura_gdf = gpd.read_file(ura_path)
    except Exception as exc:
        print(f"Failed to load URA subzones from {ura_path}: {exc}")
        return
    # Ensure CRS matches; assume URA file is WGS84 or convert
    if ura_gdf.crs is None:
        ura_gdf = ura_gdf.set_crs("EPSG:4326")
    elif str(ura_gdf.crs) != "EPSG:4326":
        ura_gdf = ura_gdf.to_crs("EPSG:4326")
    # Find the subzone name column
    subzone_col = None
    for col in ura_gdf.columns:
        if "subzone" in col.lower() and not col.lower().startswith("subzone_no"):  # skip codes
            subzone_col = col
            break
    if subzone_col is None:
        print("Could not identify subzone name column in URA GeoJSON.")
        return
    # Spatial join: assign each H3 cell to a subzone
    # Use intersect join (inner).  This may produce multiple matches if boundaries overlap; drop duplicates.
    try:
        joined = gpd.sjoin(h3_gdf[["pred_proba", "geometry"]], ura_gdf[[subzone_col, "geometry"]], how="inner", predicate="intersects")
    except Exception as exc:
        print(f"Spatial join failed: {exc}")
        return
    # Aggregate by subzone
    if joined.empty:
        print("No H3 cells joined to any subzones; check geometry alignment.")
        return
    if agg_method == "sum":
        agg_series = joined.groupby(subzone_col)["pred_proba"].sum()
    elif agg_method == "max":
        agg_series = joined.groupby(subzone_col)["pred_proba"].max()
    else:
        # default to mean
        agg_series = joined.groupby(subzone_col)["pred_proba"].mean()
    # Merge aggregated probabilities back onto URA polygons
    ura_gdf = ura_gdf.merge(agg_series.rename("risk"), on=subzone_col, how="left")
    # Plot the map
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot URA polygons colored by risk; subzones with NaN risk are plotted in grey
    ura_gdf.plot(
        column="risk",
        cmap="OrRd",
        linewidth=0.5,
        edgecolor="white",
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(
        f"Predicted dengue cluster risk\nISO Year {iso_year}, Week {iso_week}",
        fontsize=12,
    )
    # Save figure
    outfile = out_dir_path / f"hindcast_map_{iso_year}_{iso_week}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Hindcast map saved to {outfile}")
