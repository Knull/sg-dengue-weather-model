"""The command-line interface for this project"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import datetime # Added for weather download

import typer

from dengueweather.config import load_config
from dengueweather.ingest.sgcharts_archive import merge_archive
from dengueweather.ingest.mss_weather import weekly_from_dir
from dengueweather.ingest.moh_weekly_cases import load_weekly_counts, append_pdf_weeks
from dengueweather.ingest.nea_live_archiver import archive_today
from dengueweather.ingest.weather_client import MSSWeatherClient # Added import
from dengueweather.build.cluster_history import build_history, build_cluster_week
from dengueweather.build.features import build_features
from dengueweather.model.panel_logit import fit_model as fit_panel_logit
from dengueweather.model.panel_logit import eval_model as eval_panel_logit
from dengueweather.model.eval import year_block_cv, spatial_block_cv # Added spatial_block_cv
from dengueweather.viz.lag_curves import plot_lag_curves
from dengueweather.viz.interaction_surfaces import plot_interaction
from dengueweather.viz.maps import map_hindcast
from dengueweather.model.panel_gbm import fit_gbm, eval_gbm, run_cv  

app = typer.Typer(add_completion=False, help="Dengue-weather command-line interface")

# INGESTION COMMANDS 👇

@app.command("download-weather")
def download_weather(
    station_code: str = typer.Option("S24", help="MSS Station Code (e.g. S24 for Changi)"),
    start_year: int = typer.Option(2012, help="Start year"),
    end_year: Optional[int] = typer.Option(None, help="End year (default: current year)"),
    out_dir: str = typer.Option("data/raw/mss_weather/daily", help="Directory to save CSVs")
) -> None:
    """
    Download historical daily weather data using the engineered MSS client.
    
    This command fetches daily CSV records from the MSS archive, handling retries
    and session management automatically.
    """
    if end_year is None:
        end_year = datetime.date.today().year

    # Ensure output directory exists specific to station
    target_dir = Path(out_dir) / station_code
    
    typer.echo(f"Initializing Weather API Client for station {station_code} ({start_year}-{end_year})...")
    
    with MSSWeatherClient() as client:
        count = 0
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Don't fetch future months
                if datetime.date(year, month, 1) > datetime.date.today():
                    break
                
                try:
                    path = client.download_month(station_code, year, month, target_dir)
                    if path:
                        count += 1
                except Exception as e:
                    typer.secho(f"Failed to fetch {year}-{month}: {e}", fg=typer.colors.RED)
    
    typer.echo(f"Finished. Downloaded {count} files to {target_dir}")


@app.command("ingest-archive")
def ingest_archive(
    input_dir: str = typer.Argument(..., help="Directory containing SGCharts CSV snapshots"),
    out: str = typer.Argument(..., help="Path to write merged archive Parquet")
) -> None:
    """Merges the SGCharts archive CSVs into a single Parquet file.

    This command expects a directory tree containing the historical CSV files
    downloaded from https://outbreak.sgcharts.com/data/ (both the `csv` and
    `incorrect_latitude_longitude` subfolders). It concatenates all
    available snapshots, parses dates, filters out invalid coordinates and
    writes a single Parquet file for downstream processing.
    """
    merge_archive(input_dir, out)


@app.command("ingest-weather")
def ingest_weather(
    station_dir: str = typer.Option(
        ...,
        "--station-dir",
        help=(
            "Directory containing daily MSS CSVs.  For example, the Changi station "
            "files should live under `data/raw/mss_weather/CHANGI/daily`." # 
        ),
    ),
    out_daily: str = typer.Option(
        "data/interim/weather_daily.parquet",
        help="Path to write the concatenated daily Parquet file",
    ),
    out_weekly: str = typer.Option(
        "data/interim/weather_weekly.parquet",
        help="Path to write the aggregated weekly Parquet file",
    ),
    compute_ah: bool = typer.Option(
        True,
        help="Whether to compute absolute humidity from temperature and RH if available",
    ),
) -> None:
    """Aggregate MSS daily files into daily and weekly summaries.

    The `station_dir` should contain monthly CSVs as downloaded from the
    Meteorological Service Singapore (e.g., Changi station). The helper
    `weekly_from_dir` will normalise column names, compute absolute
    humidity (if requested) and aggregate to ISO week. Output files will be created with the specified names.
    """
    weekly_from_dir(station_dir, out_daily, out_weekly, compute_ah=compute_ah)


@app.command("ingest-moh")
def ingest_moh(
    csv_path: str = typer.Option(..., help="Path to the MOH weekly cases CSV (2012-2022)"),
    pdf_dir: Optional[str] = typer.Option(
        None,
        help=(
            "Directory containing MOH weekly bulletin PDFs for 2023 onwards.  "
            "If provided, the counts in these PDFs will be appended."
        ),
    ),
    out: str = typer.Option(
        "data/interim/moh_weekly.parquet",
        help="Path to write the combined weekly dengue counts Parquet",
    ),
) -> None:
    """Load and optionally extend the MOH weekly infectious disease bulletin.

    This command reads the national weekly infectious disease counts CSV
    (downloaded via the Data.gov.sg API), filters for dengue fever and
    normalises columns.  If a directory of PDF bulletins is provided, it
    attempts to parse each PDF for weekly counts and appends them to the
    CSV.  The resulting table (`iso_year`, `iso_week`, `cases`) is written to a Parquet file.
    """
    df = load_weekly_counts(csv_path)
    if pdf_dir:
        df = append_pdf_weeks(df, pdf_dir)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    typer.echo(f"Wrote {out} ({len(df)} rows)")


@app.command("ingest-nea-live")
def ingest_nea_live(
    outdir: str = typer.Option(
        "data/raw/nea_live",
        help="Directory into which to archive today's NEA clusters GeoJSON",
    ),
    dataset_id: Optional[str] = typer.Option(
        None,
        help="Override the Data.gov.sg dataset ID for live clusters",
    ),
    direct_url: Optional[str] = typer.Option(
        None,
        help="Download directly from this GeoJSON URL instead of polling the API",
    ),
) -> None:
    """Archive today's live NEA dengue clusters.

    This command fetches the current dengue clusters GeoJSON via the
    `archive_today` helper and stores it in `outdir` with a date- stamped filename.
    Use this command to build your own prospective cluster history going forward.
    If `direct_url` is specified, the download will bypass the Data.gov.sg API entirely.
    """
    path = archive_today(outdir, dataset_id=dataset_id, direct_url=direct_url)
    typer.echo(f"Saved {path}")


# BUILD COMMANDS

@app.command("build-history")
def build_cluster_history(
    merged_path: str = typer.Option(
        "data/interim/archive_merged.parquet",
        help="Path to the merged archive Parquet produced by ingest-archive",
    ),
    out_path: str = typer.Option(
        "data/interim/cluster_history.parquet",
        help="Path to write the cluster history Parquet file",
    ),
) -> None:
    """Construct stable cluster histories from merged snapshots."""
    build_history(merged_path, out_path)


@app.command("build-cluster-week")
def build_cluster_week_cmd(
    history_path: str = typer.Option(
        "data/interim/cluster_history.parquet",
        help="Path to the cluster history Parquet generated by build-history",
    ),
    out_path: str = typer.Option(
        "data/processed/cluster_week.parquet",
        help="Path to write the cluster-week labels Parquet",
    ),
) -> None:
    """Explode cluster histories into a cluster-week presence table."""
    build_cluster_week(history_path, out_path)


@app.command("build-features")
def build_features_cmd(
    cluster_week_path: str = typer.Option(
        "data/processed/cluster_week.parquet",
        help="Path to the cluster-week Parquet file from build-cluster-week",
    ),
    weather_path: str = typer.Option(
        "data/interim/weather_weekly.parquet",
        help="Path to the weekly weather Parquet from ingest-weather",
    ),
    out: str = typer.Option(
        "data/processed/unit_week_features.parquet",
        help="Path to write the unit-week feature set",
    ),
    lags: Optional[str] = typer.Option(
        None,
        help=(
            "Comma-separated list of lags (weeks) to compute.  If omitted, the "
            "lags configured in config.features.lags_weeks will be used."
        ),
    ),
) -> None:
    """Construct unit-week feature table by merging labels with weather and creating lags."""
    cfg = load_config()
    lag_list: Optional[list[int]] = None
    if lags:
        lag_list = [int(x.strip()) for x in lags.split(",") if x.strip().isdigit()]
    build_features(cluster_week_path, weather_path, out, lags=lag_list)


# MODELLING COMMANDS
@app.command("cv-gbm")
def cv_gbm_cmd(
    features_path: str = typer.Option("data/processed/unit_week_features.parquet"),
    out: str = typer.Option("data/processed/cv_results_gbm.csv"),
    n_splits: int = typer.Option(3, help="Number of time blocks to split")
) -> None:
    """Run a realistic Time-Block Cross-Validation on the GBM model."""
    run_cv(features_path, out, n_splits=n_splits)
    
@app.command("fit-panel-logit")
def fit_panel_logit_cmd(
    features_path: str = typer.Option(
        "data/processed/unit_week_features.parquet",
        help="Path to the feature table produced by build-features",
    ),
    out_path: str = typer.Option(
        "data/processed/model.joblib",
        help="Path to save the trained logistic regression model",
    ),
    class_weight: Optional[str] = typer.Option(
        None,
        help="Class weight spec passed to scikit-learn (e.g. 'balanced').",
    ),
    max_iter: int = typer.Option(
        1000,
        help="Maximum iterations for the logistic regression optimiser.",
    ),
) -> None:
    """Fit a baseline logistic regression on the unit-week features."""
    params = {"max_iter": max_iter}
    if class_weight:
        params["class_weight"] = class_weight
    fit_panel_logit(features_path, out_path, **params)


@app.command("eval-panel-logit")
def eval_panel_logit_cmd(
    model_path: str = typer.Option(
        "data/processed/model.joblib",
        help="Path to the trained logistic regression model",
    ),
    features_path: str = typer.Option(
        "data/processed/unit_week_features.parquet",
        help="Path to the feature table used for evaluation",
    ),
    out: str = typer.Option(
        "data/processed/metrics.json",
        help="Path to write evaluation metrics as JSON",
    ),
) -> None:
    """Evaluate a trained logistic regression model on the full dataset."""
    eval_panel_logit(model_path, features_path, out)


@app.command("cv-panel-logit")
def cv_panel_logit_cmd(
    features_path: str = typer.Option(
        "data/processed/unit_week_features.parquet",
        help="Path to the feature table for cross-validation",
    ),
    n_splits: int = typer.Option(
        3,
        help="Number of folds in cross-validation",
    ),
    out: str = typer.Option(
        "data/processed/cv_metrics.parquet",
        help="Path to write cross-validation metrics as Parquet",
    ),
    max_iter: int = typer.Option(
        1000,
        help="Maximum iterations for each fold's logistic regression.",
    ),
    class_weight: Optional[str] = typer.Option(
        None,
        help="Class weight spec passed to scikit-learn (e.g. 'balanced').",
    ),
    mode: str = typer.Option(
        "temporal",
        help="CV mode: 'temporal' (year-block) or 'spatial' (block/subzone)"
    )
) -> None:
    """Perform cross-validation on the panel logistic model."""
    params = {"max_iter": max_iter, "class_weight": class_weight}
    
    if mode == "spatial":
        # Note: requires a 'subzone_id' or similar column, or falls back to H3 parents
        df_cv = spatial_block_cv(features_path, n_splits=n_splits, model_params=params)
    else:
        df_cv = year_block_cv(features_path, n_splits=n_splits, model_params=params)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_cv.to_parquet(out, index=False)
    typer.echo(f"Wrote {mode} CV metrics to {out}")


# VISUALIZATION COMMANDS

@app.command("rank-riskiest")
def rank_riskiest(
    model_path: str = typer.Option("data/processed/model.joblib", help="Trained model path"),
    features_path: str = typer.Option("data/processed/unit_week_features.parquet", help="Features path"),
    iso_year: int = typer.Option(..., help="Year to rank"),
    iso_week: int = typer.Option(..., help="Week to rank"),
    top_k: int = typer.Option(20, help="Number of top zones to display"),
    out: Optional[str] = typer.Option(None, help="Optional path to save CSV results")
) -> None:
    """
    Generate a ranked list of the 'Top-K' highest risk zones for a specific week.
    
    This command uses the calibrated probabilities to rank all Singapore subzones/hexagons.
    """
    import pandas as pd
    import joblib
    
    # Load Model
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    # Handle models with or without scalers (LightGBM doesn't use one)
    scaler = payload.get("scaler") 
    
    # Load Data
    df = pd.read_parquet(features_path)
    subset = df[(df["iso_year"] == iso_year) & (df["iso_week"] == iso_week)].copy()
    
    if subset.empty:
        typer.secho(f"No data found for {iso_year} week {iso_week}", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    # Predict
    # Ensure columns match exactly what the model was trained on
    X = subset[feature_cols].fillna(0.0).astype("float64").values
    
    # Apply scaling only if the model requires it (e.g. Logistic Regression)
    if scaler:
        X = scaler.transform(X)
        
    probs = model.predict_proba(X)[:, 1]
    
    subset["risk_score"] = probs
    
    # Rank
    ranked = subset.sort_values("risk_score", ascending=False).head(top_k)
    
    # Display
    columns_to_show = ["h3", "risk_score", "y_cluster_present"]
    # Add coordinates if possible
    if "h3" in ranked.columns:
        import h3
        ranked["lat"] = ranked["h3"].apply(lambda x: h3.h3_to_geo(x)[0])
        ranked["lon"] = ranked["h3"].apply(lambda x: h3.h3_to_geo(x)[1])
        columns_to_show += ["lat", "lon"]
        
    typer.echo(f"\nTop {top_k} Riskiest Zones for {iso_year}-W{iso_week:02d}:")
    typer.echo(ranked[columns_to_show].to_markdown(index=False))
    
    if out:
        ranked.to_csv(out, index=False)
        typer.echo(f"Saved to {out}")    

@app.command("plot-lag-curves")
def plot_lag_curves_cmd(
    model_path: str = typer.Option(
        "data/processed/model.joblib",
        help="Path to the trained logistic regression model",
    ),
    out_dir: str = typer.Option(
        "docs/figures_lag",
        help="Directory to write lag-response curve figures",
    ),
) -> None:
    """Generate lag-response curve figures using the trained model."""
    plot_lag_curves(model_path, out_dir)


@app.command("plot-interaction")
def plot_interaction_cmd(
    model_path: str = typer.Option(
        "data/processed/model.joblib",
        help="Path to the trained logistic regression model",
    ),
    out_dir: str = typer.Option(
        "docs/figures_interaction",
        help="Directory to write rainfall-temperature interaction figures",
    ),
) -> None:
    """Generate rainfall-temperature interaction surface plots."""
    plot_interaction(model_path, out_dir)


@app.command("map-hindcast")
def map_hindcast_cmd(
    model_path: str = typer.Option(
        "data/processed/model.joblib",
        help="Path to the trained logistic regression model",
    ),
    features_path: Optional[str] = typer.Option(
        None,
        help=(
            "Path to the feature table.  If omitted, the path from the config file "
            "(data.processed_dir/unit_week_features.parquet) will be used."
        ),
    ),
    out_dir: str = typer.Option(
        "docs/maps",
        help="Directory to write hindcast risk maps",
    ),
    iso_year: Optional[int] = typer.Option(
        None,
        help="ISO year of the week to map.  Defaults to the latest available year.",
    ),
    iso_week: Optional[int] = typer.Option(
        None,
        help="ISO week number to map.  Defaults to the latest available week.",
    ),
) -> None:
    """Generate a hindcast risk map for a given week aggregated to URA subzones."""
    cfg = load_config()
    feat_path = features_path or str(Path(cfg.data["processed_dir"]) / "unit_week_features.parquet")
    map_hindcast(model_path, out_dir, features_path=feat_path, iso_year=iso_year, iso_week=iso_week)

@app.command("fit-gbm")
def fit_gbm_cmd(
    features_path: str = typer.Option(
        "data/processed/unit_week_features.parquet",
        help="Path to feature table (ensure it has spatial lags!)"
    ),
    out_path: str = typer.Option(
        "data/processed/model_gbm.joblib",
        help="Path to save the LightGBM model"
    ),
    n_estimators: int = 500,
) -> None:
    """
    Train the advanced Gradient Boosting Machine (LightGBM) model.
    This model uses spatial lags and non-linear weather interactions.
    """
    fit_gbm(features_path, out_path, n_estimators=n_estimators)

@app.command("eval-gbm")
def eval_gbm_cmd(
    model_path: str = typer.Option("data/processed/model_gbm.joblib"),
    features_path: str = typer.Option("data/processed/unit_week_features.parquet"),
    out: str = typer.Option("data/processed/metrics_gbm.json")
) -> None:
    """Evaluate the GBM model."""
    eval_gbm(model_path, features_path, out)


@app.command("patch-live-week")
def patch_live_week(
    live_geojson: str = typer.Option(..., help="Path to the live NEA GeoJSON file"),
    features_path: str = typer.Option("data/processed/unit_week_features.parquet"),
    out: str = typer.Option("data/processed/unit_week_features.parquet"),
) -> None:
    """
    Tactical Command: Merges Live NEA Clusters + Latest Weather into the feature set.
    Includes robust coordinate handling and centroid fallback for small polygons.
    """
    import json
    import pandas as pd
    import h3
    import datetime
    from shapely.geometry import shape, mapping
    from shapely.ops import transform
    
    # 1. Load Live Clusters
    typer.echo(f"Loading live clusters from {live_geojson}...")
    with open(live_geojson, "r") as f:
        data = json.load(f)
    
    df_feat = pd.read_parquet(features_path)
    if df_feat.empty:
        raise typer.Exit(1)
        
    sample_h3 = df_feat["h3"].iloc[0]
    res = h3.h3_get_resolution(sample_h3)
    typer.echo(f"Using H3 resolution: {res}")
    
    live_h3s = set()
    features_list = data.get("features", [])
    if not features_list:
        if data.get("type") in ["Polygon", "MultiPolygon", "GeometryCollection"]:
            features_list = [{"geometry": data}]

    typer.echo(f"Processing {len(features_list)} features...")

    for i, feature in enumerate(features_list):
        geom = feature.get("geometry", feature)
        if not geom or "coordinates" not in geom:
            continue
            
        try:
            shp = shape(geom)
            
            # Normalize to list of Polygons
            polys = []
            if shp.geom_type == 'Polygon':
                polys.append(shp)
            elif shp.geom_type == 'MultiPolygon':
                polys.extend(shp.geoms)
            
            for poly in polys:
                # STRATEGY 1: Standard Polyfill (Lon, Lat)
                added = False
                try:
                    hexes = h3.polyfill(mapping(poly), res, geo_json_conformant=True)
                    if hexes:
                        live_h3s.update(hexes)
                        added = True
                except: pass
                
                # STRATEGY 2: Coordinate Swap (Lat, Lon)
                if not added:
                    try:
                        # Swap x,y coordinates
                        swapped_poly = transform(lambda x, y, z=None: (y, x), poly)
                        hexes = h3.polyfill(mapping(swapped_poly), res, geo_json_conformant=True)
                        if hexes:
                            live_h3s.update(hexes)
                            added = True
                            typer.echo(f"  Feature {i}: Fixed via coordinate swap.")
                    except: pass

                # STRATEGY 3: Centroid Fallback (Tiny polygons)
                # If a polygon is smaller than a hex, polyfill returns empty. 
                # We simply take the center point.
                if not added:
                    centroid = poly.centroid
                    # Try Lon, Lat
                    h = h3.geo_to_h3(centroid.y, centroid.x, res)
                    if h:
                        live_h3s.add(h)
                        # typer.echo(f"  Feature {i}: Used centroid fallback.")
                        added = True
                    else:
                        # Try Lat, Lon
                        h = h3.geo_to_h3(centroid.x, centroid.y, res)
                        if h: 
                            live_h3s.add(h)
                            added = True

        except Exception as e:
            typer.echo(f"Warning: Failed to process feature {i}: {e}")

    typer.echo(f"Found {len(live_h3s)} active H3 cells in live data.")
    
    if len(live_h3s) == 0:
        typer.secho("CRITICAL: Still found 0 cells. Your GeoJSON might use a non-WGS84 projection (e.g. SVY21).", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 2. Determine Forecast Week
    target_date = datetime.date.today()
    target_year, target_week, _ = target_date.isocalendar()
    typer.echo(f"Targeting Forecast Week: {target_year}-W{target_week:02d}")

    # 3. Get Latest Weather
    df_sorted = df_feat.sort_values(["iso_year", "iso_week"])
    last_week = df_sorted.iloc[-1]
    last_year, last_iso = int(last_week["iso_year"]), int(last_week["iso_week"])
    
    # Filter grid
    grid_df = df_feat[["h3"]].drop_duplicates().copy()
    weather_source = df_sorted[(df_sorted["iso_year"] == last_year) & (df_sorted["iso_week"] == last_iso)].iloc[0]
    
    exclude_cols = {"h3", "iso_year", "iso_week", "y_cluster_present", "self_lag_1", "neighbor_pressure_lag_1"}
    keep_cols = [c for c in df_feat.columns if c not in exclude_cols]
    
    new_week_df = grid_df.copy()
    new_week_df["iso_year"] = target_year
    new_week_df["iso_week"] = target_week
    
    for c in keep_cols:
        if c in weather_source:
            new_week_df[c] = weather_source[c]
            
    # 4. Fill Cluster Features
    new_week_df["y_cluster_present"] = new_week_df["h3"].apply(lambda x: 1 if x in live_h3s else 0)
    new_week_df["self_lag_1"] = new_week_df["y_cluster_present"]
    
    h3_set = set(live_h3s)
    def get_pressure(h):
        return sum(1 for k in h3.k_ring(h, 1) if k in h3_set)

    new_week_df["neighbor_pressure_lag_1"] = new_week_df["h3"].apply(get_pressure)
    
    # 5. Save
    df_clean = df_feat[~((df_feat["iso_year"] == target_year) & (df_feat["iso_week"] == target_week))]
    final_df = pd.concat([df_clean, new_week_df], ignore_index=True)
    final_df.to_parquet(out, index=False)
    
    typer.echo(f"Success. Wrote {len(new_week_df)} rows for {target_year}-W{target_week:02d} to {out}")

def main() -> None:
    """Entry point for ``python -m src.cli``."""
    app()


if __name__ == "__main__":
    main()