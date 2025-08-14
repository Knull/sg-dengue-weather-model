"""Generate risk maps aggregated to URA subzones."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import geopandas as gpd  # type: ignore[import]
import h3
import joblib  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from shapely.geometry import Polygon  # type: ignore[import]

from dengueweather.config import load_config
from dengueweather.logging_setup import setup_logging


# --- helpers -----------------------------------------------------------------

_SUBZONE_CANDIDATES: List[str] = [
    "SUBZONE_N",
    "SUBZONE_NAME",
    "SUBZONE",
    # sometimes generic
    "Subzone Name",
    "subzone_n",
    "name",
    "NAME",
    # recent URA downloads only expose subzone data inside the Description
    # field.  Those files typically include just a generic "Name" (e.g.
    # "kml_123") and an HTML blob in the "Description" property that
    # contains the true subzone name.  We do not include "Name" here
    # because it identifies a file section rather than the planning
    # subzone; instead we explicitly parse the Description when needed.
]

def _find_subzone_col(gdf: gpd.GeoDataFrame) -> str:
    """Identify the column in a GeoDataFrame that contains the subzone name.

    This helper tries several strategies to discover a usable subzone name
    column:

      1. Look for a known subzone column (case‑insensitive) from a small set
         of expected names (e.g. ``SUBZONE_N``, ``SUBZONE_NAME``).
      2. Fallback to any column whose name contains ``subzone`` (again
         case‑insensitive) and is not an identifier like ``subzone_no``.
      3. Parse the ``Description`` column (case‑insensitive) if present.  Many
         URA downloads include an HTML blob here with rows of ``<th>``/``<td>``
         pairs.  We search for any candidate label inside the ``<th>`` tag and
         extract the corresponding value from the following ``<td>``.  The
         extracted values are assigned to a new column on the GeoDataFrame.

    If none of these strategies succeed a ``ValueError`` is raised with the
    available column names.  The function will mutate the provided
    GeoDataFrame when it needs to create a new column from the Description.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame to inspect and potentially augment.

    Returns
    -------
    str
        The name of the column containing subzone names.

    Raises
    ------
    ValueError
        If a subzone column cannot be identified or extracted.
    """
    # 1) exact matches (case‑insensitive).  Use the actual column name if a
    # candidate matches ignoring case.  Avoid returning ``Name`` because that
    # often contains a file section identifier rather than a true subzone.
    for cand in _SUBZONE_CANDIDATES:
        for col in gdf.columns:
            if col.lower() == cand.lower() and col.lower() != "name":
                return col

    # 2) fuzzy fallback: any column containing "subzone" (case‑insensitive)
    # and not starting with "subzone_no".  This catches variations like
    # "subzone" or "subzone_name".
    for col in gdf.columns:
        lc = col.lower().strip()
        if "subzone" in lc and not lc.startswith("subzone_no"):
            return col

    # 3) attempt to parse subzone from a description field.  Search for a
    # description column case‑insensitively so we handle both "Description"
    # and "description" (and any other casing).  If found, extract values
    # using regex patterns for the set of candidate field names.  We build
    # patterns that tolerate underscores or spaces between tokens to handle
    # variations like "Subzone Name" vs "SUBZONE_NAME".  When a match is
    # found we populate a new column on the GeoDataFrame and return its
    # name.  The new column is based on the candidate label but we avoid
    # clobbering existing columns by appending a numeric suffix if needed.
    desc_col = next((c for c in gdf.columns if c.lower() == "description"), None)
    if desc_col:
        import re  # imported here to avoid dependency at module import time

        def build_pattern(label: str) -> re.Pattern[str]:
            # Escape the label for regex and replace underscores and spaces
            # with a pattern that matches any combination of underscores or
            # whitespace.  This allows matching "SUBZONE_N", "Subzone Name"
            # and similar variants.  Case is ignored via the IGNORECASE flag.
            escaped = re.escape(label)
            # The escaped string may contain escaped underscores "\_" and
            # escaped spaces "\ "; replace either with a character class
            # that matches underscore or whitespace.
            replaced = escaped.replace("\\_", "[_\\s]").replace("\\ ", "[_\\s]")
            return re.compile(rf"<th>\s*{replaced}\s*</th>\s*<td>(.*?)</td>", re.IGNORECASE | re.DOTALL)

        # Precompile patterns for all candidate labels
        patterns: list[tuple[str, re.Pattern[str]]] = []
        for cand in _SUBZONE_CANDIDATES:
            patterns.append((cand, build_pattern(cand)))

        # Ensure the description column is a string series
        desc_series = gdf[desc_col].astype(str)

        for cand, pat in patterns:
            # Attempt to extract values for this candidate label
            extracted = desc_series.apply(
                lambda x: pat.search(x).group(1).strip() if isinstance(x, str) and pat.search(x) else None
            )
            if extracted.notna().any():
                # Determine a column name for the extracted values.  Prefer
                # using the uppercase version of the candidate (e.g., SUBZONE_N)
                # unless that name is already in the GeoDataFrame.  If it
                # exists, append a numeric suffix to avoid clobbering.
                base_name = cand.upper()
                new_col = base_name
                if new_col in gdf.columns:
                    i = 1
                    while f"{base_name}_{i}" in gdf.columns:
                        i += 1
                    new_col = f"{base_name}_{i}"
                gdf[new_col] = extracted
                return new_col

    # 4) last resort: unable to find or extract a subzone column
    raise ValueError(
        f"Could not identify subzone name column in URA GeoJSON. Columns found: {list(gdf.columns)}"
    )


def _h3_to_polygon(h: str) -> Polygon:
    """Convert an H3 cell index to a Shapely polygon.

    The h3 library can return boundary coordinates in either (lat, lon) or
    (lon, lat) order.  When ``geo_json=True`` is passed, the output is in
    GeoJSON convention: each coordinate is a (lon, lat) pair.  Shapely
    polygons also expect coordinates in (x, y) order = (lon, lat), so we
    simply construct the polygon directly from the returned sequence.

    Parameters
    ----------
    h : str
        H3 index string

    Returns
    -------
    shapely.geometry.Polygon
        Polygon representing the cell boundary.
    """
    # Request GeoJSON‑style output (longitude, latitude ordering)
    boundary = h3.h3_to_geo_boundary(h, geo_json=True)
    # ``boundary`` is a list of [lon, lat] pairs; shapely expects (lon, lat)
    return Polygon(boundary)


# --- main --------------------------------------------------------------------

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

    # Load features and choose week
    feats = pd.read_parquet(features_path)
    if feats.empty:
        print("Features table is empty; cannot create map.")
        return

    if iso_year is None or iso_week is None:
        latest = feats[["iso_year", "iso_week"]].dropna().sort_values(["iso_year", "iso_week"]).iloc[-1]
        iso_year, iso_week = int(latest["iso_year"]), int(latest["iso_week"])

    week_df = feats[(feats["iso_year"] == iso_year) & (feats["iso_week"] == iso_week)].copy()
    if week_df.empty:
        print(f"No records found for year {iso_year}, week {iso_week}")
        return

    # Load model (+ optional scaler)
    payload = joblib.load(model_path)
    model = payload.get("model")
    feature_cols = payload.get("feature_cols", [])
    scaler = payload.get("scaler", None)
    if model is None or not feature_cols:
        print(f"Model or feature columns missing in {model_path}")
        return

    # Predict probabilities (apply scaler if present)
    X = week_df[feature_cols].fillna(0.0).astype("float64").values
    if scaler is not None:
        X = scaler.transform(X)
    week_df["pred_proba"] = model.predict_proba(X)[:, 1]

    # Convert H3 indices to polygons → GeoDataFrame
    week_df["geometry"] = week_df["h3"].apply(_h3_to_polygon)
    h3_gdf = gpd.GeoDataFrame(week_df[["h3", "pred_proba", "geometry"]], geometry="geometry", crs="EPSG:4326")

    # Load URA subzones and standardize CRS
    ura_gdf = gpd.read_file(ura_path)
    # Some URA geometries include a Z (height) coordinate which can
    # interfere with spatial joins against 2D H3 polygons.  Drop any Z
    # coordinates by round‑tripping through WKB with a 2D output.  If the
    # shapely version does not support this API, fall back gracefully.
    try:
        from shapely import wkb  # type: ignore[import]

        def _to_2d(geom: Polygon) -> Polygon:
            try:
                return wkb.loads(wkb.dumps(geom, output_dimension=2))
            except Exception:
                return geom
        ura_gdf["geometry"] = ura_gdf["geometry"].apply(_to_2d)
    except Exception:
        # If shapely.wkb is unavailable, proceed without dropping Z
        pass

    # Ensure geometries are in EPSG:4326
    if ura_gdf.crs is None:
        ura_gdf = ura_gdf.set_crs("EPSG:4326")
    elif str(ura_gdf.crs) != "EPSG:4326":
        ura_gdf = ura_gdf.to_crs("EPSG:4326")

    subzone_col = _find_subzone_col(ura_gdf)

    # Spatial join + aggregation.  Geopandas versions prior to 0.10 use
    # the ``op`` keyword instead of ``predicate``.  Attempt to call
    # ``sjoin`` with the newer ``predicate`` argument and fall back to
    # ``op`` if a TypeError is raised.  If no intersections are found,
    # attempt a secondary join using the centroids of the H3 polygons and
    # a ``within`` predicate.  This handles cases where the URA polygons
    # contain 3D coordinates (Z values) that may prevent direct polygon
    # intersections.
    def do_join(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, how: str, pred: str):
        try:
            return gpd.sjoin(left, right, how=how, predicate=pred)
        except TypeError:
            # Fallback for older geopandas versions
            return gpd.sjoin(left, right, how=how, op=pred)

    # Try polygon–polygon intersects join first
    joined = do_join(
        h3_gdf[["pred_proba", "geometry"]],
        ura_gdf[[subzone_col, "geometry"]],
        how="inner",
        pred="intersects",
    )
    # If no matches, retry using centroids and within predicate
    if joined.empty:
        # Convert hexagons to their centroid points
        h3_pts = h3_gdf.copy()
        # The centroid attribute in geopandas returns 2D centroids
        h3_pts["geometry"] = h3_pts["geometry"].centroid
        joined = do_join(
            h3_pts[["pred_proba", "geometry"]],
            ura_gdf[[subzone_col, "geometry"]],
            how="inner",
            pred="within",
        )
        if joined.empty:
            print("No H3 cells joined to any subzones; check geometry alignment.")
            return

    if agg_method == "sum":
        risk_by_subzone = joined.groupby(subzone_col)["pred_proba"].sum()
    elif agg_method == "max":
        risk_by_subzone = joined.groupby(subzone_col)["pred_proba"].max()
    else:
        risk_by_subzone = joined.groupby(subzone_col)["pred_proba"].mean()

    ura_plot = ura_gdf.merge(risk_by_subzone.rename("risk"), on=subzone_col, how="left")

    # Plot
    out_dir_path = Path(out_dir); out_dir_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ura_plot.plot(
        column="risk",
        cmap="YlOrRd",
        linewidth=0.5,
        edgecolor="white",
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(f"Predicted dengue cluster risk\nISO Year {iso_year}, Week {iso_week}", fontsize=12)

    outfile = out_dir_path / f"hindcast_map_{iso_year}_{iso_week}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Hindcast map saved to {outfile}")
