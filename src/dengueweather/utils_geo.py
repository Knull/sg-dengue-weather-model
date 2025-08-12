"""Geospatial utility functions"""

from typing import Iterable, Tuple

import geopandas as gpd  # type: ignore[import]
import h3
from shapely.geometry import Point  # type: ignore[import]


def latlon_to_h3(lat: float, lon: float, res: int) -> str:
    """Convert latitude and longitude to an H3 index at the given resolution."""
    return h3.geo_to_h3(lat, lon, res)


def h3_to_point(hindex: str) -> Point:
    """Convert an H3 index to a Shapely point representing its centroid."""
    lat, lon = h3.h3_to_geo(hindex)
    return Point(lon, lat)


def assign_h3(df: gpd.GeoDataFrame, lat_col: str, lon_col: str, res: int) -> gpd.GeoDataFrame:
    """Assign an H3 index to each row in a GeoDataFrame."""
    df = df.copy()
    df["h3"] = df.apply(lambda row: latlon_to_h3(row[lat_col], row[lon_col], res), axis=1)
    return df