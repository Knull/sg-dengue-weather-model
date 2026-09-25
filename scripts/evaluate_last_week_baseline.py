#!/usr/bin/env python3
"""Recompute simple lagged-cluster baselines from the public SGCharts archive.

This script intentionally uses only information available before the scored week.
It reconstructs the project's H3-resolution-8 active-cluster labels from raw
SGCharts snapshots and evaluates two naive baselines on the same labelled years
used by the walk-forward GBM evaluation:

1. self_lag_1: the same H3 cell was active in the preceding ISO week.
2. local_pressure_lag_1: number of active cells in the previous week's H3
   k-ring(1), including the cell itself (matching features.py semantics).

The public mirror is used because the repository's raw/processed data are
gitignored. Results should be checked against data/processed/unit_week_features
when that local artifact is available.
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/ansellim/dengue_gaussian/"
    "master/data/dengue_clusters_archive.zip"
)
SG_LAT_MIN, SG_LAT_MAX = 1.1, 1.6
SG_LON_MIN, SG_LON_MAX = 103.5, 104.2
DEFAULT_TEST_YEARS = tuple(range(2014, 2021))


def _parse_date_code(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.fullmatch(r"(\d{6}|\d{8})", text)
    if not match:
        return pd.NaT
    code = match.group(1)
    if len(code) == 6:
        return pd.to_datetime(
            f"20{code[:2]}-{code[2:4]}-{code[4:]}",
            errors="coerce",
        )
    return pd.to_datetime(
        f"{code[:4]}-{code[4:6]}-{code[6:]}",
        errors="coerce",
    )


def _parse_date_from_name(name: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{6,8})", Path(name).name)
    return _parse_date_code(match.group(1)) if match else pd.NaT


def _latlon_to_h3(lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, resolution)
    return h3.latlng_to_cell(lat, lon, resolution)


def _kring(cell: str, radius: int = 1) -> set[str]:
    if hasattr(h3, "k_ring"):
        return set(h3.k_ring(cell, radius))
    return set(h3.grid_disk(cell, radius))


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "sg-dengue-weather-model-baseline/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def load_archive(url: str, h3_res: int = 8) -> pd.DataFrame:
    raw = _download(url)
    rows: list[pd.DataFrame] = []

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        )
        if not members:
            raise RuntimeError("No CSV files found in dengue archive.")

        for member in members:
            with archive.open(member) as handle:
                df = pd.read_csv(
                    handle,
                    header=None,
                    names=list(range(9)),
                    dtype={1: "string"},
                    on_bad_lines="skip",
                )

            df["lat"] = pd.to_numeric(df[2], errors="coerce")
            df["lon"] = pd.to_numeric(df[3], errors="coerce")
            snapshots = df[7].map(_parse_date_code)
            fallback = _parse_date_from_name(member)
            if pd.notna(fallback):
                snapshots = snapshots.fillna(fallback)
            df["snapshot_date"] = pd.to_datetime(snapshots, errors="coerce")

            df = df.dropna(subset=["lat", "lon", "snapshot_date"])
            df = df[
                df["lat"].between(SG_LAT_MIN, SG_LAT_MAX)
                & df["lon"].between(SG_LON_MIN, SG_LON_MAX)
            ].copy()
            if df.empty:
                continue

            iso = df["snapshot_date"].dt.isocalendar()
            df["iso_year"] = iso.year.astype(int)
            df["iso_week"] = iso.week.astype(int)
            df["h3"] = [
                _latlon_to_h3(float(lat), float(lon), h3_res)
                for lat, lon in zip(df["lat"], df["lon"])
            ]
            rows.append(df[["snapshot_date", "iso_year", "iso_week", "h3"]])

    if not rows:
        raise RuntimeError("Archive contained no valid Singapore cluster rows.")

    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["snapshot_date", "h3"]).reset_index(drop=True)


def _full_iso_week_table(start_year: int, end_year: int) -> pd.DataFrame:
    dates = pd.date_range(
        f"{start_year}-01-01",
        f"{end_year}-12-31",
        freq="D",
    )
    iso = dates.isocalendar()
    weeks = pd.DataFrame(
        {
            "iso_year": iso.year.astype(int).to_numpy(),
            "iso_week": iso.week.astype(int).to_numpy(),
        }
    ).drop_duplicates()
    return weeks[
        weeks["iso_year"].between(start_year, end_year)
    ].sort_values(["iso_year", "iso_week"]).reset_index(drop=True)


def build_label_panel(
    archive_rows: pd.DataFrame,
    *,
    start_year: int = 2013,
    end_year: int = 2020,
) -> pd.DataFrame:
    active = (
        archive_rows.loc[
            archive_rows["iso_year"].between(start_year, end_year),
            ["h3", "iso_year", "iso_week"],
        ]
        .drop_duplicates()
        .assign(y_cluster_present=1)
    )

    h3_cells = sorted(active["h3"].unique())
    weeks = _full_iso_week_table(start_year, end_year)
    cells = pd.DataFrame({"h3": h3_cells})
    panel = weeks.merge(cells, how="cross")
    panel = panel.merge(
        active,
        on=["h3", "iso_year", "iso_week"],
        how="left",
    )
    panel["y_cluster_present"] = (
        panel["y_cluster_present"].fillna(0).astype(int)
    )
    panel = panel.sort_values(["iso_year", "iso_week", "h3"]).reset_index(drop=True)
    panel["self_lag_1"] = (
        panel.groupby("h3", sort=False)["y_cluster_present"]
        .shift(1)
        .fillna(0)
        .astype(float)
    )

    cell_set = set(h3_cells)
    neighborhoods = {
        cell: sorted(_kring(cell, 1) & cell_set)
        for cell in h3_cells
    }
    lag_grid = panel.pivot(
        index=["iso_year", "iso_week"],
        columns="h3",
        values="self_lag_1",
    ).fillna(0.0)

    pressure_columns = {}
    for cell in h3_cells:
        neighbors = neighborhoods[cell]
        pressure_columns[cell] = (
            lag_grid[neighbors].sum(axis=1) if neighbors
            else pd.Series(0.0, index=lag_grid.index)
        )
    pressure = pd.DataFrame(pressure_columns, index=lag_grid.index)

    pressure_long = (
        pressure.reset_index()
        .melt(
            id_vars=["iso_year", "iso_week"],
            var_name="h3",
            value_name="local_pressure_lag_1",
        )
    )
    return panel.merge(
        pressure_long,
        on=["iso_year", "iso_week", "h3"],
        how="left",
        validate="one_to_one",
    )


def _expected_precision_at_k_for_ties(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """Expected precision@k under uniform random ordering within score ties."""
    n = len(labels)
    if n == 0:
        return float("nan")
    k = min(k, n)

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]

    remaining = k
    expected_hits = 0.0
    start = 0
    while remaining > 0 and start < n:
        end = start + 1
        while end < n and s[end] == s[start]:
            end += 1
        group_n = end - start
        group_pos = float(y[start:end].sum())
        take = min(remaining, group_n)
        expected_hits += take * (group_pos / group_n)
        remaining -= take
        start = end

    return expected_hits / k


def _weekly_p_at_20(
    year_df: pd.DataFrame,
    score_col: str,
) -> tuple[float, float]:
    deterministic: list[float] = []
    expected: list[float] = []

    for _, week_df in year_df.groupby(["iso_year", "iso_week"], sort=True):
        if int(week_df["y_cluster_present"].sum()) == 0:
            continue

        ranked = week_df.sort_values(
            [score_col, "h3"],
            ascending=[False, True],
            kind="mergesort",
        )
        top = ranked.head(20)
        deterministic.append(float(top["y_cluster_present"].mean()))
        expected.append(
            _expected_precision_at_k_for_ties(
                week_df["y_cluster_present"].to_numpy(dtype=float),
                week_df[score_col].to_numpy(dtype=float),
                20,
            )
        )

    return (
        float(np.mean(deterministic)) if deterministic else float("nan"),
        float(np.mean(expected)) if expected else float("nan"),
    )


def evaluate(
    panel: pd.DataFrame,
    score_col: str,
    test_years: tuple[int, ...] = DEFAULT_TEST_YEARS,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []

    for year in test_years:
        year_df = panel.loc[panel["iso_year"] == year].copy()
        y = year_df["y_cluster_present"].to_numpy(dtype=int)
        score = year_df[score_col].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue

        p20_det, p20_exp = _weekly_p_at_20(year_df, score_col)
        records.append(
            {
                "baseline": score_col,
                "test_year": year,
                "roc_auc": float(roc_auc_score(y, score)),
                "average_precision": float(average_precision_score(y, score)),
                "precision_at_20_h3_tiebreak": p20_det,
                "precision_at_20_expected_ties": p20_exp,
                "positive_rate": float(y.mean()),
                "n_rows": int(len(year_df)),
                "n_positive": int(y.sum()),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--h3-res", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/last_week_baseline.csv"),
    )
    args = parser.parse_args()

    archive_rows = load_archive(args.archive_url, h3_res=args.h3_res)
    panel = build_label_panel(archive_rows)

    results = pd.concat(
        [
            evaluate(panel, "self_lag_1"),
            evaluate(panel, "local_pressure_lag_1"),
        ],
        ignore_index=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    print(
        f"Archive: {len(archive_rows):,} valid snapshot-cell rows, "
        f"{archive_rows['h3'].nunique():,} H3 cells"
    )
    print(f"Panel: {len(panel):,} H3-week rows")
    print(results.to_string(index=False))
    print("\nMacro means by baseline:")
    print(
        results.groupby("baseline")[
            [
                "roc_auc",
                "average_precision",
                "precision_at_20_h3_tiebreak",
                "precision_at_20_expected_ties",
            ]
        ]
        .mean()
        .to_string()
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
