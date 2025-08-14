# src/dengueweather/build/cluster_history.py
"""Construct stable cluster histories and expand to (H3, week) labels."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

try:
    import h3
except Exception:
    h3 = None  

from dengueweather.config import load_config


def _mode_or_first(s: pd.Series) -> str:
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else s.iloc[0]


def build_history(in_path: str, out_path: str) -> None:
    df = pd.read_parquet(in_path)
    if df.empty:
        raise SystemExit("archive is empty. did ingest-archive run?")

    # Overall window for censor flags
    min_d = pd.to_datetime(df["snapshot_date"]).min().normalize()
    max_d = pd.to_datetime(df["snapshot_date"]).max().normalize()

    g = (
        df.groupby("cluster_key", as_index=False)
          .agg(
              first_seen=("snapshot_date", "min"),
              last_seen =("snapshot_date", "max"),
              peak_cases=("cases", "max"),
              lat=("lat", "median"),
              lon=("lon", "median"),
              address=("address", _mode_or_first),
          )
    )
    g["first_seen"] = pd.to_datetime(g["first_seen"])
    g["last_seen"]  = pd.to_datetime(g["last_seen"])

    g["left_censored"]  = g["first_seen"].dt.normalize().eq(min_d)
    g["right_censored"] = g["last_seen"].dt.normalize().eq(max_d)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out_path, index=False)
    print(f"history: {len(g)} clusters → {out_path}")


def build_cluster_week(history_path: str, out_path: str, h3_res: int | None = None) -> None:
    if h3 is None:
        raise RuntimeError("h3 package is required for build_cluster_week. Install `h3`.")

    hist = pd.read_parquet(history_path)
    if hist.empty:
        raise SystemExit("empty history – run build_history first")

    cfg = load_config()
    res = h3_res if h3_res is not None else int(cfg.project.get("h3_res", 8))

    # For each cluster, enumerate ISO weeks it was active
    def _weeks_for_row(r: pd.Series) -> pd.DataFrame:
        rng = pd.date_range(r["first_seen"], r["last_seen"], freq="D")
        if rng.empty:
            return pd.DataFrame(columns=["cluster_key", "h3", "iso_year", "iso_week"])
        wk = pd.DataFrame({"date": rng})
        iso = wk["date"].dt.isocalendar()
        wk["iso_year"] = iso.year.astype(int)
        wk["iso_week"] = iso.week.astype(int)
        wk = wk.drop_duplicates(["iso_year", "iso_week"])
        wk["cluster_key"] = r["cluster_key"]
        wk["h3"] = h3.geo_to_h3(float(r["lat"]), float(r["lon"]), res)
        return wk[["cluster_key", "h3", "iso_year", "iso_week"]]

    parts = []
    for _, r in hist.iterrows():
        parts.append(_weeks_for_row(r))

    cw = pd.concat(parts, ignore_index=True)
    cw["y_cluster_present"] = 1

    keep = ["cluster_key", "peak_cases", "first_seen", "last_seen", "left_censored", "right_censored"]
    cw = cw.merge(hist[keep], on="cluster_key", how="left")

    # Build full panel across all observed weeks and H3s that ever had a cluster
    weeks = cw[["iso_year", "iso_week"]].drop_duplicates()
    h3s   = cw[["h3"]].drop_duplicates()
    panel = weeks.merge(h3s, how="cross")
    panel = panel.merge(
        cw[["h3", "iso_year", "iso_week", "y_cluster_present", "peak_cases", "first_seen", "last_seen",
            "left_censored", "right_censored"]],
        on=["h3", "iso_year", "iso_week"], how="left"
    )
    panel["y_cluster_present"] = panel["y_cluster_present"].fillna(0).astype(int)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"cluster-week: {len(panel):,} rows → {out_path}")
