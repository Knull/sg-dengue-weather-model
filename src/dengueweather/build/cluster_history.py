# src/dengueweather/build/cluster_week.py
"""
This file converts a list of dengue cluster observations (snapshots) into a weekly panel dataset.
It shows for every geographic area and for every week in the study period whether a dengue cluster was present or not.
"""
from pathlib import Path
import pandas as pd
import h3
from dengueweather.config import load_config

def build_cluster_week(in_path: str, out_path: str):
    hist = pd.read_parquet(in_path)  # expects columns: cluster_key, address, lat, lon, cases, snapshot_date
    cfg = load_config()
    res = cfg.project.h3_res

    # collapse per cluster_key across snapshots --> first/last seen, peak cases, median coords
    g = hist.groupby("cluster_key", as_index=False).agg(
        first_seen=("snapshot_date","min"),
        last_seen =("snapshot_date","max"),
        peak_cases=("cases","max"),
        lat=("lat","median"),
        lon=("lon","median"),
        address=("address", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
    )
    # censor flags
    min_d, max_d = hist["snapshot_date"].min().normalize(), hist["snapshot_date"].max().normalize()
    g["left_censored"]  = g["first_seen"].dt.normalize().eq(min_d)
    g["right_censored"] = g["last_seen"].dt.normalize().eq(max_d)

    # explode dates --> weeks
    def cluster_weeks(r):
        rng = pd.date_range(r["first_seen"], r["last_seen"], freq="D")
        wk = pd.DataFrame({"date": rng})
        wk["iso_year"] = wk["date"].dt.isocalendar().year.astype(int)
        wk["iso_week"] = wk["date"].dt.isocalendar().week.astype(int)
        wk = wk.drop_duplicates(["iso_year","iso_week"])
        wk["cluster_key"] = r["cluster_key"]
        wk["h3"] = h3.geo_to_h3(r["lat"], r["lon"], res)
        return wk[["cluster_key","h3","iso_year","iso_week"]]

    parts = [cluster_weeks(r) for _, r in g.iterrows()]
    cw = pd.concat(parts, ignore_index=True)

    # label y=1 for presence; keep peak_cases etc for context
    cw = cw.merge(g[["cluster_key","peak_cases","first_seen","last_seen","left_censored","right_censored"]],
                  on="cluster_key", how="left")
    cw["y_cluster_present"] = 1

    # build full panel over observed weeks x all H3s that ever had a cluster
    weeks = cw[["iso_year","iso_week"]].drop_duplicates()
    h3s   = cw[["h3"]].drop_duplicates()
    panel = weeks.merge(h3s, how="cross")  # cartesian product
    panel = panel.merge(cw[["h3","iso_year","iso_week","y_cluster_present","peak_cases","first_seen","last_seen","left_censored","right_censored"]],
                        on=["h3","iso_year","iso_week"], how="left")
    panel["y_cluster_present"] = panel["y_cluster_present"].fillna(0).astype(int)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"wrote unit-week labels --> {out_path} (rows={len(panel)})")

def main_cli():
    build_cluster_week("data/interim/archive_merged.parquet", "data/processed/cluster_week.parquet")

if __name__ == "__main__":
    main_cli()
