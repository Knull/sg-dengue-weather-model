# src/dengueweather/ingest/sgcharts_archive.py
from pathlib import Path
import pandas as pd
import re, hashlib
from dengueweather.config import load_config
# Bounds of singapore
SG_LAT_MIN, SG_LAT_MAX = 1.1, 1.6
SG_LON_MIN, SG_LON_MAX = 103.5, 104.2

def _parse_date_code(s: str) -> pd.Timestamp | None:
    if pd.isna(s): return None
    s = str(s).strip()
    m = re.fullmatch(r"(\d{6}|\d{8})", s)
    if not m: return None
    x = m.group(1)
    if len(x) == 6:  # YYMMDD -> assume 20YY
        return pd.to_datetime(f"20{x[:2]}-{x[2:4]}-{x[4:]}", errors="coerce")
    return pd.to_datetime(f"{x[:4]}-{x[4:6]}-{x[6:]}", errors="coerce")

def _parse_date_from_name(name: str) -> pd.Timestamp | None:
    m = re.search(r"(\d{6,8})", name)
    return _parse_date_code(m.group(1)) if m else None

def _norm_addr(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _cluster_key(address: str, lat: float, lon: float) -> str:
    base = f"{_norm_addr(address)}|{round(float(lat),5)}|{round(float(lon),5)}"
    return hashlib.md5(base.encode()).hexdigest()

def merge_archive(input_dir: str, out_path: str):
    root = Path(input_dir)
    files = sorted(list(root.rglob("*.csv")))
    if not files:
        raise SystemExit(f"No CSVs under {root}. Put SGCharts folders in there.")

    rows = []
    for f in files:
        df = pd.read_csv(f, header=None, names=[0,1,2,3,4,5,6,7,8])
        # expected: 1=address, 2=lat, 3=lon, 6=cases, 7=date_code (YYMMDD)
        df["address"] = df[1].astype(str).fillna("")
        df["lat"] = pd.to_numeric(df[2], errors="coerce")
        df["lon"] = pd.to_numeric(df[3], errors="coerce")
        df["cases"] = pd.to_numeric(df[6], errors="coerce")
        # try date from col7, else from filename
        snap = df[7].astype(str).map(_parse_date_code)
        fallback = _parse_date_from_name(f.name)
        df["snapshot_date"] = snap.fillna(fallback)
        # bounds + minimal required fields
        df = df.dropna(subset=["lat","lon","snapshot_date"])
        df = df[(df["lat"].between(SG_LAT_MIN, SG_LAT_MAX)) & (df["lon"].between(SG_LON_MIN, SG_LON_MAX))]
        df["cluster_key"] = [_cluster_key(a, la, lo) for a, la, lo in zip(df["address"], df["lat"], df["lon"])]
        rows.append(df[["cluster_key","address","lat","lon","cases","snapshot_date"]])

    out = pd.concat(rows, ignore_index=True).sort_values("snapshot_date").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"merged {len(files)} CSVs → {out.shape[0]} rows → {out_path}")

def main_cli():
    cfg = load_config()
    merge_archive(cfg.data.sgcharts_csv_dir, "data/interim/archive_merged.parquet")

if __name__ == "__main__":
    main_cli()
