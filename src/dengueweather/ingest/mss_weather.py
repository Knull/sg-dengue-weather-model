# src/dengueweather/ingest/mss_weather.py
"""
Build weekly weather features from MSS monthly daily CSVs (e.g., Changi).

Input directory layout (for MVP):
  data/raw/mss_weather/CHANGI/daily/*.csv
Each CSV has columns like:
  Station, Year, Month, Day, Daily Rainfall Total (mm), Mean Temperature (°C),
  Mean Relative Humidity (%), Mean Wind Speed (km/h), ...

Outputs:
  data/interim/weather_daily.parquet
  data/interim/weather_weekly.parquet
with canonical columns:
  rain_mm, temp_c, rh_pct, wind_kmh, abs_humidity_gm3 (optional),
  iso_year, iso_week
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from dengueweather.config import load_config
from ..logging_setup import setup_logging


def _normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c0 = str(c).strip().replace("—", "-")
        c0 = re.sub(r"\(.*?\)", "", c0)  # drop "(mm)" etc.
        c0 = c0.replace("°", "")
        c0 = c0.lower()
        c0 = re.sub(r"[^a-z0-9]+", "_", c0).strip("_")
        out.append(c0)
    return out


def _absolute_humidity(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """
    AH (g/m^3) from T(°C) and RH(%):
      e_s (hPa) = 6.112 * exp(17.67*T/(T+243.5))
      e = RH * e_s / 100
      AH = 2.1674 * e / (273.15 + T)
    """
    T = temp_c.astype(float)
    es = 6.112 * np.exp(17.67 * T / (T + 243.5))
    e = (rh_pct.astype(float) / 100.0) * es
    return 2.1674 * e / (273.15 + T)


def _read_month_csv(path: Path) -> pd.DataFrame:
    """Read one MSS monthly daily CSV and return normalized daily rows."""
    df = pd.read_csv(path)
    df.columns = _normalize_cols(list(df.columns))

    # Prefer Year/Month/Day; fall back to any date column if present.
    if all(c in df.columns for c in ["year", "month", "day"]):
        df["date"] = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce"
        )
    else:
        date_col = next((c for c in df.columns if "date" in c), None)
        if not date_col:
            raise ValueError(f"Could not find year/month/day or date column in {path.name}")
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)

    def pick(*substrs) -> Optional[str]:
        for c in df.columns:
            if any(s in c for s in substrs):
                return c
        return None

    c_rain = pick("daily_rainfall_total")
    c_temp = pick("mean_temperature")
    c_rh   = pick("mean_relative_humidity")
    c_wind = pick("mean_wind_speed")

    keep = ["date"]
    ren = {}
    if c_rain: keep.append(c_rain); ren[c_rain] = "rain_mm"
    if c_temp: keep.append(c_temp); ren[c_temp] = "temp_c"
    if c_rh:   keep.append(c_rh);   ren[c_rh]   = "rh_pct"
    if c_wind: keep.append(c_wind); ren[c_wind] = "wind_kmh"

    out = df[keep].rename(columns=ren)

    # Clean numeric columns (convert dashes / blanks to NaN)
    for c in ["rain_mm", "temp_c", "rh_pct", "wind_kmh"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def weekly_from_dir(raw_dir: str, out_daily: str, out_weekly: str, compute_ah: bool = True) -> None:
    """
    Read all monthly daily CSVs under raw_dir and produce daily/weekly parquet files.
    """
    raw = Path(raw_dir)
    files = sorted(raw.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs found under {raw}. Expected monthly files like CHANGI_201501.csv")

    frames = []
    for f in files:
        try:
            frames.append(_read_month_csv(f))
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
    daily = pd.concat(frames, ignore_index=True).dropna(subset=["date"]).sort_values("date")

    if compute_ah and {"temp_c", "rh_pct"}.issubset(daily.columns):
        daily["abs_humidity_gm3"] = _absolute_humidity(daily["temp_c"], daily["rh_pct"])

    # Save daily
    Path(out_daily).parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_daily, index=False)

    # ISO week aggregation
    daily["iso_year"] = daily["date"].dt.isocalendar().year.astype(int)
    daily["iso_week"] = daily["date"].dt.isocalendar().week.astype(int)

    agg = {
        "rain_mm": "sum",            # weekly rainfall total
        "temp_c": "mean",            # weekly means
        "rh_pct": "mean",
        "wind_kmh": "mean",
        "abs_humidity_gm3": "mean",
    }
    cols = [c for c in agg if c in daily.columns]
    weekly = (
        daily[["iso_year", "iso_week"] + cols]
        .groupby(["iso_year", "iso_week"], as_index=False)
        .agg({c: agg[c] for c in cols})
        .sort_values(["iso_year", "iso_week"])
    )

    Path(out_weekly).parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(out_weekly, index=False)
    print(f"Wrote:\n  {out_daily}\n  {out_weekly}")


def main_cli():
    """
    CLI entry point used by `make ingest-weather`.
    Reads config.data.mss_weather_dir and processes CHANGI/daily/*.csv.
    """
    setup_logging()
    cfg = load_config()
    root = Path(cfg.data.mss_weather_dir) / "CHANGI" / "daily"
    weekly_from_dir(
        raw_dir=str(root),
        out_daily="data/interim/weather_daily.parquet",
        out_weekly="data/interim/weather_weekly.parquet",
        compute_ah=cfg.features.use_absolute_humidity,
    )


if __name__ == "__main__":
    main_cli()
