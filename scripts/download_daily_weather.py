"""
download_daily_weather.py

Downloads daily climate records from the MSS climate portal for a range of years and months. 
It saves each month as a CSV and then compresses them into a zip file
"""

import os
import requests
import calendar
from datetime import date
from zipfile import ZipFile

BASE_URL = "https://www.weather.gov.sg/files/dailydata/DAILYDATA_{station}_{yyyymm}.csv"

def download_month(station_code: str, year: int, month: int, out_dir: str):
    """Download one month of data."""
    yyyymm = f"{year}{month:02d}"
    url = BASE_URL.format(station=station_code, yyyymm=yyyymm)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.weather.gov.sg/climate-historical-daily",
        "Origin": "https://www.weather.gov.sg",
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{station_code}_{yyyymm}.csv")
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"Saved {path}")
        return path
    else:
        print(f"Failed {url} -> HTTP {resp.status_code}")
        return None

def download_range(station_code: str, start_year: int, end_year: int, out_dir: str):
    """Download a continuous range of months (inclusive)."""
    saved_files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if date(year, month, 1) > date.today():
                continue
            path = download_month(station_code, year, month, out_dir)
            if path:
                saved_files.append(path)
    return saved_files

def zip_files(file_list, zip_path):
    """Zip up all downloaded CSV files."""
    with ZipFile(zip_path, "w") as zf:
        for f in file_list:
            zf.write(f, arcname=os.path.basename(f))
    print(f"Created zip: {zip_path}")

if __name__ == "__main__":
    station_code = "S24" # this is the changi MSS code
    files = download_range(station_code, 2013, 2020, "weather_csvs")
    zip_files(files, "weather_changi_2013_2020.zip")
