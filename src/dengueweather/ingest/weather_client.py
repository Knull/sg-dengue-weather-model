"""
Engineered client for interacting with MSS (Meteorological Service Singapore) data endpoints.
Handles session lifecycles, retries, and rate-limiting to ensure robust data ingestion.
"""

import time
from pathlib import Path
from typing import Optional
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

MSS_BASE_URL = "https://www.weather.gov.sg/files/dailydata/DAILYDATA_{station}_{yyyymm}.csv"

class WeatherAPIError(Exception):
    """Custom exception for weather API failures."""
    pass

class MSSWeatherClient:
    """
    A robust client for fetching historical daily weather data.
    """
    def __init__(self, retries: int = 3, backoff_factor: float = 0.5):
        self.session = requests.Session()
        
        # Configure robust retry strategy
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Mimic browser to avoid basic bot detection
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DengueWeatherModel/1.0",
            "Referer": "https://www.weather.gov.sg/climate-historical-daily",
        })

    def download_month(self, station_code: str, year: int, month: int, dest_dir: Path) -> Optional[Path]:
        """
        Fetches a specific month's data (simulating an API endpoint).
        """
        yyyymm = f"{year}{month:02d}"
        url = MSS_BASE_URL.format(station=station_code, yyyymm=yyyymm)
        dest_path = dest_dir / f"{station_code}_{yyyymm}.csv"

        if dest_path.exists():
            logger.info(f"Skipping existing: {dest_path.name}")
            return dest_path

        try:
            logger.debug(f"Fetching {url}")
            resp = self.session.get(url, timeout=10)
            
            if resp.status_code == 200:
                # Basic validation: Check if it looks like a CSV or HTML error page
                content_snippet = resp.content[:50].decode('utf-8', errors='ignore')
                if "<html" in content_snippet.lower():
                    raise WeatherAPIError(f"Endpoint returned HTML instead of CSV for {yyyymm}")

                dest_dir.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Downloaded: {dest_path}")
                return dest_path
            elif resp.status_code == 404:
                logger.warning(f"Data not found (404) for {yyyymm}")
                return None
            else:
                raise WeatherAPIError(f"HTTP {resp.status_code} for {url}")

        except requests.RequestException as e:
            logger.error(f"Network error fetching {yyyymm}: {e}")
            raise WeatherAPIError(f"Connection failed") from e

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()