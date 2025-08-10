"""
Download the MOH weekly infectious disease cases (Jan 2012-Dec 2022).

This script calls the Data.gov.sg API to retrieve a pre-signed download
URL for the dataset and saves the CSV to data/raw/moh_weekly_cases.csv.
The dataset contains weekly counts of various infectious diseases. You can
filter for dengue fever later using the helper in `src/dengueweather/ingest/`
or by hand.

If you're wondering why the dataset ID was hardcoded, its because it rarely changes. If a new
version is published, update the `DATASET_ID` constant.
"""

import json
import os
from pathlib import Path
from typing import Optional
import requests


DATASET_ID = "d_ca168b2cb763640d72c4600a68f9909e" 
API_URL = (
    "https://api-open.data.gov.sg/v1/public/api/datasets/"
    f"{DATASET_ID}/poll-download"
)
DEST_PATH = Path("data/raw/moh_weekly_cases.csv")


def fetch_presigned_url(api_url: str) -> str:
    """Returns the presigned download URL from the Data.gov.sg API."""
    resp = requests.get(api_url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(
            f"API responded with code {data.get('code')}: {data.get('errorMsg')}"
        )
    url = data.get("data", {}).get("url")
    if not url:
        raise RuntimeError("No download URL found in API response")
    return url


def download_file(url: str, dest: Path) -> None:
    """Download a file from `url`to `dest``. Overwrites existing file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main() -> None:
    print(f"Fetching presigned download URL for dataset {DATASET_ID} ")
    url = fetch_presigned_url(API_URL)
    print(f"Downloading CSV from {url[:80]}…")
    download_file(url, DEST_PATH)
    print(f"Saved CSV to {DEST_PATH}")


if __name__ == "__main__":
    main()