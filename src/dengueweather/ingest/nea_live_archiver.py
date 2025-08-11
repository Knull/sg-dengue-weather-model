# src/dengueweather/ingest/nea_live_archiver.py
"""
Archive today's live NEA dengue clusters (GeoJSON) via Data.gov.sg.

Usage:
  python -m dengueweather.ingest.nea_live_archiver --outdir data/raw/nea_live
  # or supply a direct URL to a GeoJSON (overrides API):
  python -m dengueweather.ingest.nea_live_archiver --outdir data/raw/nea_live --url https://...

Notes:
- The Data.gov.sg dataset provides *current active clusters at fetch time*.
- Running this daily builds your own prospective history from today onward.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ..logging_setup import setup_logging

# NEA Dengue Clusters (GEOJSON) dataset on Data.gov.sg
# See: https://data.gov.sg/datasets/d_dbfabf16158d1b0e1c420627c0819168/view
DEFAULT_DATASET_ID = "d_dbfabf16158d1b0e1c420627c0819168"
POLL_ENDPOINT = "https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"


def _find_url(obj: Any) -> Optional[str]:
    """Recursively search for a field named 'url' in a nested JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("http"):
                return v
            out = _find_url(v)
            if out:
                return out
    elif isinstance(obj, list):
        for item in obj:
            out = _find_url(item)
            if out:
                return out
    return None


def _get_presigned_url(dataset_id: str) -> str:
    url = POLL_ENDPOINT.format(dataset_id=dataset_id)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    presigned = _find_url(data)
    if not presigned:
        raise RuntimeError("Could not locate a download URL in Data.gov.sg response.")
    return presigned


def archive_today(outdir: str, dataset_id: Optional[str] = None, direct_url: Optional[str] = None) -> Path:
    dataset_id = dataset_id or os.getenv("NEA_DENGUE_DATASET_ID", DEFAULT_DATASET_ID)
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    out_path = out_dir / f"{today}.geojson"

    # If user supplied a direct URL, use it; otherwise poll the DGS API.
    if direct_url:
        dl_url = direct_url
    else:
        dl_url = _get_presigned_url(dataset_id)

    resp = requests.get(dl_url, timeout=120)
    resp.raise_for_status()

    # Basic sanity check: should be JSON/GeoJSON
    ctype = resp.headers.get("Content-Type", "")
    if "json" not in ctype.lower():
        # Still save (some servers omit headers), but warn
        print(f"[WARN] Unexpected content-type: {ctype}")

    out_path.write_bytes(resp.content)
    return out_path


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Directory to save YYYY-MM-DD.geojson")
    ap.add_argument("--dataset-id", default=None, help="Override Data.gov.sg dataset ID")
    ap.add_argument("--url", default=None, help="Direct GeoJSON URL (skip Data.gov.sg API)")
    args = ap.parse_args()

    out = archive_today(args.outdir, dataset_id=args.dataset_id, direct_url=args.url)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
