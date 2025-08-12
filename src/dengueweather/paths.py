"""streamlines common filesystem paths for the project"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def data_dir(subdir: str) -> Path:
    """Return the path to a subdirectory under `data/`"""
    return ROOT / "data" / subdir

def config_dir() -> Path:
    """Return the path to the configuration directory."""
    return ROOT / "config"

def docs_dir() -> Path:
    """Return the path to the documentation directory (`docs/`)."""
    return ROOT / "docs"