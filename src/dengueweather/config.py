"""Configuration loading utilities"""

from pathlib import Path
from typing import Any, Dict

import yaml
try:
    from pydantic import BaseModel, Field, ConfigDict  
    _PYD_V2 = True
except Exception:  
    from pydantic import BaseModel, Field
    ConfigDict = None  # type: ignore
    _PYD_V2 = False

from .paths import config_dir


class Config(BaseModel):
    """Pydantic model for the project configuration."""

    project: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    features: Dict[str, Any] = Field(default_factory=dict)
    model: Dict[str, Any] = Field(default_factory=dict)
    viz: Dict[str, Any] = Field(default_factory=dict)
    if _PYD_V2:
        model_config = ConfigDict(extra="allow")  # type: ignore
    else:
        # v1 style
        class Config:  # type: ignore[override]
            extra = "allow"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config() -> Config:
    """Load and merge the default and local configuration files."""
    cfg_dir = config_dir()
    default_path = cfg_dir / "config.default.yaml"
    local_path = cfg_dir / "config.local.yaml"

    if not default_path.exists():
        raise FileNotFoundError(f"Missing default config: {default_path}")

    default_cfg = _load_yaml(default_path)
    local_cfg: Dict[str, Any] = _load_yaml(local_path) if local_path.exists() else {}

    def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = merge_dicts(base[key], value)  # type: ignore[index]
            else:
                base[key] = value
        return base

    merged = merge_dicts(default_cfg.copy(), local_cfg)

    # v2 vs v1 entry point
    if _PYD_V2:
        return Config.model_validate(merged)
    else:
        return Config.parse_obj(merged)
