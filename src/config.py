
import os
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"

_ALL_KEYS = ("HPA_DATA", "HPA_META", "UKB_DATA", "DATA_DIR", "RESULT_DIR")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def get_paths(config_path: str | None = None) -> dict:

    cfg_file = Path(config_path) if config_path else _DEFAULT_CONFIG
    cfg = _load_yaml(cfg_file)
    paths = {}
    missing = []
    for key in _ALL_KEYS:
        value = os.environ.get(f"HPA_UKB_{key}") or cfg.get(key)
        if value is None or str(value).strip() == "":
            missing.append(key)
        else:
            paths[key] = str(value)
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(
            "Missing required path configuration for: "
            f"{missing_str}. Set these keys in config.yaml or as "
            "HPA_UKB_<KEY> environment variables. See config.example.yaml."
        )
    return paths
