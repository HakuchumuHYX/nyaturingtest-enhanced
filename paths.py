"""Workspace-local runtime paths for the nyaturingtest plugin.

Path resolution is intentionally side-effect free.  Directories are created by
the component that owns them, not while this module is imported.
"""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "data" / "nyaturingtest"
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "cache" / "nyaturingtest"
DEFAULT_BACKUP_DIR = WORKSPACE_ROOT / "data" / "nyaturingtest_backups"
DEFAULT_PRESET_DIR = WORKSPACE_ROOT / "config" / "nyaturingtest" / "nya_presets"


def _configured_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def get_data_dir() -> Path:
    return _configured_path("NYATURINGTEST_DATA_DIR", DEFAULT_DATA_DIR)


def get_cache_dir() -> Path:
    return _configured_path("NYATURINGTEST_CACHE_DIR", DEFAULT_CACHE_DIR)


def get_backup_dir() -> Path:
    return _configured_path("NYATURINGTEST_BACKUP_DIR", DEFAULT_BACKUP_DIR)


def get_preset_dir() -> Path:
    return _configured_path("NYATURINGTEST_PRESET_DIR", DEFAULT_PRESET_DIR)


def get_vector_dir(session_id: str) -> Path:
    return get_data_dir() / f"vector_index_{session_id}"


def get_image_cache_dir() -> Path:
    return get_cache_dir() / "image_cache"
