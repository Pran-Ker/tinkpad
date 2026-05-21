"""Paths, env, constants."""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()
TINKPAD_DIR = Path(os.environ.get("TINKPAD_DIR", HOME / ".tinkpad"))
REGISTRY_PATH = TINKPAD_DIR / "registry.json"
ACTIVE_PATH = TINKPAD_DIR / "active"
FAVORITES_PATH = TINKPAD_DIR / "favorites.json"
SCAN_ROOTS_PATH = TINKPAD_DIR / "scan_roots.json"
SCAN_STAMP_PATH = TINKPAD_DIR / "scan.stamp"
# How fresh a scan must be to skip the implicit refresh on ls/runs.
SCAN_TTL_SECONDS = 60 * 60

TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

DEFAULT_SCAN_ROOTS = [HOME / "Developer"]


def ensure_dir() -> None:
    TINKPAD_DIR.mkdir(parents=True, exist_ok=True)


def api_key() -> str | None:
    return os.environ.get("TINKER_API_KEY")
