"""Local run_id -> experiment-name registry.

The registry maps Tinker training-run IDs (e.g. `5a2c64eb-...:train:0`) to a
short human-friendly experiment name (typically the local folder name where
the run was launched from).

Two ways entries arrive:
  1. Manual: `tinkpad register <run_id> <name>`.
  2. Auto-scan: we walk configured scan roots (default ~/Developer) looking
     for the project's `Zexp/`+`Zlog/<run_id>/` convention or any folder that
     contains a file/dir whose name embeds a tinker run_id. The enclosing
     project folder name becomes the experiment label.

Entries persist as JSON at ~/.tinkpad/registry.json.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from ._atomic import atomic_write_text
from .config import (
    REGISTRY_PATH,
    SCAN_ROOTS_PATH,
    SCAN_STAMP_PATH,
    SCAN_TTL_SECONDS,
    DEFAULT_SCAN_ROOTS,
    ensure_dir,
)

# Tinker run IDs look like: 5a2c64eb-08b6-5dc5-b927-63429a38f004 (UUID) or
# the longer form 5a2c64eb-08b6-5dc5-b927-63429a38f004:train:0
_RUN_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?::train:\d+)?"
)


@dataclass
class Entry:
    run_id: str
    name: str
    source_path: str | None = None  # local folder that produced this mapping
    note: str | None = None


class Registry:
    def __init__(self, path: Path | None = None):
        # Resolve at call time so tests can monkeypatch REGISTRY_PATH.
        self.path = path if path is not None else REGISTRY_PATH
        self._entries: dict[str, Entry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text() or "{}")
        except json.JSONDecodeError:
            # A truncated / corrupt registry must not kill the CLI. Quarantine
            # the bad file (so the user can recover it) and start fresh.
            backup = self.path.with_suffix(self.path.suffix + f".corrupt.{int(time.time())}")
            try:
                self.path.rename(backup)
            except OSError:
                pass
            return
        if not isinstance(raw, dict):
            return
        for rid, data in raw.items():
            try:
                self._entries[rid] = Entry(**data)
            except TypeError:
                # Schema drift on a single row shouldn't poison the rest.
                continue

    def save(self) -> None:
        ensure_dir()
        atomic_write_text(
            self.path,
            json.dumps({rid: asdict(e) for rid, e in self._entries.items()}, indent=2),
        )

    def get(self, run_id: str) -> Entry | None:
        return self._entries.get(_normalize(run_id))

    def name_for(self, run_id: str) -> str | None:
        e = self.get(run_id)
        return e.name if e else None

    def set(self, run_id: str, name: str, source_path: str | None = None, note: str | None = None) -> Entry:
        """Add/update an entry.

        A manual entry (no source_path) is never clobbered by an auto-scan
        (source_path set). Auto-scans CAN refresh other auto-scan entries.
        """
        run_id = _normalize(run_id)
        existing = self._entries.get(run_id)
        is_auto_scan_write = source_path is not None
        existing_is_manual = existing is not None and existing.source_path is None
        if existing and is_auto_scan_write and existing_is_manual:
            return existing
        e = Entry(run_id=run_id, name=name, source_path=source_path, note=note)
        self._entries[run_id] = e
        return e

    def remove(self, run_id: str) -> bool:
        run_id = _normalize(run_id)
        return self._entries.pop(run_id, None) is not None

    def all(self) -> dict[str, Entry]:
        return dict(self._entries)


def _normalize(run_id: str) -> str:
    # Strip tinker:// prefix and any trailing path
    rid = run_id.removeprefix("tinker://")
    rid = rid.split("/", 1)[0]
    return rid


def load_scan_roots() -> list[Path]:
    if SCAN_ROOTS_PATH.exists():
        try:
            data = json.loads(SCAN_ROOTS_PATH.read_text() or "[]")
            if isinstance(data, list):
                return [Path(p) for p in data]
        except json.JSONDecodeError:
            pass
    return list(DEFAULT_SCAN_ROOTS)


def save_scan_roots(roots: Iterable[Path]) -> None:
    atomic_write_text(SCAN_ROOTS_PATH, json.dumps([str(p) for p in roots], indent=2))


def scan_is_fresh(ttl: int = SCAN_TTL_SECONDS) -> bool:
    try:
        return (time.time() - SCAN_STAMP_PATH.stat().st_mtime) < ttl
    except OSError:
        return False


def maybe_scan(roots: Iterable[Path] | None = None) -> list[Entry]:
    """Run scan() only if the last scan is older than SCAN_TTL_SECONDS.

    Called as a cheap implicit refresh from `ls`/`runs`.
    """
    if scan_is_fresh():
        return []
    return scan(roots)


def scan(roots: Iterable[Path] | None = None, max_depth: int = 4) -> list[Entry]:
    """Walk roots looking for Zlog/<run_id> folders and similar conventions.

    Returns a list of newly-discovered (or refreshed) entries.
    """
    roots = list(roots) if roots is not None else load_scan_roots()
    reg = Registry()
    found: list[Entry] = []
    for root in roots:
        if not root.exists():
            continue
        for path in _walk(root, max_depth):
            # Two heuristics:
            # (a) The folder NAME itself is/contains a run_id.
            # (b) Inside the folder, a subdir named "Zlog" contains run_id dirs
            #     (the user's documented convention).
            m = _RUN_ID_RE.search(path.name)
            if m and path.is_dir():
                run_id = m.group(0)
                exp_name = _experiment_name_for(path)
                e = reg.set(run_id, exp_name, source_path=str(path), note="auto-scan")
                found.append(e)
                continue
            zlog = path / "Zlog"
            if path.is_dir() and zlog.is_dir():
                exp_name = path.name
                for child in zlog.iterdir():
                    if not child.is_dir():
                        continue
                    m = _RUN_ID_RE.search(child.name)
                    if not m:
                        continue
                    run_id = m.group(0)
                    e = reg.set(run_id, exp_name, source_path=str(child), note="auto-scan")
                    found.append(e)
    reg.save()
    ensure_dir()
    try:
        SCAN_STAMP_PATH.touch()
    except OSError:
        pass
    return found


def _walk(root: Path, max_depth: int):
    """Iterate dirs up to max_depth from root."""
    root = root.resolve()
    base_depth = len(root.parts)
    try:
        stack = [root]
    except Exception:
        return
    while stack:
        cur = stack.pop()
        try:
            children = list(cur.iterdir())
        except (PermissionError, OSError):
            continue
        for c in children:
            if c.is_symlink():
                continue
            if c.is_dir():
                depth = len(c.parts) - base_depth
                yield c
                if depth < max_depth:
                    stack.append(c)


def _experiment_name_for(path: Path) -> str:
    # Use the nearest meaningful ancestor folder name (skip Zlog itself).
    parts = list(path.parts)
    for p in reversed(parts[:-1]):
        if p not in {"Zlog", "Zexp", "logs", "runs"}:
            return p
    return path.name
