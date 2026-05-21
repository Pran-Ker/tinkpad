"""Local metadata cache.

We never download checkpoint *files* — only the names, sizes, types, and
timestamps. The cache makes `ls`/`runs`/`tree` instant (no network) and
preserves a recent snapshot of what was on the server.

Layout: a single JSON file at ~/.tinkpad/cache.json:

  {
    "synced_at": "2026-05-21T12:34:56Z",
    "runs":      [<Run dict>, ...],
    "checkpoints": [<Checkpoint dict>, ...]
  }

Refresh is explicit (`tinkpad sync`) or implicit when the cache is older
than CACHE_TTL_SECONDS.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ._atomic import atomic_write_text
from .config import TINKPAD_DIR, ensure_dir
from .tinker_client import Checkpoint, Run, TinkerClient

CACHE_PATH = TINKPAD_DIR / "cache.json"
CACHE_TTL_SECONDS = 5 * 60


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _run_to_dict(r: Run) -> dict:
    d = asdict(r)
    d["last_request_time"] = _iso(r.last_request_time)
    d["last_checkpoint_created_at"] = _iso(r.last_checkpoint_created_at)
    d["last_sampler_checkpoint_created_at"] = _iso(r.last_sampler_checkpoint_created_at)
    return d


def _dict_to_run(d: dict) -> Run:
    return Run(
        run_id=d["run_id"],
        base_model=d["base_model"],
        is_lora=d["is_lora"],
        lora_rank=d.get("lora_rank"),
        corrupted=d.get("corrupted", False),
        last_request_time=_parse_iso(d.get("last_request_time")),
        last_checkpoint_path=d.get("last_checkpoint_path"),
        last_sampler_checkpoint_path=d.get("last_sampler_checkpoint_path"),
        last_checkpoint_created_at=_parse_iso(d.get("last_checkpoint_created_at")),
        last_sampler_checkpoint_created_at=_parse_iso(d.get("last_sampler_checkpoint_created_at")),
    )


def _ckpt_to_dict(c: Checkpoint) -> dict:
    d = asdict(c)
    d["created_at"] = _iso(c.created_at)
    d["expires_at"] = _iso(c.expires_at)
    return d


def _dict_to_ckpt(d: dict) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=d["checkpoint_id"],
        type=d["type"],
        tinker_path=d["tinker_path"],
        size_bytes=d.get("size_bytes", 0),
        created_at=_parse_iso(d.get("created_at")),
        public=d.get("public", False),
        expires_at=_parse_iso(d.get("expires_at")),
        run_id=d["run_id"],
    )


def load() -> tuple[list[Run], list[Checkpoint], float | None]:
    """Return (runs, checkpoints, synced_at_epoch). Empty lists if no cache."""
    if not CACHE_PATH.exists():
        return [], [], None
    try:
        raw = json.loads(CACHE_PATH.read_text() or "{}")
    except json.JSONDecodeError:
        return [], [], None
    runs = [_dict_to_run(d) for d in raw.get("runs", [])]
    ckpts = [_dict_to_ckpt(d) for d in raw.get("checkpoints", [])]
    synced_at = _parse_iso(raw.get("synced_at"))
    return runs, ckpts, (synced_at.timestamp() if synced_at else None)


def is_fresh(ttl: int = CACHE_TTL_SECONDS) -> bool:
    _, _, ts = load()
    return ts is not None and (time.time() - ts) < ttl


def save(runs: list[Run], ckpts: list[Checkpoint]) -> None:
    ensure_dir()
    payload = {
        "synced_at": _iso(datetime.now(timezone.utc)),
        "runs": [_run_to_dict(r) for r in runs],
        "checkpoints": [_ckpt_to_dict(c) for c in ckpts],
    }
    atomic_write_text(CACHE_PATH, json.dumps(payload, indent=2))


def sync(client: TinkerClient | None = None) -> tuple[int, int]:
    """Refresh the cache from the live API. Returns (n_runs, n_ckpts)."""
    client = client or TinkerClient()
    runs = client.list_runs()
    ckpts = client.list_checkpoints()
    save(runs, ckpts)
    return len(runs), len(ckpts)


def load_or_sync(client: TinkerClient | None = None, *, ttl: int = CACHE_TTL_SECONDS, force: bool = False) -> tuple[list[Run], list[Checkpoint]]:
    """Return cached runs/ckpts, syncing first if stale or absent.

    Falls back to whatever's on disk if sync fails (offline mode).
    """
    if force or not is_fresh(ttl):
        try:
            sync(client)
        except Exception:
            pass  # fall back to stale cache silently
    runs, ckpts, _ = load()
    return runs, ckpts
