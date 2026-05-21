"""Tests for the new cache layer and the register_current_run helper."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from tinkpad.tinker_client import Run, Checkpoint


def _fake_run(rid="abcd1234-aaaa-aaaa-aaaa-000000000000:train:0"):
    return Run(
        run_id=rid, base_model="openai/gpt-oss-120b", is_lora=True, lora_rank=8,
        corrupted=False, last_request_time=datetime(2026, 5, 20, tzinfo=timezone.utc),
        last_checkpoint_path=None, last_sampler_checkpoint_path=None,
    )


def _fake_ckpt(rid="abcd1234-aaaa-aaaa-aaaa-000000000000:train:0"):
    return Checkpoint(
        checkpoint_id="sampler_weights/final", type="sampler",
        tinker_path=f"tinker://{rid}/sampler_weights/final",
        size_bytes=42, created_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        public=False, expires_at=None, run_id=rid,
    )


# ---------- cache ----------
def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, cache
    importlib.reload(config); importlib.reload(cache)
    runs = [_fake_run()]
    ckpts = [_fake_ckpt()]
    cache.save(runs, ckpts)
    r2, c2, ts = cache.load()
    assert len(r2) == 1 and len(c2) == 1
    assert r2[0].run_id == runs[0].run_id
    assert c2[0].tinker_path == ckpts[0].tinker_path
    assert ts is not None


def test_cache_load_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, cache
    importlib.reload(config); importlib.reload(cache)
    r, c, ts = cache.load()
    assert r == [] and c == [] and ts is None
    assert cache.is_fresh() is False


def test_cache_load_or_sync_falls_back_to_disk_on_api_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, cache
    importlib.reload(config); importlib.reload(cache)
    cache.save([_fake_run()], [_fake_ckpt()])
    broken_client = MagicMock()
    broken_client.list_runs.side_effect = RuntimeError("offline")
    # force=True should attempt sync, fail, and still return on-disk cache.
    r, c = cache.load_or_sync(broken_client, force=True)
    assert len(r) == 1 and len(c) == 1


def test_cache_handles_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, cache
    importlib.reload(config); importlib.reload(cache)
    (tmp_path / "cache.json").write_text("{ not json")
    r, c, ts = cache.load()
    assert r == [] and c == [] and ts is None


# ---------- helper ----------
def test_register_current_run_uses_cwd_when_no_name(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, registry as reg_mod, helpers
    importlib.reload(config); importlib.reload(reg_mod); importlib.reload(helpers)

    proj = tmp_path / "my-cool-experiment"
    proj.mkdir()
    monkeypatch.chdir(proj)
    rid = "abcd1234-aaaa-aaaa-aaaa-000000000000:train:0"
    out = helpers.register_current_run(rid)
    assert out == "my-cool-experiment"
    assert reg_mod.Registry().name_for(rid) == "my-cool-experiment"


def test_register_current_run_explicit_name(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, registry as reg_mod, helpers
    importlib.reload(config); importlib.reload(reg_mod); importlib.reload(helpers)
    rid = "abcd1234-aaaa-aaaa-aaaa-000000000000:train:0"
    helpers.register_current_run(rid, "explicit-name")
    assert reg_mod.Registry().name_for(rid) == "explicit-name"
