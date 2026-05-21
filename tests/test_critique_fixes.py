"""Tests added in response to the critique pass.

Cover: corruption survival, ambiguous resolver, manual-entry persistence,
realistic SDK-shape fixture, _resolve_path helpers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from tinkpad import registry as reg_mod
from tinkpad.registry import Registry, scan
from tinkpad import tinker_client as tc
from tinkpad.tinker_client import Run, Checkpoint, _to_run, _to_ckpt
from tinkpad import cli as cli_mod
from tinkpad.cli import _resolve_path, _resolve_run_id, app


runner = CliRunner()


def _fake_run(rid="abcd1234-08b6-5dc5-b927-63429a38f004:train:0"):
    return Run(
        run_id=rid, base_model="openai/gpt-oss-120b", is_lora=True, lora_rank=32,
        corrupted=False, last_request_time=datetime(2026, 5, 20, tzinfo=timezone.utc),
        last_checkpoint_path=None, last_sampler_checkpoint_path=None,
    )


def _fake_ckpt(rid="abcd1234-08b6-5dc5-b927-63429a38f004:train:0", step="final", kind="sampler", t=None):
    t = t or datetime(2026, 5, 20, tzinfo=timezone.utc)
    suffix = f"sampler_weights/{step}" if kind == "sampler" else f"weights/{step}"
    return Checkpoint(
        checkpoint_id=suffix, type=kind,
        tinker_path=f"tinker://{rid}/{suffix}",
        size_bytes=1, created_at=t, public=False, expires_at=None, run_id=rid,
    )


# ---------- corruption survival ----------
def test_registry_survives_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", tmp_path / "reg.json")
    (tmp_path / "reg.json").write_text("{ not json")
    r = Registry()  # must not raise
    assert r.all() == {}
    # Original corrupt file should have been moved aside
    corrupt = list(tmp_path.glob("reg.json.corrupt.*"))
    assert len(corrupt) == 1


def test_registry_skips_bad_row_keeps_good_one(tmp_path, monkeypatch):
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", tmp_path / "reg.json")
    (tmp_path / "reg.json").write_text(
        '{"good":{"run_id":"good","name":"x","source_path":null,"note":null},'
        '"bad":{"run_id":"bad","NOT_A_FIELD":"x"}}'
    )
    r = Registry()
    assert "good" in r.all()
    assert "bad" not in r.all()


# ---------- ambiguous resolver ----------
def test_resolve_run_id_ambiguous_exits(monkeypatch):
    client = MagicMock()
    client.list_runs.return_value = [
        _fake_run("abcd1234-aaaa-aaaa-aaaa-000000000000:train:0"),
        _fake_run("abcd1234-bbbb-bbbb-bbbb-000000000000:train:0"),
    ]
    import typer
    with pytest.raises(typer.Exit):
        _resolve_run_id(client, "abcd1234")


def test_resolve_run_id_unique_prefix_ok(monkeypatch):
    client = MagicMock()
    client.list_runs.return_value = [
        _fake_run("abcd1234-aaaa-aaaa-aaaa-000000000000:train:0"),
        _fake_run("ef999999-bbbb-bbbb-bbbb-000000000000:train:0"),
    ]
    out = _resolve_run_id(client, "ef999")
    assert out.startswith("ef999")


# ---------- _resolve_path ----------
def test_resolve_path_full_uri_passthrough():
    client = MagicMock()
    p = "tinker://abcd/sampler_weights/final"
    assert _resolve_path(client, p) == p


def test_resolve_path_latest_for_run():
    client = MagicMock()
    rid = "abcd1234-08b6-5dc5-b927-63429a38f004:train:0"
    client.list_runs.return_value = [_fake_run(rid)]
    client.list_checkpoints.return_value = [
        _fake_ckpt(rid, "000010", "sampler", datetime(2026, 5, 1, tzinfo=timezone.utc)),
        _fake_ckpt(rid, "final", "sampler", datetime(2026, 5, 20, tzinfo=timezone.utc)),
    ]
    out = _resolve_path(client, "abcd1234")
    assert out.endswith("/sampler_weights/final")


def test_resolve_path_short_with_step():
    client = MagicMock()
    rid = "abcd1234-08b6-5dc5-b927-63429a38f004:train:0"
    client.list_runs.return_value = [_fake_run(rid)]
    client.list_checkpoints.return_value = [
        _fake_ckpt(rid, "000010", "sampler"),
        _fake_ckpt(rid, "000030", "sampler"),
    ]
    out = _resolve_path(client, "abcd1234:30")
    assert out.endswith("/sampler_weights/000030")


# ---------- scan: manual entry not clobbered (even on repeated rescans) ----------
def test_manual_entry_not_clobbered_on_repeated_scan(tmp_path, monkeypatch):
    rid = "abcd1234-08b6-5dc5-b927-63429a38f004"
    (tmp_path / "auto-name" / "Zlog" / rid).mkdir(parents=True)
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", tmp_path / "reg.json")
    monkeypatch.setattr(reg_mod, "SCAN_STAMP_PATH", tmp_path / "stamp")

    r = Registry()
    r.set(rid, "MANUAL")
    r.save()
    scan([tmp_path])
    scan([tmp_path])
    assert Registry().name_for(rid) == "MANUAL"


# ---------- _to_run / _to_ckpt against SDK-shaped fixture ----------
def test_to_ckpt_from_sdk_shape():
    raw = SimpleNamespace(
        checkpoint_id="sampler_weights/final",
        checkpoint_type="sampler",
        time=datetime(2026, 5, 20, tzinfo=timezone.utc),
        tinker_path="tinker://abcd1234-08b6-5dc5-b927-63429a38f004:train:0/sampler_weights/final",
        size_bytes=12345,
        public=False,
        expires_at=None,
    )
    c = _to_ckpt(raw)
    assert c.run_id == "abcd1234-08b6-5dc5-b927-63429a38f004:train:0"
    assert c.type == "sampler"
    assert c.size_bytes == 12345


def test_to_run_handles_null_last_checkpoint():
    raw = SimpleNamespace(
        training_run_id="abcd1234:train:0", base_model="x", model_owner="y",
        is_lora=True, lora_rank=8, corrupted=False,
        last_request_time=datetime(2026, 5, 20, tzinfo=timezone.utc),
        last_checkpoint=None, last_sampler_checkpoint=None, user_metadata=None,
    )
    r = _to_run(raw)
    assert r.last_checkpoint_path is None
    assert r.last_sampler_checkpoint_path is None


# ---------- use --no-verify still works ----------
def test_use_no_verify_writes_active(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, active as active_mod, cli as cli_mod2
    importlib.reload(config)
    importlib.reload(active_mod)
    importlib.reload(cli_mod2)
    # Mock the client; with --no-verify we shouldn't probe.
    p = "tinker://abcd1234-08b6-5dc5-b927-63429a38f004:train:0/sampler_weights/final"
    with patch("tinkpad.cli.TinkerClient", return_value=MagicMock()), \
         patch("tinkpad.cli._resolve_path", return_value=p), \
         patch("tinkpad.cli.probe_one") as mock_probe:
        res = runner.invoke(cli_mod2.app, ["use", "--no-verify", p])
    assert res.exit_code == 0
    mock_probe.assert_not_called()
