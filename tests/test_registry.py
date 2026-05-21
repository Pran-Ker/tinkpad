from pathlib import Path

import json
import pytest

from tinkpad import registry as reg_mod
from tinkpad.registry import Registry, scan, _normalize


def test_normalize_strips_prefix_path_and_train_suffix():
    rid = "tinker://abcd1234-08b6-5dc5-b927-63429a38f004:train:0/sampler_weights/000030"
    assert _normalize(rid) == "abcd1234-08b6-5dc5-b927-63429a38f004"
    # All three forms collapse to the same key.
    assert _normalize("abcd1234-08b6-5dc5-b927-63429a38f004:train:0") == "abcd1234-08b6-5dc5-b927-63429a38f004"
    assert _normalize("abcd1234-08b6-5dc5-b927-63429a38f004") == "abcd1234-08b6-5dc5-b927-63429a38f004"


def test_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "reg.json"
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", p)
    r = Registry(p)
    r.set("abcd1234-08b6-5dc5-b927-63429a38f004:train:0", "myexp")
    r.save()
    r2 = Registry(p)
    assert r2.name_for("abcd1234-08b6-5dc5-b927-63429a38f004:train:0") == "myexp"


def test_scan_zlog_convention(tmp_path, monkeypatch):
    # Build a fake project tree:
    # tmp/proj-foo/Zlog/abcd1234-08b6-5dc5-b927-63429a38f004/run.log
    proj = tmp_path / "proj-foo"
    rid = "abcd1234-08b6-5dc5-b927-63429a38f004"
    (proj / "Zlog" / rid).mkdir(parents=True)
    (proj / "Zlog" / rid / "run.log").write_text("hi")

    reg_path = tmp_path / "reg.json"
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", reg_path)

    found = scan([tmp_path])
    assert any(e.run_id == rid and e.name == "proj-foo" for e in found)
    r = Registry(reg_path)
    assert r.name_for(rid) == "proj-foo"


def test_scan_folder_named_run_id(tmp_path, monkeypatch):
    rid = "abcd1234-08b6-5dc5-b927-63429a38f004"
    proj = tmp_path / "experiments" / "myexp" / rid
    proj.mkdir(parents=True)

    reg_path = tmp_path / "reg.json"
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", reg_path)
    found = scan([tmp_path])
    names = {(e.run_id, e.name) for e in found}
    assert (rid, "myexp") in names


def test_manual_set_not_clobbered_by_scan(tmp_path, monkeypatch):
    rid = "abcd1234-08b6-5dc5-b927-63429a38f004"
    proj = tmp_path / "auto-name" / "Zlog" / rid
    proj.mkdir(parents=True)
    reg_path = tmp_path / "reg.json"
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", reg_path)

    # set manually first
    r = Registry(reg_path)
    r.set(rid, "MANUAL_NAME")
    r.save()

    scan([tmp_path])
    r2 = Registry(reg_path)
    assert r2.name_for(rid) == "MANUAL_NAME"
