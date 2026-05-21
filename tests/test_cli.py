from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from typer.testing import CliRunner

from tinkpad.cli import app
from tinkpad.formatting import console as fmt_console
from tinkpad.tinker_client import Run, Checkpoint

runner = CliRunner()

# Force a wide terminal so Rich tables don't truncate columns.
fmt_console.width = 200


def _fake_run(rid="abcd1234-08b6-5dc5-b927-63429a38f004:train:0"):
    return Run(
        run_id=rid,
        base_model="openai/gpt-oss-120b",
        is_lora=True,
        lora_rank=32,
        corrupted=False,
        last_request_time=datetime(2026, 5, 20, tzinfo=timezone.utc),
        last_checkpoint_path=None,
        last_sampler_checkpoint_path=None,
    )


def _fake_ckpt(rid="abcd1234-08b6-5dc5-b927-63429a38f004:train:0", kind="sampler"):
    suffix = "sampler_weights/final" if kind == "sampler" else "weights/final"
    return Checkpoint(
        checkpoint_id=suffix,
        type=kind,
        tinker_path=f"tinker://{rid}/{suffix}",
        size_bytes=5_244_480_826,
        created_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        public=False,
        expires_at=None,
        run_id=rid,
    )


def test_version():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "tinkpad" in res.stdout


def test_runs_lists_with_mocked_client(monkeypatch):
    mock = MagicMock()
    mock.list_runs.return_value = [_fake_run()]
    with patch("tinkpad.cli.TinkerClient", return_value=mock), \
         patch("tinkpad.cli.scan", return_value=[]):
        res = runner.invoke(app, ["runs"])
    assert res.exit_code == 0
    assert "gpt-oss-120b" in res.stdout


def test_ls_with_mocked_client(monkeypatch):
    mock = MagicMock()
    mock.list_runs.return_value = [_fake_run()]
    mock.list_checkpoints.return_value = [_fake_ckpt(kind="sampler"), _fake_ckpt(kind="training")]
    with patch("tinkpad.cli.TinkerClient", return_value=mock), \
         patch("tinkpad.cli.scan", return_value=[]):
        res = runner.invoke(app, ["ls"])
    assert res.exit_code == 0
    assert "sampler" in res.stdout and "training" in res.stdout


def test_use_and_active(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    # Reload modules so they pick up the new TINKPAD_DIR
    import importlib
    from tinkpad import config, active as active_mod, cli as cli_mod
    importlib.reload(config)
    importlib.reload(active_mod)
    importlib.reload(cli_mod)

    path = "tinker://abcd1234-08b6-5dc5-b927-63429a38f004:train:0/sampler_weights/final"
    res = runner.invoke(cli_mod.app, ["use", path])
    assert res.exit_code == 0
    res2 = runner.invoke(cli_mod.app, ["active"])
    assert res2.exit_code == 0
    assert path in res2.stdout


def test_reg_set_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKPAD_DIR", str(tmp_path))
    import importlib
    from tinkpad import config, registry, cli as cli_mod
    importlib.reload(config)
    importlib.reload(registry)
    importlib.reload(cli_mod)

    rid = "abcd1234-08b6-5dc5-b927-63429a38f004:train:0"
    runner.invoke(cli_mod.app, ["reg", "set", rid, "my-exp"])
    res = runner.invoke(cli_mod.app, ["reg", "list"])
    assert "my-exp" in res.stdout
