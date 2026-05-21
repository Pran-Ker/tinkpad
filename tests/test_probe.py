from unittest.mock import patch, MagicMock

import tinker

from tinkpad import probe


def test_skipped_for_training_ckpt():
    r = probe.probe("tinker://abcd/weights/000010", key="tml-fake")
    assert r.status == "skipped"


def test_probe_ok(monkeypatch):
    # Fake the SDK chain: ServiceClient -> SamplingClient -> tokenizer + sample()
    fake_tok = MagicMock()
    fake_tok.encode.return_value = [1, 2, 3]
    fake_tok.decode.return_value = "OK"
    fake_sample_future = MagicMock()
    fake_sample_future.result.return_value = MagicMock(sequences=[MagicMock(tokens=[7, 8])])
    fake_sclient = MagicMock()
    fake_sclient.get_tokenizer.return_value = fake_tok
    fake_sclient.sample.return_value = fake_sample_future
    fake_sc = MagicMock()
    fake_sc.create_sampling_client.return_value = fake_sclient
    with patch("tinkpad.probe.tinker.ServiceClient", return_value=fake_sc):
        r = probe.probe("tinker://abcd/sampler_weights/final", key="tml-fake")
    assert r.status == "ok"
    assert r.sample == "OK"
    assert r.latency_ms is not None


def test_probe_fail_on_exception(monkeypatch):
    fake_sc = MagicMock()
    fake_sc.create_sampling_client.side_effect = RuntimeError("boom")
    with patch("tinkpad.probe.tinker.ServiceClient", return_value=fake_sc):
        r = probe.probe("tinker://abcd/sampler_weights/final", key="tml-fake")
    assert r.status == "fail"
    assert "boom" in r.error


def test_probe_no_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    r = probe.probe("tinker://abcd/sampler_weights/final")
    assert r.status == "fail"
    assert "TINKER_API_KEY" in r.error
