from tinkpad import active, config


def test_set_get_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TINKPAD_DIR", tmp_path)
    monkeypatch.setattr(config, "ACTIVE_PATH", tmp_path / "active")
    monkeypatch.setattr(active, "TINKPAD_DIR", tmp_path)
    monkeypatch.setattr(active, "ACTIVE_PATH", tmp_path / "active")

    p = "tinker://abcd1234-08b6-5dc5-b927-63429a38f004:train:0/sampler_weights/000030"
    active.set_active(p)
    assert active.get_active() == p
    env = (tmp_path / "active.env").read_text()
    assert p in env and "OPENAI_BASE_URL" in env

    active.clear_active()
    assert active.get_active() is None
