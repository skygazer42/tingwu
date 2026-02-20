import src.core.engine as engine_mod


def test_injection_hotwords_uses_context_plus_forced_union(monkeypatch):
    monkeypatch.setattr(engine_mod.settings, "hotword_injection_enable", True, raising=False)
    monkeypatch.setattr(engine_mod.settings, "hotword_injection_max", 10, raising=False)

    engine = engine_mod.TranscriptionEngine()
    engine._context_hotwords_list = ["Bar", "Baz"]
    engine._hotwords_list = ["Foo", "Bar"]

    # Context hotwords are preferred for safety, but injection is a hint-only path,
    # so we also include forced hotwords to improve proper-noun recall.
    assert engine._get_injection_hotwords() == "Bar\nBaz\nFoo"


def test_injection_hotwords_custom_overrides_everything(monkeypatch):
    monkeypatch.setattr(engine_mod.settings, "hotword_injection_enable", True, raising=False)
    engine = engine_mod.TranscriptionEngine()
    engine._context_hotwords_list = ["A"]
    engine._hotwords_list = ["B"]

    assert engine._get_injection_hotwords("X\nY") == "X\nY"


def test_injection_hotwords_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(engine_mod.settings, "hotword_injection_enable", False, raising=False)

    engine = engine_mod.TranscriptionEngine()
    engine._context_hotwords_list = ["A"]
    engine._hotwords_list = ["B"]

    assert engine._get_injection_hotwords() is None

