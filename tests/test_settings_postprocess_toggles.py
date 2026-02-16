def test_settings_postprocess_toggles_can_be_set_via_env(monkeypatch):
    from src.config import Settings

    monkeypatch.setenv("SPOKEN_PUNC_ENABLE", "true")
    monkeypatch.setenv("ACRONYM_MERGE_ENABLE", "true")

    s = Settings()

    assert hasattr(s, "spoken_punc_enable")
    assert hasattr(s, "acronym_merge_enable")
    assert s.spoken_punc_enable is True
    assert s.acronym_merge_enable is True

