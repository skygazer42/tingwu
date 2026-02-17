def test_settings_parse_external_diarizer_env(monkeypatch):
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_ENABLE", "true")
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_BASE_URL", "http://diarizer:8000")
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_TIMEOUT_S", "12.5")
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_MAX_TURN_DURATION_S", "15")
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_MAX_TURNS", "123")
    from src.config import Settings
    s = Settings()
    assert s.speaker_external_diarizer_enable is True
    assert s.speaker_external_diarizer_base_url == "http://diarizer:8000"
    assert s.speaker_external_diarizer_timeout_s == 12.5
    assert s.speaker_external_diarizer_max_turn_duration_s == 15.0
    assert s.speaker_external_diarizer_max_turns == 123
