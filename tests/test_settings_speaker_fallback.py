def test_settings_speaker_fallback_env(monkeypatch):
    from src.config import Settings

    monkeypatch.setenv("SPEAKER_FALLBACK_DIARIZATION_ENABLE", "true")
    monkeypatch.setenv("SPEAKER_FALLBACK_DIARIZATION_BASE_URL", "http://example:8000")
    monkeypatch.setenv("SPEAKER_FALLBACK_DIARIZATION_TIMEOUT_S", "12.5")
    monkeypatch.setenv("SPEAKER_FALLBACK_MAX_TURN_DURATION_S", "15")
    monkeypatch.setenv("SPEAKER_FALLBACK_MAX_TURNS", "123")

    s = Settings()

    assert s.speaker_fallback_diarization_enable is True
    assert s.speaker_fallback_diarization_base_url == "http://example:8000"
    assert s.speaker_fallback_diarization_timeout_s == 12.5
    assert s.speaker_fallback_max_turn_duration_s == 15.0
    assert s.speaker_fallback_max_turns == 123

