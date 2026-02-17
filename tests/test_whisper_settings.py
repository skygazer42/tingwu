def test_settings_accepts_whisper_backend(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "whisper")

    from src.config import Settings

    s = Settings()
    assert s.asr_backend == "whisper"

