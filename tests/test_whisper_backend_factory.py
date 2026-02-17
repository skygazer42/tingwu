def test_get_backend_whisper_returns_backend_instance():
    from src.models.backends import get_backend
    from src.models.backends.whisper import WhisperBackend

    backend = get_backend(backend_type="whisper", model="small", device="cpu")
    assert isinstance(backend, WhisperBackend)
    assert backend.supports_speaker is False
